import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app import config


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Initialize Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.GOOGLE_API_KEY,
    )


def _read_text_file(file_path: Path) -> str:
    """Read TXT/MD files safely."""
    return file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def _read_pdf_file(file_path: Path) -> str:
    """Extract text from PDF."""
    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents_from_folder() -> list[Document]:
    """
    Load all supported documents from docs/ folder.
    """
    config.ensure_directories()

    documents: list[Document] = []

    for file_path in sorted(config.DOCS_DIR.iterdir()):

        if (
            not file_path.is_file()
            or file_path.suffix.lower()
            not in config.SUPPORTED_EXTENSIONS
        ):
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                text = _read_pdf_file(file_path)
            else:
                text = _read_text_file(file_path)

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path.resolve()),
                        "type": file_path.suffix.lower().replace(".", ""),
                    },
                )
            )

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    chunk_counts: Counter[str] = Counter()

    for chunk in chunks:
        source = chunk.metadata["source"]

        chunk_counts[source] += 1

        chunk.metadata["chunk_id"] = (
            f"{source}-chunk-{chunk_counts[source]}"
        )

    return chunks


def ingest_documents() -> dict:
    """
    Ingest documents into FAISS vector database.
    """

    if not config.has_google_api_key():
        raise ValueError("GOOGLE_API_KEY is missing.")

    documents = load_documents_from_folder()

    if not documents:
        raise ValueError("No supported documents found in docs/.")

    chunks = split_documents(documents)

    embeddings_model = _get_embeddings()

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    # -----------------------------
    # Batch Embedding
    # -----------------------------

    try:
        vectors = embeddings_model.embed_documents(texts)

    except Exception as e:

        print(f"Batch embedding failed: {e}")

        vectors = []

    # -----------------------------
    # Length mismatch fix
    # -----------------------------

    if len(vectors) != len(texts):

        print("Falling back to one-by-one embeddings...")

        vectors = []

        for text in texts:

            try:
                vector = embeddings_model.embed_query(text)
                vectors.append(vector)

            except Exception as e:
                print(f"Embedding failed for chunk: {e}")

    # -----------------------------
    # Final safety check
    # -----------------------------

    if len(vectors) != len(texts):

        raise ValueError(
            f"Embedding count mismatch. "
            f"Texts: {len(texts)}, "
            f"Vectors: {len(vectors)}"
        )

    # -----------------------------
    # Create FAISS Vector Store
    # -----------------------------

    vector_store = FAISS.from_embeddings(
        list(zip(texts, vectors)),
        embeddings_model,
        metadatas=metadatas,
    )

    # -----------------------------
    # Save Index
    # -----------------------------

    config.INDEX_DIR.mkdir(exist_ok=True)

    vector_store.save_local(str(config.INDEX_DIR))

    # -----------------------------
    # Metadata
    # -----------------------------

    chunk_counts: Counter[str] = Counter(
        chunk.metadata["source"]
        for chunk in chunks
    )

    indexed_docs = []

    for doc in documents:

        indexed_docs.append(
            {
                "name": doc.metadata["source"],
                "path": doc.metadata["path"],
                "type": doc.metadata["type"],
                "chunk_count": chunk_counts[
                    doc.metadata["source"]
                ],
                "indexed": True,
            }
        )

    metadata = {
        "indexed_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "document_count": len(documents),

        "chunk_count": len(chunks),

        "documents": indexed_docs,
    }

    config.METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # -----------------------------
    # Final Response
    # -----------------------------

    return {
        "message": "Ingested successfully.",
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": indexed_docs,
        "indexed": True,
	"index_path": str(config.INDEX_DIR.resolve())
    }


def is_index_available() -> bool:
    """
    Check if FAISS index exists.
    """

    return (
        config.INDEX_DIR.exists()
        and (config.INDEX_DIR / "index.faiss").exists()
    )


def load_vector_store() -> FAISS:
    """
    Load FAISS vector store.
    """

    return FAISS.load_local(
        str(config.INDEX_DIR),
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def list_documents() -> list[dict]:
    """
    List indexed documents.
    """

    if config.METADATA_PATH.exists():

        metadata = json.loads(
            config.METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        return metadata.get("documents", [])

    return []