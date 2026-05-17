import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
INDEX_DIR = BASE_DIR / "faiss_index"
METADATA_PATH = INDEX_DIR / "metadata.json"
FEEDBACK_PATH = BASE_DIR / "feedback.jsonl"

GEMINI_MODEL = "gemini-3-flash-preview"
EMBEDDING_MODEL = "models/gemini-embedding-2"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 4
MAX_RETRIEVAL_RETRY = 1
MAX_WEB_RESULTS = 3
MAX_ANSWER_TOKENS = 500

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def ensure_directories() -> None:
    DOCS_DIR.mkdir(exist_ok=True)


def has_google_api_key() -> bool:
    return bool(GOOGLE_API_KEY)


def has_tavily_api_key() -> bool:
    return bool(TAVILY_API_KEY)
