import json
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.ingestion import ingest_documents, is_index_available, list_documents
from app.rag_graph import run_query
from app.schemas import (
    DocumentListResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

app = FastAPI(
    title="Technical Documentation Assistant",
    description="Minimal RAG assistant built with FastAPI, LangGraph, FAISS, and Gemini.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    config.ensure_directories()


@app.post("/ingest", response_model=IngestResponse)
def ingest() -> IngestResponse:
    try:
        result = ingest_documents()
        return IngestResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@app.get("/documents", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    return DocumentListResponse(
        index_available=is_index_available(),
        documents=list_documents(),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not config.has_google_api_key():
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is missing. Add it to your .env file first.")

    if not is_index_available():
        raise HTTPException(
            status_code=400,
            detail="No FAISS index found. Add files to docs/ and call /ingest first.",
        )

    try:
        result = run_query(question=request.question.strip(), top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    payload = request.model_dump()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()

    with config.FEEDBACK_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")

    return FeedbackResponse(
        message="Feedback saved successfully.",
        saved_to=str(config.FEEDBACK_PATH.resolve()),
    )
