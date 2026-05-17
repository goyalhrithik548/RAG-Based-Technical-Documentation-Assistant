from typing import Literal

from pydantic import BaseModel, Field

from app import config


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=8)


class SourceItem(BaseModel):
    title: str
    source: str
    type: str


class RetrievedChunk(BaseModel):
    source: str
    content: str
    score: float | None = None
    grade: str | None = None
    chunk_id: str | None = None


class QueryResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    sources: list[SourceItem]
    retrieved_chunks: list[RetrievedChunk]
    used_web_search: bool
    retry_count: int


class DocumentItem(BaseModel):
    name: str
    path: str
    type: str
    chunk_count: int = 0
    indexed: bool = False


class IngestResponse(BaseModel):
    message: str
    document_count: int
    chunk_count: int
    index_path: str
    documents: list[DocumentItem]


class DocumentListResponse(BaseModel):
    index_available: bool
    documents: list[DocumentItem]


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=3)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)
    sources: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    message: str
    saved_to: str
