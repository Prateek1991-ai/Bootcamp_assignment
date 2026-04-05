"""
Shared Pydantic data models.
Defines the canonical schema for chunks, API requests, and responses.
"""

from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Chunk types ───────────────────────────────────────────────────────────────

class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentChunk(BaseModel):
    """A single extracted and embedded unit from a PDF document."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    source_file: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="1-based page number")
    chunk_type: ChunkType = Field(..., description="text | table | image")
    content: str = Field(
        ..., description="Raw text, table markdown, or image caption"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata (image path, table html, etc.)",
    )


# ── API request models ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language question to answer from the indexed documents",
        examples=["What was the revenue growth rate reported in Q4 2023?"],
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (overrides server default)",
    )


# ── API response models ───────────────────────────────────────────────────────

class SourceReference(BaseModel):
    chunk_id: str
    source_file: str
    page_number: int
    chunk_type: ChunkType
    content_preview: str = Field(..., description="First 200 chars of chunk content")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceReference]
    chunks_retrieved: int
    model_used: str


class IngestionSummary(BaseModel):
    filename: str
    text_chunks: int
    table_chunks: int
    image_chunks: int
    total_chunks: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    model: str
    embedding_model: str
    indexed_documents: int
    total_chunks: int
    uptime_seconds: float


class DocumentListResponse(BaseModel):
    documents: list[str]
    total_documents: int


class DeleteResponse(BaseModel):
    message: str
    filename: str
    chunks_removed: int
