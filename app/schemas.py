"""Pydantic request/response models (the API contract for your local PC)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ memory
class SaveMemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000, description="Snippet / doc text to remember")
    vector: list[float] = Field(description="Embedding computed locally (length must equal EMBED_DIM)")
    kind: str = Field(default="snippet", max_length=32)
    language: str | None = Field(default=None, max_length=32)
    source: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SaveMemoryOut(BaseModel):
    id: int
    created: bool = Field(description="False if identical text already existed and was updated")


class GetHintIn(BaseModel):
    vector: list[float] = Field(description="Query embedding (length must equal EMBED_DIM)")
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0,
                                  description="Drop hits below this cosine similarity")
    kind: str | None = None
    language: str | None = None
    include_errors: bool = Field(default=True, description="Also return similar past mistakes")
    max_chars: int = Field(default=6000, ge=500, le=40_000,
                           description="Budget for the ready-to-paste `context` string")


class MemoryHit(BaseModel):
    id: int
    similarity: float
    content: str
    kind: str
    language: str | None
    source: str | None
    metadata: dict[str, Any]


class PastError(BaseModel):
    id: int
    similarity: float
    task: str
    error_message: str
    root_cause: str | None
    lesson: str | None
    fix: str | None
    occurrences: int


class GetHintOut(BaseModel):
    hits: list[MemoryHit]
    past_errors: list[PastError]
    context: str = Field(description="Prompt-ready text block, trimmed to max_chars")


# ------------------------------------------------------------------ errors
class ErrorLogIn(BaseModel):
    task: str = Field(min_length=1, max_length=4000, description="What the AI was trying to do")
    error_message: str = Field(min_length=1, max_length=8000)
    root_cause: str | None = Field(default=None, max_length=4000)
    lesson: str | None = Field(default=None, max_length=4000, description="What to do differently")
    fix: str | None = Field(default=None, max_length=8000)
    agent: str | None = Field(default=None, max_length=64)
    tags: list[str] = Field(default_factory=list, max_length=20)
    vector: list[float] | None = Field(default=None, description="Optional embedding of task+error")


class ErrorLogOut(BaseModel):
    id: int
    occurrences: int
    is_repeat: bool = Field(description="True if this exact mistake was logged before")


# --------------------------------------------------------------- web search
class SearchWebIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=5, ge=1, le=10)
    fetch_top: int = Field(default=0, ge=0, le=3,
                           description="Also download and extract text from the top N pages")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    content: str | None = None


class SearchWebOut(BaseModel):
    query: str
    engine: str
    cached: bool
    results: list[SearchResult]
    context: str
