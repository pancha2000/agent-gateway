"""Pydantic request/response models (the API contract for your local PC)."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Scope / project names: short and boring on purpose (they end up in SQL parameters
# and in prompts, so no whitespace or odd characters).
SCOPE_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"
Scope = Annotated[str, Field(pattern=SCOPE_PATTERN)]


def _not_blank(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be blank")
    return v


# ------------------------------------------------------------------ memory
class SaveMemoryIn(BaseModel):
    text: str = Field(min_length=1, max_length=20_000, description="Snippet / doc text to remember")
    vector: list[float] = Field(description="Embedding computed locally (length must equal EMBED_DIM)")
    kind: str = Field(default="snippet", max_length=32)
    language: str | None = Field(default=None, max_length=32)
    source: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: Scope = Field(default="shared", description="'shared' or a private agent/role scope")
    project_id: str | None = Field(default=None, pattern=SCOPE_PATTERN)
    parent_id: int | None = Field(default=None, ge=1)
    chunk_index: int | None = Field(default=None, ge=0)
    title: str | None = Field(default=None, max_length=300)

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return _not_blank(v)


class SaveMemoryOut(BaseModel):
    id: int
    created: bool = Field(description="False if identical text already existed and was updated")


class BatchItem(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    vector: list[float]
    kind: str | None = Field(default=None, max_length=32)
    language: str | None = Field(default=None, max_length=32)
    source: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int | None = Field(default=None, ge=0)
    title: str | None = Field(default=None, max_length=300)

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return _not_blank(v)


class BatchParent(BaseModel):
    """The document record that the chunks of a batch belong to (kind='doc')."""
    text: str = Field(min_length=1, max_length=20_000)
    vector: list[float]
    title: str | None = Field(default=None, max_length=300)
    source: str | None = Field(default=None, max_length=512)
    kind: str = Field(default="doc", max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        return _not_blank(v)


class SaveBatchIn(BaseModel):
    items: list[BatchItem] = Field(min_length=1, max_length=40)
    parent: BatchParent | None = None
    parent_id: int | None = Field(default=None, ge=1, description="Use an existing document row instead of `parent`")
    kind: str = Field(default="chunk", max_length=32, description="Default kind for items without their own")
    scope: Scope = "shared"
    project_id: str | None = Field(default=None, pattern=SCOPE_PATTERN)
    source: str | None = Field(default=None, max_length=512)
    language: str | None = Field(default=None, max_length=32)


class SaveBatchOut(BaseModel):
    parent_id: int | None
    ids: list[int]
    created: list[bool]


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
    scope: Scope = Field(default="shared", description="Caller's scope: sees 'shared' + this scope")
    all_scopes: bool = Field(default=False, description="Main agent: ignore scopes, see everything")
    project_id: str | None = Field(default=None, pattern=SCOPE_PATTERN,
                                   description="Limit to this project + untagged general knowledge")
    snippet_chars: int | None = Field(default=None, ge=50, le=5000,
                                      description="Return only the first N chars of each hit (a 'hint'); "
                                                  "use /recall/{id} for the full text")


class MemoryHit(BaseModel):
    id: int
    similarity: float
    content: str
    kind: str
    language: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: str = "shared"
    project_id: str | None = None
    parent_id: int | None = None
    chunk_index: int | None = None
    title: str | None = None


class PastError(BaseModel):
    id: int
    similarity: float
    task: str
    error_message: str
    root_cause: str | None = None
    lesson: str | None = None
    fix: str | None = None
    occurrences: int
    attempts: int = 1
    resolved: bool = False


class GetHintOut(BaseModel):
    hits: list[MemoryHit]
    past_errors: list[PastError]
    context: str = Field(description="Prompt-ready text block, trimmed to max_chars")


class MemoryOut(BaseModel):
    id: int
    content: str
    kind: str
    language: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: str = "shared"
    project_id: str | None = None
    parent_id: int | None = None
    chunk_index: int | None = None
    title: str | None = None
    created_at: int | None = None
    updated_at: int | None = None


class RecallOut(BaseModel):
    memory: MemoryOut
    children: list[MemoryOut] = Field(default_factory=list)


class DeleteOut(BaseModel):
    deleted: list[int]


class StatsRow(BaseModel):
    scope: str
    kind: str
    n: int


class StatsErrors(BaseModel):
    total: int
    unresolved: int


class StatsOut(BaseModel):
    memories: list[StatsRow]
    errors: StatsErrors


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
    attempt: int = Field(default=1, ge=1, le=50, description="Which retry this report is about")
    resolved: bool = Field(default=False, description="True if a working fix was confirmed")


class ErrorLogOut(BaseModel):
    id: int
    occurrences: int
    is_repeat: bool = Field(description="True if this exact mistake was logged before")
    attempts: int = 1
    resolved: bool = False


class ErrorResolveIn(BaseModel):
    fix: str | None = Field(default=None, max_length=8000)
    lesson: str | None = Field(default=None, max_length=4000)


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


# ------------------------------------------------------- ingest / research
class PreparedChunk(BaseModel):
    index: int
    text: str
    exists: bool = Field(description="Already stored in the target scope: no need to embed/save it again")


class PrepareIn(BaseModel):
    url: str | None = Field(default=None, max_length=2000, description="Web page or PDF to fetch")
    text: str | None = Field(default=None, max_length=200_000, description="Raw text/code to chunk")
    title: str | None = Field(default=None, max_length=300)
    source: str | None = Field(default=None, max_length=512)
    scope: Scope = Field(default="shared", description="Scope the chunks WILL be saved in (for the `exists` check)")
    chunk_chars: int | None = Field(default=None, ge=200, le=4000)
    overlap: int | None = Field(default=None, ge=0, le=500)
    max_chunks: int | None = Field(default=None, ge=1, le=300)

    @model_validator(mode="after")
    def _exactly_one(self) -> "PrepareIn":
        if bool(self.url) == bool(self.text):
            raise ValueError("send exactly one of `url` or `text`")
        return self


class PrepareOut(BaseModel):
    title: str
    source: str
    kind: str = Field(description="html | text | pdf")
    parent_text: str = Field(description="Embed this and save it as the document (kind='doc') record")
    total_chunks: int
    truncated: bool = Field(description="True if the document had more chunks than max_chunks")
    chunks: list[PreparedChunk]


class ResearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    vector: list[float] | None = Field(default=None, description="Embedding of the query (for the memory check)")
    scope: Scope = "shared"
    all_scopes: bool = False
    project_id: str | None = Field(default=None, pattern=SCOPE_PATTERN)
    save_scope: Scope = Field(default="shared", description="Scope the new web chunks will be saved in")
    top_k: int = Field(default=4, ge=1, le=10)
    min_similarity: float = Field(default=0.25, ge=-1.0, le=1.0)
    web_threshold: float = Field(default=0.6, ge=0.0, le=1.0,
                                 description="If the best memory hit is at least this similar, skip the web")
    force_web: bool = False
    include_errors: bool = True
    max_results: int = Field(default=5, ge=1, le=10)
    fetch_top: int = Field(default=3, ge=1, le=4)
    max_chars: int = Field(default=7000, ge=500, le=40_000)


class ResearchPage(BaseModel):
    url: str
    title: str
    kind: str
    parent_text: str
    total_chunks: int
    truncated: bool
    chunks: list[PreparedChunk]


class OtherResult(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchOut(BaseModel):
    source: str = Field(description="'memory' (answered from the database) or 'web' (fresh pages fetched)")
    best_similarity: float
    engine: str | None = None
    hits: list[MemoryHit]
    past_errors: list[PastError]
    pages: list[ResearchPage]
    other_results: list[OtherResult]
    context: str
    note: str | None = None


# --------------------------------------------------------------- escalation
class Attempt(BaseModel):
    attempt: int = Field(default=1, ge=1, le=50)
    approach: str = Field(max_length=3000)
    result: str | None = Field(default=None, max_length=3000)


class EscalationIn(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    error_message: str = Field(min_length=1, max_length=8000)
    code: str | None = Field(default=None, max_length=20_000)
    attempts: list[Attempt] = Field(default_factory=list, max_length=10)
    vector: list[float] | None = None
    scope: Scope = "shared"
    all_scopes: bool = False
    project_id: str | None = Field(default=None, pattern=SCOPE_PATTERN)
    include_memory: bool = Field(default=True, description="False = do not attach database content (privacy)")
    top_k: int = Field(default=4, ge=1, le=10)
    max_chars: int = Field(default=12_000, ge=2000, le=60_000)


class EscalationOut(BaseModel):
    markdown: str
    filename: str
    memory_hits: int
