"""Memory endpoints: save (one / batch), search (get_hint), recall, delete, stats."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from .. import repo, vectors
from ..config import get_settings
from ..context import build_hint_context
from ..schemas import (
    SCOPE_PATTERN, DeleteOut, GetHintIn, GetHintOut, RecallOut, SaveBatchIn, SaveBatchOut,
    SaveMemoryIn, SaveMemoryOut, StatsOut,
)
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])

MAX_METADATA_JSON = 10_000


def _check_metadata(meta: dict, field: str = "metadata") -> None:
    if len(json.dumps(meta)) > MAX_METADATA_JSON:
        raise HTTPException(422, f"{field} too large (max 10 KB of JSON)")


@router.post("/save_memory", response_model=SaveMemoryOut)
async def save_memory(body: SaveMemoryIn, request: Request) -> SaveMemoryOut:
    """Store text + locally computed embedding. Saving identical text again just updates it."""
    settings = get_settings()
    vectors.validate(body.vector, settings.embed_dim)
    _check_metadata(body.metadata)

    memory_id, created = await repo.save_memory(
        request.app.state.turso,
        text=body.text, vector=body.vector, kind=body.kind,
        language=body.language, source=body.source, metadata=body.metadata,
        scope=body.scope, project_id=body.project_id, parent_id=body.parent_id,
        chunk_index=body.chunk_index, title=body.title,
    )
    return SaveMemoryOut(id=memory_id, created=created)


@router.post("/save_memory_batch", response_model=SaveBatchOut)
async def save_memory_batch(body: SaveBatchIn, request: Request) -> SaveBatchOut:
    """
    Save up to 40 chunks in one call. Send `parent` (the document record) to have it
    saved first, so every chunk gets its id as `parent_id`. Re-sending an already
    stored text just refreshes it (`created` is false for those).
    """
    settings = get_settings()
    db = request.app.state.turso

    if body.parent:
        vectors.validate(body.parent.vector, settings.embed_dim, "parent.vector")
        _check_metadata(body.parent.metadata, "parent.metadata")
    for i, it in enumerate(body.items):
        vectors.validate(it.vector, settings.embed_dim, f"items[{i}].vector")
        _check_metadata(it.metadata, f"items[{i}].metadata")

    parent_id = body.parent_id
    if body.parent:
        parent_id, _ = await repo.save_memory(
            db, text=body.parent.text, vector=body.parent.vector, kind=body.parent.kind,
            language=None, source=body.parent.source or body.source,
            metadata=body.parent.metadata, scope=body.scope, project_id=body.project_id,
            title=body.parent.title,
        )

    items = [{
        "text": it.text, "vector": it.vector, "kind": it.kind,
        "language": it.language or body.language, "source": it.source or body.source,
        "metadata": it.metadata, "chunk_index": it.chunk_index, "title": it.title,
    } for it in body.items]

    results = await repo.save_memories(
        db, items, scope=body.scope, project_id=body.project_id,
        parent_id=parent_id, kind=body.kind,
    )
    return SaveBatchOut(
        parent_id=parent_id, ids=[i for i, _ in results], created=[c for _, c in results])


@router.post("/get_hint", response_model=GetHintOut)
async def get_hint(body: GetHintIn, request: Request) -> GetHintOut:
    """
    Cosine-similarity search. Returns the structured hits AND a `context` string
    you can paste straight into the local model's prompt.

    Scope: a caller sees 'shared' + its own `scope`; the main agent sends
    all_scopes=true. Set `snippet_chars` to get short hints and fetch the full text
    later with GET /recall/{id}.
    """
    settings = get_settings()
    vectors.validate(body.vector, settings.embed_dim)

    hits, errors = await repo.get_hint(
        request.app.state.turso, settings,
        vector=body.vector, top_k=body.top_k, min_similarity=body.min_similarity,
        kind=body.kind, language=body.language, include_errors=body.include_errors,
        scope=body.scope, all_scopes=body.all_scopes, project_id=body.project_id,
    )
    if body.snippet_chars:
        n = body.snippet_chars
        for h in hits:
            if len(h["content"]) > n:
                h["content"] = h["content"][:n].rstrip() + f" … [truncated, GET /recall/{h['id']} for full text]"
    return GetHintOut(
        hits=hits, past_errors=errors,
        context=build_hint_context(hits, errors, body.max_chars),
    )


@router.get("/recall/{memory_id}", response_model=RecallOut)
async def recall(
    memory_id: int, request: Request,
    scope: str = Query("shared", pattern=SCOPE_PATTERN),
    all_scopes: bool = False,
    children: bool = Query(False, description="Also return the chunks that belong to this document"),
    limit: int = Query(50, ge=1, le=200),
) -> RecallOut:
    """Full text of one memory by id (what a short hint points to)."""
    db = request.app.state.turso
    memory = await repo.get_memory(db, memory_id, scope=scope, all_scopes=all_scopes)
    if memory is None:
        raise HTTPException(404, "memory not found (or not visible in this scope)")
    kids = []
    if children:
        kids = await repo.get_children(db, memory_id, scope=scope, all_scopes=all_scopes, limit=limit)
    return RecallOut(memory=memory, children=kids)


@router.delete("/memory/{memory_id}", response_model=DeleteOut)
async def delete_memory(
    memory_id: int, request: Request,
    scope: str = Query("shared", pattern=SCOPE_PATTERN),
    all_scopes: bool = False,
) -> DeleteOut:
    """Delete a memory and, if it is a document record, all of its chunks."""
    deleted = await repo.delete_memory(
        request.app.state.turso, memory_id, scope=scope, all_scopes=all_scopes)
    if not deleted:
        raise HTTPException(404, "memory not found (or not visible in this scope)")
    return DeleteOut(deleted=deleted)


@router.get("/stats", response_model=StatsOut)
async def stats(request: Request) -> StatsOut:
    """Row counts per scope/kind and mistake counts (reads the whole table: use sparingly)."""
    return StatsOut(**await repo.stats(request.app.state.turso))
