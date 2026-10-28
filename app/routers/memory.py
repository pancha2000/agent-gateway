"""POST /save_memory and POST /get_hint"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import repo, vectors
from ..config import get_settings
from ..schemas import GetHintIn, GetHintOut, SaveMemoryIn, SaveMemoryOut
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/save_memory", response_model=SaveMemoryOut)
async def save_memory(body: SaveMemoryIn, request: Request) -> SaveMemoryOut:
    """Store text + locally computed embedding. Saving identical text again just updates it."""
    settings = get_settings()
    vectors.validate(body.vector, settings.embed_dim)
    if len(json.dumps(body.metadata)) > 10_000:
        raise HTTPException(422, "metadata too large (max 10 KB of JSON)")

    memory_id, created = await repo.save_memory(
        request.app.state.turso,
        text=body.text, vector=body.vector, kind=body.kind,
        language=body.language, source=body.source, metadata=body.metadata,
    )
    return SaveMemoryOut(id=memory_id, created=created)


@router.post("/get_hint", response_model=GetHintOut)
async def get_hint(body: GetHintIn, request: Request) -> GetHintOut:
    """
    Cosine-similarity search. Returns the structured hits AND a `context` string
    you can paste straight into the local model's prompt.
    """
    settings = get_settings()
    vectors.validate(body.vector, settings.embed_dim)

    hits, errors = await repo.get_hint(
        request.app.state.turso, settings,
        vector=body.vector, top_k=body.top_k, min_similarity=body.min_similarity,
        kind=body.kind, language=body.language, include_errors=body.include_errors,
    )
    return GetHintOut(
        hits=hits, past_errors=errors,
        context=_build_context(hits, errors, body.max_chars),
    )


def _build_context(hits: list[dict], errors: list[dict], budget: int) -> str:
    """Format hits/errors into one prompt block, never exceeding `budget` chars."""
    parts: list[str] = []
    if hits:
        parts.append("### Relevant code & notes from memory")
        for i, h in enumerate(hits, 1):
            tag = ", ".join(x for x in (h["kind"], h["language"], h["source"]) if x)
            parts.append(f"[{i}] ({tag}; similarity {h['similarity']:.2f})\n{h['content']}")
    if errors:
        parts.append("### Past mistakes to avoid")
        for e in errors:
            line = f"- Task: {e['task']}\n  Error: {e['error_message']}"
            if e.get("lesson"):
                line += f"\n  Lesson: {e['lesson']}"
            if e.get("fix"):
                line += f"\n  Fix: {e['fix']}"
            if e["occurrences"] > 1:
                line += f"\n  (repeated {e['occurrences']} times!)"
            parts.append(line)
    return "\n\n".join(parts)[:budget]
