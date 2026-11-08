"""Shared helpers for /ingest/prepare and /research: chunk a document and flag known chunks."""
from __future__ import annotations

from typing import Any

from . import chunking, repo
from .turso import Turso


def parent_text(title: str, source: str, text: str, limit: int = 800) -> str:
    """Text that represents a whole document (title + source + opening). Embed it as the doc record."""
    return f"{title}\n{source}\n{text[:limit]}".strip()


async def prepare_chunks(
    db: Turso, *, text: str, scope: str, size: int, overlap: int, max_chunks: int,
) -> tuple[list[dict[str, Any]], int, bool]:
    """
    Chunk `text` and mark which chunks are already stored in `scope`.
    Returns (chunks, total_chunks, truncated).
    """
    chunks = chunking.chunk_text(text, size, overlap)
    total = len(chunks)
    kept = chunks[:max_chunks]
    hashes = [repo.memory_hash(scope, c) for c in kept]
    have = await repo.existing_hashes(db, hashes)
    out = [{"index": i, "text": c, "exists": h in have} for i, (c, h) in enumerate(zip(kept, hashes))]
    return out, total, total > max_chunks
