"""POST /ingest/prepare - fetch (URL / PDF) or take raw text, clean it and cut it into chunks."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import websearch
from ..config import get_settings
from ..prepare import parent_text, prepare_chunks
from ..schemas import PrepareIn, PrepareOut
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/ingest/prepare", response_model=PrepareOut)
async def ingest_prepare(body: PrepareIn, request: Request) -> PrepareOut:
    """
    The VPS does the heavy lifting (download, PDF/HTML -> text, chunking, duplicate
    check); it returns chunks WITHOUT storing anything. Your PC embeds the chunks
    where `exists` is false and sends them to POST /save_memory_batch.
    """
    s = get_settings()

    if body.url:
        doc = await websearch.fetch_document(
            request.app.state.web, body.url,
            max_bytes=s.web_max_page_bytes, max_chars=s.ingest_max_chars,
            pdf_max_bytes=s.pdf_max_bytes, pdf_max_pages=s.pdf_max_pages,
        )
        if not doc or not doc["text"]:
            raise HTTPException(
                422, "could not get readable text from that URL (blocked address, not HTML/text/PDF, "
                     "scanned PDF, too large, or unreachable)")
        text, kind = doc["text"], doc["type"]
        title = body.title or doc["title"] or body.url
        source = body.source or body.url
    else:
        text, kind = body.text, "text"
        title = body.title or "untitled"
        source = body.source or "manual"

    chunks, total, truncated = await prepare_chunks(
        request.app.state.turso, text=text, scope=body.scope,
        size=body.chunk_chars or s.chunk_chars, overlap=body.overlap if body.overlap is not None else s.chunk_overlap,
        max_chunks=body.max_chunks or 120,
    )
    return PrepareOut(
        title=title, source=source, kind=kind,
        parent_text=parent_text(title, source, text),
        total_chunks=total, truncated=truncated, chunks=chunks,
    )
