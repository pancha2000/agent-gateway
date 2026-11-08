"""POST /research - 'check the database first, otherwise go to the web and prepare what we find'."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends, Request

from .. import repo, vectors, websearch
from ..config import get_settings
from ..context import build_hint_context, build_pages_context
from ..prepare import parent_text, prepare_chunks
from ..schemas import ResearchIn, ResearchOut
from ..security import require_api_key

log = logging.getLogger("gateway.research")
router = APIRouter(dependencies=[Depends(require_api_key)])

MAX_CHUNKS_PER_PAGE = 15


@router.post("/research", response_model=ResearchOut)
async def research(body: ResearchIn, request: Request) -> ResearchOut:
    """
    1. If `vector` is sent, search the memory. A strong hit (>= web_threshold) is
       returned immediately, nothing goes to the web.
    2. Otherwise search the web, download the top pages/PDFs, extract text, cut it
       into chunks and mark the chunks that are already stored.
    The VPS never embeds anything: your PC embeds the chunks where `exists` is false
    and stores them with POST /save_memory_batch (kind='web', parent = page record).
    """
    s = get_settings()
    db = request.app.state.turso

    hits: list[dict] = []
    errors: list[dict] = []
    if body.vector is not None:
        vectors.validate(body.vector, s.embed_dim)
        hits, errors = await repo.get_hint(
            db, s, vector=body.vector, top_k=body.top_k, min_similarity=body.min_similarity,
            kind=None, language=None, include_errors=body.include_errors,
            scope=body.scope, all_scopes=body.all_scopes, project_id=body.project_id)
    best = max((h["similarity"] for h in hits), default=0.0)

    if hits and best >= body.web_threshold and not body.force_web:
        return ResearchOut(
            source="memory", best_similarity=best, engine=None, hits=hits, past_errors=errors,
            pages=[], other_results=[], context=build_hint_context(hits, errors, body.max_chars),
            note="answered from memory (send force_web=true to search the web anyway)")

    # ---- web search (shares the /search_web cache) ----------------------------------
    cache: websearch.TTLCache = request.app.state.search_cache
    key = (body.query.strip().lower(), body.max_results, 0)
    try:
        cached = cache.get(key)
        if cached is not None:
            engine, results = cached
        else:
            engine, results = await websearch.search(
                request.app.state.web, query=body.query, max_results=body.max_results, fetch_top=0,
                brave_key=s.brave_api_key, max_page_bytes=s.web_max_page_bytes,
                max_page_chars=s.web_max_page_chars)
            cache.put(key, (engine, results))
    except websearch.SearchError as exc:
        if hits:                                   # degrade gracefully: weak memory beats nothing
            return ResearchOut(
                source="memory", best_similarity=best, engine=None, hits=hits, past_errors=errors,
                pages=[], other_results=[], context=build_hint_context(hits, errors, body.max_chars),
                note=f"web search failed ({exc}); returning the best memory hits instead")
        raise                                      # -> 502 via the app's SearchError handler

    # ---- download + extract the top pages, in parallel ------------------------------
    targets = results[: body.fetch_top]
    docs = await asyncio.gather(*[
        websearch.fetch_document(
            request.app.state.web, r["url"], max_bytes=s.web_max_page_bytes,
            max_chars=s.research_page_chars, pdf_max_bytes=s.pdf_max_bytes,
            pdf_max_pages=s.pdf_max_pages)
        for r in targets
    ], return_exceptions=True)

    pages: list[dict] = []
    fetched: set[str] = set()
    for r, doc in zip(targets, docs):
        if isinstance(doc, BaseException):
            log.warning("fetch failed for %s: %s", r["url"], doc)
            continue
        if not doc or not doc["text"]:
            continue
        title = doc["title"] or r["title"]
        chunks, total, truncated = await prepare_chunks(
            db, text=doc["text"], scope=body.save_scope, size=s.chunk_chars,
            overlap=s.chunk_overlap, max_chunks=MAX_CHUNKS_PER_PAGE)
        pages.append({
            "url": r["url"], "title": title, "kind": doc["type"],
            "parent_text": parent_text(title, r["url"], doc["text"]),
            "total_chunks": total, "truncated": truncated, "chunks": chunks,
        })
        fetched.add(r["url"])

    others = [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
              for r in results if r["url"] not in fetched]

    pages_ctx = build_pages_context(pages, others, body.max_chars)
    if hits or errors:
        hint_ctx = build_hint_context(hits, errors, body.max_chars // 2)
        context = (hint_ctx + "\n\n" + pages_ctx)[: body.max_chars]
    else:
        context = pages_ctx

    note = None
    if not pages:
        note = "search worked but no page could be fetched; only the result snippets are available"
    return ResearchOut(
        source="web", best_similarity=best, engine=engine, hits=hits, past_errors=errors,
        pages=pages, other_results=others, context=context, note=note)
