"""POST /search_web - fetch documentation snippets when the AI asks for them."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import websearch
from ..config import get_settings
from ..schemas import SearchWebIn, SearchWebOut
from ..security import require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/search_web", response_model=SearchWebOut)
async def search_web(body: SearchWebIn, request: Request) -> SearchWebOut:
    s = get_settings()
    cache: websearch.TTLCache = request.app.state.search_cache
    key = (body.query.strip().lower(), body.max_results, body.fetch_top)

    if (hit := cache.get(key)) is not None:
        engine, results = hit
        cached = True
    else:
        try:
            engine, results = await websearch.search(
                request.app.state.web,
                query=body.query, max_results=body.max_results, fetch_top=body.fetch_top,
                brave_key=s.brave_api_key,
                max_page_bytes=s.web_max_page_bytes, max_page_chars=s.web_max_page_chars,
            )
        except websearch.SearchError as exc:
            raise HTTPException(502, f"web search failed: {exc}") from exc
        cache.put(key, (engine, results))
        cached = False

    return SearchWebOut(
        query=body.query, engine=engine, cached=cached, results=results,
        context=websearch.build_context(results),
    )
