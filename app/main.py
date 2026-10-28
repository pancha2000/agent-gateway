"""
Local Multi-Agent Framework - VPS gateway.

Run (single worker, no reload; this is what keeps RAM low):
    uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 32

Endpoints (all require the X-API-Key header, except /health):
    POST /save_memory   text + metadata + vector  -> Turso
    POST /get_hint      query vector              -> top-k snippets (+ past mistakes)
    POST /error_log     Reflexion error tracking
    POST /search_web    DuckDuckGo / Brave search (+ optional page text)
    GET  /health        liveness + current RSS (to verify the memory budget)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import repo
from .config import get_settings
from .routers import errors, memory, web
from .turso import Turso, TursoError
from .websearch import SearchError, TTLCache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


def _rss_mb() -> float | None:
    """Resident memory of THIS process in MB (Linux only)."""
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1_048_576, 1)
    except (OSError, ValueError, IndexError):
        return None


def create_app(
    *,
    turso_transport: httpx.AsyncBaseTransport | None = None,   # test hooks
    web_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.turso = Turso(settings.turso_url, settings.turso_token, transport=turso_transport)
        app.state.web = httpx.AsyncClient(
            timeout=settings.web_timeout,
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=2),
            transport=web_transport,
        )
        app.state.search_cache = TTLCache(max_items=64, ttl=600)

        # Create tables on boot (idempotent). Retry a few times in case the
        # network isn't fully up yet right after a VPS reboot.
        for attempt in range(1, 6):
            try:
                await repo.init_schema(app.state.turso, settings)
                log.info("Turso schema ready (EMBED_DIM=%d, vector_index=%s)",
                         settings.embed_dim, settings.use_vector_index)
                break
            except TursoError as exc:
                log.error("schema init failed (attempt %d/5): %s", attempt, exc)
                if attempt == 5:
                    raise
                await asyncio.sleep(2 * attempt)
        yield
        await app.state.web.aclose()
        await app.state.turso.aclose()

    # Swagger/ReDoc/OpenAPI are disabled by default: less RAM, smaller attack surface.
    docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Agent Memory Gateway", version="1.0.0", lifespan=lifespan, **docs)

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        """Cheap guard against oversized payloads (checks Content-Length)."""
        size = request.headers.get("content-length")
        if size and size.isdigit() and int(size) > settings.max_body_bytes:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(TursoError)
    async def turso_error_handler(_: Request, exc: TursoError):
        log.error("database error: %s", exc)                  # details stay in the log
        return JSONResponse({"detail": "database error"}, status_code=502)

    @app.exception_handler(SearchError)
    async def search_error_handler(_: Request, exc: SearchError):
        return JSONResponse({"detail": f"web search failed: {exc}"}, status_code=502)

    app.include_router(memory.router)
    app.include_router(errors.router)
    app.include_router(web.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "rss_mb": _rss_mb()}

    return app


app = create_app()
