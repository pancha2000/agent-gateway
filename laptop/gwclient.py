"""
Laptop-side client for the Agent Memory Gateway (VPS) + Ollama embeddings.

The VPS does the heavy work (web search, downloading, PDF/HTML -> text, chunking,
duplicate checks, vector search). This file only:
  * turns text into vectors with a SMALL embedding model in Ollama (never the chat model),
  * sends/receives JSON.

    from gwclient import connect
    gw = connect()                      # reads laptop/.env
    gw.remember("the Qwen server listens on :8080", title="qwen port")
    print(gw.hint("which port does qwen use?")["context"])
    out = gw.research("python asyncio TaskGroup")      # memory first, then web (and stores it)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".json": "json", ".sh": "bash", ".c": "c", ".h": "c", ".cpp": "cpp", ".go": "go",
    ".rs": "rust", ".java": "java", ".php": "php", ".rb": "ruby", ".sql": "sql", ".html": "html",
    ".css": "css", ".yml": "yaml", ".yaml": "yaml", ".md": "markdown", ".txt": None,
}


class GatewayError(RuntimeError):
    pass


def load_env(path: Path | None = None) -> None:
    """Tiny .env loader (does not override real environment variables)."""
    path = path or HERE / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# =========================================================================
# Embeddings (Ollama)
# =========================================================================
class Embedder:
    def __init__(self, url: str, model: str, on_cpu: bool = True, timeout: float = 180.0) -> None:
        self.model = model
        self.on_cpu = on_cpu
        self.http = httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        # nomic-embed-text was trained with task prefixes; other models need none.
        nomic = "nomic" in model.lower()
        self.doc_prefix = os.getenv("EMBED_DOC_PREFIX", "search_document: " if nomic else "")
        self.query_prefix = os.getenv("EMBED_QUERY_PREFIX", "search_query: " if nomic else "")

    def embed(self, texts: list[str], *, role: str = "doc") -> list[list[float]]:
        """role = 'doc' for things you store, 'query' for things you search with."""
        if not texts:
            return []
        prefix = self.query_prefix if role == "query" else self.doc_prefix
        out: list[list[float]] = []
        for i in range(0, len(texts), 16):
            batch = [prefix + t for t in texts[i:i + 16]]
            payload: dict[str, Any] = {"model": self.model, "input": batch, "keep_alive": "10m"}
            if self.on_cpu:
                payload["options"] = {"num_gpu": 0}
            try:
                r = self.http.post("/api/embed", json=payload)
                if r.status_code >= 400 and self.on_cpu:          # older Ollama: retry without options
                    payload.pop("options", None)
                    r = self.http.post("/api/embed", json=payload)
            except httpx.HTTPError as exc:
                raise GatewayError(f"Ollama embedding failed ({exc}). Is `ollama serve` running?") from exc
            if r.status_code >= 400:
                raise GatewayError(f"Ollama embed HTTP {r.status_code}: {r.text[:200]} "
                                   f"(did you `ollama pull {self.model}`?)")
            out.extend([[round(float(x), 6) for x in v] for v in r.json()["embeddings"]])
        return out


# =========================================================================
# Gateway client
# =========================================================================
class Gateway:
    def __init__(
        self, url: str, api_key: str, embedder: Embedder, *, scope: str = "shared",
        all_scopes: bool = False, project_id: str | None = None, timeout: float = 90.0,
    ) -> None:
        self.embedder = embedder
        self.scope = scope
        self.all_scopes = all_scopes
        self.project_id = project_id
        self.http = httpx.Client(base_url=url.rstrip("/"), headers={"X-API-Key": api_key}, timeout=timeout)

    # ---- plumbing ----------------------------------------------------------
    def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.http.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise GatewayError(f"cannot reach the gateway ({exc})") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise GatewayError(f"{path}: HTTP {r.status_code}: {str(detail)[:300]}")
        return r.json()

    def health(self) -> dict[str, Any]:
        return self._req("GET", "/health")

    def check(self) -> dict[str, Any]:
        """Verify connectivity AND that the embedding size matches the VPS (EMBED_DIM)."""
        h = self.health()
        dim = len(self.embedder.embed(["dimension probe"], role="query")[0])
        if h.get("embed_dim") != dim:
            raise GatewayError(
                f"embedding size mismatch: {self.embedder.model} gives {dim} dims but the VPS has "
                f"EMBED_DIM={h.get('embed_dim')}. Fix EMBED_DIM (and recreate the tables) or change EMBED_MODEL.")
        return h

    def _vis(self) -> dict[str, Any]:
        return {"scope": self.scope, "all_scopes": self.all_scopes}

    # ---- memory ------------------------------------------------------------
    def remember(
        self, text: str, *, kind: str = "note", language: str | None = None, source: str | None = None,
        title: str | None = None, scope: str | None = None, project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        vec = self.embedder.embed([text], role="doc")[0]
        return self._req("POST", "/save_memory", json={
            "text": text, "vector": vec, "kind": kind, "language": language, "source": source,
            "title": title, "scope": scope or self.scope, "project_id": project_id or self.project_id,
            "metadata": metadata or {},
        })

    def hint(
        self, query: str, *, top_k: int = 4, min_similarity: float = 0.3, snippet_chars: int | None = None,
        max_chars: int = 6000, include_errors: bool = True, kind: str | None = None, language: str | None = None,
    ) -> dict[str, Any]:
        vec = self.embedder.embed([query[:4000]], role="query")[0]
        body: dict[str, Any] = {
            "vector": vec, "top_k": top_k, "min_similarity": min_similarity, "max_chars": max_chars,
            "include_errors": include_errors, "project_id": self.project_id, **self._vis(),
        }
        if snippet_chars:
            body["snippet_chars"] = snippet_chars
        if kind:
            body["kind"] = kind
        if language:
            body["language"] = language
        return self._req("POST", "/get_hint", json=body)

    def recall(self, memory_id: int, *, children: bool = False) -> dict[str, Any]:
        return self._req("GET", f"/recall/{memory_id}", params={**self._vis(), "children": children})

    def delete(self, memory_id: int) -> dict[str, Any]:
        return self._req("DELETE", f"/memory/{memory_id}", params=self._vis())

    def stats(self) -> dict[str, Any]:
        return self._req("GET", "/stats")

    # ---- learning from mistakes -------------------------------------------
    def log_error(
        self, task: str, error: str, *, lesson: str | None = None, fix: str | None = None,
        root_cause: str | None = None, attempt: int = 1, resolved: bool = False,
        agent: str | None = None, tags: list[str] | None = None,
    ) -> dict[str, Any]:
        vec = self.embedder.embed([f"{task}\n{error}"[:3000]], role="doc")[0]
        return self._req("POST", "/error_log", json={
            "task": task[:4000], "error_message": error[:8000], "lesson": lesson, "fix": fix,
            "root_cause": root_cause, "attempt": attempt, "resolved": resolved,
            "agent": agent or self.scope, "tags": tags or [], "vector": vec,
        })

    def resolve_error(self, error_id: int, *, fix: str | None = None, lesson: str | None = None) -> dict[str, Any]:
        return self._req("POST", f"/error_log/{error_id}/resolve", json={"fix": fix, "lesson": lesson})

    def escalation(
        self, task: str, error: str, *, code: str | None = None, attempts: list[dict[str, Any]] | None = None,
        include_memory: bool = True,
    ) -> dict[str, Any]:
        """Returns {'markdown', 'filename', 'memory_hits'}. Send `markdown` to a cloud model or save it as .txt."""
        vec = self.embedder.embed([f"{task}\n{error}"[:3000]], role="query")[0] if include_memory else None
        return self._req("POST", "/escalation_package", json={
            "task": task[:4000], "error_message": error[:8000], "code": code[:20000] if code else None,
            "attempts": attempts or [], "vector": vec, "include_memory": include_memory,
            "project_id": self.project_id, **self._vis(),
        })

    # ---- storing prepared documents ---------------------------------------
    def _store_prepared(
        self, *, title: str, source: str, parent_text: str, chunks: list[dict[str, Any]], kind: str,
        scope: str, project_id: str | None, language: str | None = None, index_offset: int = 0,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        todo = [c for c in chunks if not c["exists"]]
        skipped = len(chunks) - len(todo)
        if not todo:
            return {"stored": 0, "skipped": skipped, "parent_id": parent_id}

        parent = None
        if parent_id is None:
            parent = {"text": parent_text, "vector": self.embedder.embed([parent_text], role="doc")[0],
                      "title": title, "source": source}
        stored = 0
        for i in range(0, len(todo), 24):
            batch = todo[i:i + 24]
            vecs = self.embedder.embed([c["text"] for c in batch], role="doc")
            body: dict[str, Any] = {
                "items": [{"text": c["text"], "vector": v, "chunk_index": index_offset + c["index"]}
                          for c, v in zip(batch, vecs)],
                "kind": kind, "scope": scope, "project_id": project_id, "source": source, "language": language,
            }
            if parent is not None:
                body["parent"] = parent
            else:
                body["parent_id"] = parent_id
            res = self._req("POST", "/save_memory_batch", json=body)
            parent_id, parent = res["parent_id"], None
            stored += len(res["ids"])
        return {"stored": stored, "skipped": skipped, "parent_id": parent_id}

    # ---- research / ingestion ---------------------------------------------
    def research(
        self, query: str, *, force_web: bool = False, store: bool = True, save_scope: str = "shared",
        fetch_top: int = 3, web_threshold: float = 0.6, max_chars: int = 7000,
    ) -> dict[str, Any]:
        """
        Memory first; if nothing good is stored the VPS searches the web, downloads the pages and
        returns chunks. With store=True the new chunks are embedded here and saved for next time.
        Returns the VPS answer plus `stored_chunks`.
        """
        qv = self.embedder.embed([query[:2000]], role="query")[0]
        out = self._req("POST", "/research", json={
            "query": query[:300], "vector": qv, "save_scope": save_scope, "fetch_top": fetch_top,
            "web_threshold": web_threshold, "force_web": force_web, "max_chars": max_chars,
            "project_id": self.project_id, **self._vis(),
        })
        stored = 0
        if store and out["source"] == "web":
            for pg in out["pages"]:
                res = self._store_prepared(
                    title=pg["title"], source=pg["url"], parent_text=pg["parent_text"], chunks=pg["chunks"],
                    kind="web", scope=save_scope, project_id=None)
                stored += res["stored"]
        out["stored_chunks"] = stored
        return out

    def ingest_url(self, url: str, *, title: str | None = None, scope: str | None = None,
                   project_id: str | None = None) -> dict[str, Any]:
        """Web page or PDF by URL: the VPS downloads/cleans/chunks it, this PC embeds it."""
        scope = scope or self.scope
        prep = self._req("POST", "/ingest/prepare", json={"url": url, "title": title, "scope": scope})
        res = self._store_prepared(
            title=prep["title"], source=prep["source"], parent_text=prep["parent_text"], chunks=prep["chunks"],
            kind="web", scope=scope, project_id=project_id or self.project_id)
        return {**res, "title": prep["title"], "total_chunks": prep["total_chunks"], "truncated": prep["truncated"]}

    def ingest_text(self, text: str, *, title: str, source: str = "manual", scope: str | None = None,
                    project_id: str | None = None, language: str | None = None) -> dict[str, Any]:
        scope = scope or self.scope
        prep = self._req("POST", "/ingest/prepare", json={
            "text": text[:200_000], "title": title, "source": source, "scope": scope, "max_chunks": 300})
        res = self._store_prepared(
            title=prep["title"], source=prep["source"], parent_text=prep["parent_text"], chunks=prep["chunks"],
            kind="chunk", scope=scope, project_id=project_id or self.project_id, language=language)
        return {**res, "title": prep["title"], "total_chunks": prep["total_chunks"]}

    def ingest_file(self, path: str | Path, *, title: str | None = None, scope: str | None = None,
                    project_id: str | None = None) -> dict[str, Any]:
        """Local code / text / PDF file of any size (cut into 100k-char windows)."""
        p = Path(path).expanduser()
        if not p.is_file():
            raise GatewayError(f"not a file: {p}")
        ext = p.suffix.lower()
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise GatewayError("pip install pypdf to ingest local PDFs") from exc
            text = "\n\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
        else:
            text = p.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            raise GatewayError("no readable text in that file")

        scope = scope or self.scope
        language = LANG_BY_EXT.get(ext)
        parent_id: int | None = None
        stored = skipped = total = 0
        for window in _windows(text, 100_000):
            prep = self._req("POST", "/ingest/prepare", json={
                "text": window, "title": title or p.name, "source": str(p), "scope": scope, "max_chunks": 300})
            res = self._store_prepared(
                title=prep["title"], source=prep["source"], parent_text=prep["parent_text"], chunks=prep["chunks"],
                kind="chunk", scope=scope, project_id=project_id or self.project_id, language=language,
                index_offset=total, parent_id=parent_id)
            parent_id = res["parent_id"] or parent_id
            stored += res["stored"]
            skipped += res["skipped"]
            total += prep["total_chunks"]
        return {"stored": stored, "skipped": skipped, "total_chunks": total, "parent_id": parent_id, "title": title or p.name}


def _windows(text: str, size: int) -> list[str]:
    """Split a huge text into <=size windows, preferably at line breaks."""
    out: list[str] = []
    rest = text
    while len(rest) > size:
        cut = rest.rfind("\n", 0, size)
        if cut < size // 2:
            cut = size
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest.strip():
        out.append(rest)
    return out


def strip_think(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def connect() -> Gateway:
    """Build a Gateway from laptop/.env (or real environment variables)."""
    load_env()
    url, key = os.getenv("GATEWAY_URL", ""), os.getenv("GATEWAY_API_KEY", "")
    if not url or not key:
        raise GatewayError("set GATEWAY_URL and GATEWAY_API_KEY (see laptop/.env.example)")
    # EMBED_OLLAMA_URL = a separate (CPU-only) Ollama instance just for embeddings, so the chat
    # model never has to be unloaded to make room for the embedding model.
    emb = Embedder(os.getenv("EMBED_OLLAMA_URL") or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
                   os.getenv("EMBED_MODEL", "nomic-embed-text"),
                   on_cpu=os.getenv("EMBED_ON_CPU", "1") != "0")
    return Gateway(
        url, key, emb, scope=os.getenv("AGENT_SCOPE", "shared") or "shared",
        all_scopes=os.getenv("ALL_SCOPES", "0") == "1", project_id=os.getenv("PROJECT_ID") or None)
