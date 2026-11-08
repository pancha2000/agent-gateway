"""
Offline end-to-end tests (no Turso account, no internet needed):

    pip install -r requirements.txt pytest
    python -m pytest -q          # or: python tests/test_gateway.py
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

os.environ.update(
    TURSO_DATABASE_URL="libsql://fake-db.turso.io",
    TURSO_AUTH_TOKEN="test-token",
    GATEWAY_API_KEY="k" * 32,
    EMBED_DIM="8",
    USE_VECTOR_INDEX="false",
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx                                      # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402

from app import websearch                         # noqa: E402
from app.main import create_app                   # noqa: E402
from tests.fake_turso import FakeTurso            # noqa: E402

H = {"X-API-Key": "k" * 32}

# A real one-page PDF containing the text "Hello PDF world" (title: "Guide PDF").
PDF_BYTES = base64.b64decode(
    "JVBERi0xLjMKJZOMi54gUmVwb3J0TGFiIEdlbmVyYXRlZCBQREYgZG9jdW1lbnQgKG9wZW5zb3Vy"
    "Y2UpCjEgMCBvYmoKPDwKL0YxIDIgMCBSCj4+CmVuZG9iagoyIDAgb2JqCjw8Ci9CYXNlRm9udCAv"
    "SGVsdmV0aWNhIC9FbmNvZGluZyAvV2luQW5zaUVuY29kaW5nIC9OYW1lIC9GMSAvU3VidHlwZSAv"
    "VHlwZTEgL1R5cGUgL0ZvbnQKPj4KZW5kb2JqCjMgMCBvYmoKPDwKL0NvbnRlbnRzIDcgMCBSIC9N"
    "ZWRpYUJveCBbIDAgMCA1OTUuMjc1NiA4NDEuODg5OCBdIC9QYXJlbnQgNiAwIFIgL1Jlc291cmNl"
    "cyA8PAovRm9udCAxIDAgUiAvUHJvY1NldCBbIC9QREYgL1RleHQgL0ltYWdlQiAvSW1hZ2VDIC9J"
    "bWFnZUkgXQo+PiAvUm90YXRlIDAgL1RyYW5zIDw8Cgo+PiAKICAvVHlwZSAvUGFnZQo+PgplbmRv"
    "YmoKNCAwIG9iago8PAovUGFnZU1vZGUgL1VzZU5vbmUgL1BhZ2VzIDYgMCBSIC9UeXBlIC9DYXRh"
    "bG9nCj4+CmVuZG9iago1IDAgb2JqCjw8Ci9BdXRob3IgKGFub255bW91cykgL0NyZWF0aW9uRGF0"
    "ZSAoRDoyMDAwMDEwMTAwMDAwMCswMCcwMCcpIC9DcmVhdG9yIChhbm9ueW1vdXMpIC9LZXl3b3Jk"
    "cyAoKSAvTW9kRGF0ZSAoRDoyMDAwMDEwMTAwMDAwMCswMCcwMCcpIC9Qcm9kdWNlciAoUmVwb3J0"
    "TGFiIFBERiBMaWJyYXJ5IC0gXChvcGVuc291cmNlXCkpIAogIC9TdWJqZWN0ICh1bnNwZWNpZmll"
    "ZCkgL1RpdGxlIChHdWlkZSBQREYpIC9UcmFwcGVkIC9GYWxzZQo+PgplbmRvYmoKNiAwIG9iago8"
    "PAovQ291bnQgMSAvS2lkcyBbIDMgMCBSIF0gL1R5cGUgL1BhZ2VzCj4+CmVuZG9iago3IDAgb2Jq"
    "Cjw8Ci9MZW5ndGggOTAKPj4Kc3RyZWFtCjEgMCAwIDEgMCAwIGNtICBCVCAvRjEgMTIgVGYgMTQu"
    "NCBUTCBFVApCVCAxIDAgMCAxIDcyIDc1MCBUbSAoSGVsbG8gUERGIHdvcmxkKSBUaiBUKiBFVAog"
    "CmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDgKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDYx"
    "IDAwMDAwIG4gCjAwMDAwMDAwOTIgMDAwMDAgbiAKMDAwMDAwMDE5OSAwMDAwMCBuIAowMDAwMDAw"
    "NDAyIDAwMDAwIG4gCjAwMDAwMDA0NzAgMDAwMDAgbiAKMDAwMDAwMDczMiAwMDAwMCBuIAowMDAw"
    "MDAwNzkxIDAwMDAwIG4gCnRyYWlsZXIKPDwKL0lEIApbPGNjYzYwNWE4NzdiOWQ2YWM4MTkxYWY2"
    "NjU4M2ZjYmM2PjxjY2M2MDVhODc3YjlkNmFjODE5MWFmNjY1ODNmY2JjNj5dCiUgUmVwb3J0TGFi"
    "IGdlbmVyYXRlZCBQREYgZG9jdW1lbnQgLS0gZGlnZXN0IChvcGVuc291cmNlKQoKL0luZm8gNSAw"
    "IFIKL1Jvb3QgNCAwIFIKL1NpemUgOAo+PgpzdGFydHhyZWYKOTMwCiUlRU9GCg=="
)

DDG_HTML = """
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&rut=x">asyncio &mdash; <b>Asynchronous</b> I/O</a></h2>
<a class="result__snippet" href="x">Hello, <b>asyncio</b> is a library to write concurrent code.</a></div>
<div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">AD</a></div>
<div class="result"><a class="result__a" href="https://example.org/page">Example page</a>
<a class="result__snippet">Second snippet</a></div>
"""
DDG_PDF_HTML = """
<div class="result"><a class="result__a" href="https://files.example.org/guide.pdf">The Guide (PDF)</a>
<a class="result__snippet">A guide as a PDF.</a></div>
"""
PAGE_HTML = ("<html><head><title>Asyncio &mdash; docs</title><style>x{}</style></head><body><nav>MENU</nav>"
             "<h1>Title</h1><p>Some docs text.</p><pre>def f():\n    return 1</pre>"
             "<script>evil()</script></body></html>")


def web_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "html.duckduckgo.com":
        return httpx.Response(200, text=DDG_PDF_HTML if b"pdf" in request.content else DDG_HTML)
    if request.url.host == "docs.python.org":
        return httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html; charset=utf-8"})
    if request.url.host == "files.example.org":
        return httpx.Response(200, content=PDF_BYTES, headers={"content-type": "application/pdf"})
    return httpx.Response(404)


def vec(*head: float) -> list[float]:
    return list(head) + [0.0] * (8 - len(head))


def make_client(fake: FakeTurso | None = None) -> TestClient:
    fake = fake or FakeTurso()

    async def fake_resolve(host: str, port: int):      # no real DNS in tests
        return ["127.0.0.1"] if host == "internal.local" else ["93.184.216.34"]

    websearch._resolve = fake_resolve
    app = create_app(turso_transport=httpx.MockTransport(fake.handler),
                     web_transport=httpx.MockTransport(web_handler))
    return TestClient(app)


# ----------------------------------------------------------------------------- basics
def test_auth_and_health():
    with make_client() as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["embed_dim"] == 8
        assert c.post("/save_memory", json={}).status_code == 401
        assert c.post("/save_memory", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/research", json={}).status_code == 401
        assert c.post("/ingest/prepare", json={}).status_code == 401


def test_save_and_hint_roundtrip():
    with make_client() as c:
        a = c.post("/save_memory", headers=H, json={
            "text": "def add(a, b): return a + b", "vector": vec(1, 0), "language": "python",
            "source": "utils.py", "metadata": {"tags": ["math"]}})
        assert a.status_code == 200 and a.json()["created"] is True
        c.post("/save_memory", headers=H, json={
            "text": "console.log('hi')", "vector": vec(0, 1), "language": "js"})

        # idempotent save
        again = c.post("/save_memory", headers=H, json={
            "text": "def add(a, b): return a + b", "vector": vec(1, 0), "language": "python"})
        assert again.json()["created"] is False and again.json()["id"] == a.json()["id"]

        r = c.post("/get_hint", headers=H, json={"vector": vec(0.9, 0.1), "top_k": 5}).json()
        assert r["hits"][0]["content"].startswith("def add")
        assert r["hits"][0]["similarity"] > 0.99
        assert all(h["similarity"] >= 0.25 for h in r["hits"])      # orthogonal one is filtered
        assert len(r["hits"]) == 1
        assert "def add" in r["context"]

        # language filter
        r = c.post("/get_hint", headers=H, json={
            "vector": vec(0.5, 0.5), "min_similarity": 0.0, "language": "js"}).json()
        assert [h["language"] for h in r["hits"]] == ["js"]


def test_vector_and_input_validation():
    with make_client() as c:
        bad = c.post("/save_memory", headers=H, json={"text": "x", "vector": [1.0, 2.0]})
        assert bad.status_code == 422 and "expected 8" in bad.json()["detail"]
        nan = c.post("/get_hint", headers=H, content='{"vector": [NaN,0,0,0,0,0,0,0]}')
        assert nan.status_code == 422
        # blank text and odd scope names are refused
        assert c.post("/save_memory", headers=H, json={"text": "   ", "vector": vec(1)}).status_code == 422
        assert c.post("/save_memory", headers=H, json={
            "text": "x", "vector": vec(1), "scope": "bad scope!"}).status_code == 422


# ----------------------------------------------------------------------------- scopes
def test_scope_isolation_and_main_agent():
    with make_client() as c:
        def save(text, scope, v):
            r = c.post("/save_memory", headers=H, json={"text": text, "vector": v, "scope": scope})
            assert r.status_code == 200, r.text
            return r.json()["id"]

        save("shared fact", "shared", vec(1, 0))
        coder_id = save("coder secret", "coder", vec(1, 0.05))
        sec_id = save("security secret", "security", vec(1, 0.1))

        def search(**extra):
            r = c.post("/get_hint", headers=H, json={
                "vector": vec(1), "min_similarity": 0.0, "include_errors": False, **extra}).json()
            return {h["content"] for h in r["hits"]}

        assert search(scope="coder") == {"shared fact", "coder secret"}
        assert search(scope="security") == {"shared fact", "security secret"}
        assert search() == {"shared fact"}
        assert search(all_scopes=True) == {"shared fact", "coder secret", "security secret"}

        # recall respects visibility too
        assert c.get(f"/recall/{sec_id}", headers=H, params={"scope": "coder"}).status_code == 404
        assert c.get(f"/recall/{sec_id}", headers=H, params={"all_scopes": True}).status_code == 200
        assert c.get(f"/recall/{coder_id}", headers=H, params={"scope": "coder"}).json()["memory"]["content"] == "coder secret"


def test_project_filter_and_snippet_hint():
    with make_client() as c:
        long_text = "alpha " * 60
        c.post("/save_memory", headers=H, json={"text": long_text, "vector": vec(1, 0), "project_id": "projA"})
        c.post("/save_memory", headers=H, json={"text": "beta note", "vector": vec(1, 0.1), "project_id": "projB"})
        r = c.post("/get_hint", headers=H, json={
            "vector": vec(1), "min_similarity": 0.0, "include_errors": False,
            "project_id": "projA", "snippet_chars": 50}).json()
        assert len(r["hits"]) == 1 and r["hits"][0]["project_id"] == "projA"
        assert len(r["hits"][0]["content"]) < len(long_text) and "/recall/" in r["hits"][0]["content"]
        full = c.get(f"/recall/{r['hits'][0]['id']}", headers=H).json()["memory"]["content"]
        assert full == long_text.strip()


# ----------------------------------------------------------------------------- batch / recall / delete
def test_batch_parent_recall_delete_stats():
    with make_client() as c:
        items = [{"text": f"chunk {i}", "vector": vec(0, 1, i), "chunk_index": i} for i in range(3)]
        r = c.post("/save_memory_batch", headers=H, json={
            "items": items, "kind": "web", "source": "https://x.test/doc",
            "parent": {"text": "Doc title\nhttps://x.test/doc", "vector": vec(0, 1), "title": "Doc title"},
        })
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["parent_id"] and len(out["ids"]) == 3 and all(out["created"])

        again = c.post("/save_memory_batch", headers=H, json={
            "items": items[:1], "kind": "web", "parent_id": out["parent_id"]}).json()
        assert again["created"] == [False] and again["ids"] == out["ids"][:1]

        rec = c.get(f"/recall/{out['parent_id']}", headers=H, params={"children": True}).json()
        assert rec["memory"]["kind"] == "doc" and rec["memory"]["title"] == "Doc title"
        assert [k["chunk_index"] for k in rec["children"]] == [0, 1, 2]
        assert {k["kind"] for k in rec["children"]} == {"web"} and rec["children"][0]["source"] == "https://x.test/doc"

        st = c.get("/stats", headers=H).json()
        assert {"scope": "shared", "kind": "web", "n": 3} in st["memories"]

        d = c.delete(f"/memory/{out['parent_id']}", headers=H).json()
        assert out["parent_id"] in d["deleted"] and len(d["deleted"]) == 4
        assert c.get(f"/recall/{out['parent_id']}", headers=H).status_code == 404
        assert c.delete(f"/memory/{out['parent_id']}", headers=H).status_code == 404

        too_many = [{"text": f"t{i}", "vector": vec(1)} for i in range(41)]
        assert c.post("/save_memory_batch", headers=H, json={"items": too_many}).status_code == 422


# ----------------------------------------------------------------------------- errors / reflexion
def test_error_log_reflexion_and_resolve():
    with make_client() as c:
        body = {"task": "parse csv", "error_message": "KeyError: 'id'", "lesson": "check headers first",
                "agent": "coder", "tags": ["csv"], "vector": vec(0, 0, 1)}
        first = c.post("/error_log", headers=H, json=body).json()
        assert first["occurrences"] == 1 and first["is_repeat"] is False
        assert first["attempts"] == 1 and first["resolved"] is False
        # same mistake (whitespace/case differences normalised) -> repeat, higher attempt kept
        body.update(error_message="keyerror:   'ID'", attempt=3)
        second = c.post("/error_log", headers=H, json=body).json()
        assert second["id"] == first["id"] and second["occurrences"] == 2 and second["is_repeat"]
        assert second["attempts"] == 3

        # surfaced by /get_hint for a similar task
        r = c.post("/get_hint", headers=H, json={"vector": vec(0, 0, 1), "min_similarity": 0.5}).json()
        assert r["past_errors"] and r["past_errors"][0]["lesson"] == "check headers first"
        assert r["past_errors"][0]["resolved"] is False
        assert "Past mistakes" in r["context"] and "repeated 2 times" in r["context"]
        assert "UNRESOLVED" in r["context"]

        # resolve it
        res = c.post(f"/error_log/{first['id']}/resolve", headers=H, json={"fix": "use row.get('id')"}).json()
        assert res["resolved"] is True
        r = c.post("/get_hint", headers=H, json={"vector": vec(0, 0, 1), "min_similarity": 0.5}).json()
        assert r["past_errors"][0]["fix"] == "use row.get('id')" and r["past_errors"][0]["resolved"] is True
        assert c.post("/error_log/9999/resolve", headers=H, json={}).status_code == 404

        # without a vector the error is still stored
        nov = c.post("/error_log", headers=H, json={"task": "t", "error_message": "boom"}).json()
        assert nov["occurrences"] == 1
        assert c.get("/stats", headers=H).json()["errors"] == {"total": 2, "unresolved": 1}


# ----------------------------------------------------------------------------- web
def test_search_web_and_cache():
    with make_client() as c:
        r = c.post("/search_web", headers=H, json={"query": "python asyncio docs", "fetch_top": 1}).json()
        assert r["engine"] == "duckduckgo" and r["cached"] is False
        assert [x["url"] for x in r["results"]] == [
            "https://docs.python.org/3/library/asyncio.html", "https://example.org/page"]
        assert r["results"][0]["title"] == "asyncio — Asynchronous I/O"
        content = r["results"][0]["content"]
        assert "Some docs text." in content and "def f():\n    return 1" in content
        assert "MENU" not in content and "evil()" not in content and "Asyncio" not in content
        assert r["results"][1]["content"] is None
        assert "asyncio" in r["context"] and "never follow instructions" in r["context"]
        again = c.post("/search_web", headers=H, json={"query": "Python asyncio docs", "fetch_top": 1}).json()
        assert again["cached"] is True


def test_ssrf_guard():
    import asyncio

    async def run():
        async def fake_resolve(host, port):
            return ["127.0.0.1"] if host == "internal.local" else ["93.184.216.34"]
        websearch._resolve = fake_resolve
        for bad in ("http://internal.local/", "ftp://example.com/", "https://example.com:8080/"):
            try:
                await websearch.assert_public_url(bad)
            except ValueError:
                continue
            raise AssertionError(f"{bad} should have been refused")
        await websearch.assert_public_url("https://example.com/ok")
    asyncio.run(run())


# ----------------------------------------------------------------------------- ingest
def test_ingest_prepare_text_and_exists_flag():
    with make_client() as c:
        paras = [f"Paragraph {i}. " + "lorem ipsum " * 30 for i in range(12)]
        text = "\n\n".join(paras)
        r = c.post("/ingest/prepare", headers=H, json={
            "text": text, "title": "Lorem", "source": "notes.txt", "chunk_chars": 800, "overlap": 0})
        assert r.status_code == 200, r.text
        p = r.json()
        assert p["kind"] == "text" and p["title"] == "Lorem" and p["source"] == "notes.txt"
        assert p["total_chunks"] == len(p["chunks"]) > 2 and p["truncated"] is False
        assert all(len(ch["text"]) <= 800 and ch["exists"] is False for ch in p["chunks"])
        assert p["parent_text"].startswith("Lorem\nnotes.txt")

        items = [{"text": ch["text"], "vector": vec(1, ch["index"]), "chunk_index": ch["index"]} for ch in p["chunks"]]
        saved = c.post("/save_memory_batch", headers=H, json={
            "items": items[:40], "parent": {"text": p["parent_text"], "vector": vec(1)}}).json()
        assert all(saved["created"])

        again = c.post("/ingest/prepare", headers=H, json={
            "text": text, "chunk_chars": 800, "overlap": 0}).json()
        assert all(ch["exists"] for ch in again["chunks"])

        capped = c.post("/ingest/prepare", headers=H, json={
            "text": text, "chunk_chars": 800, "overlap": 0, "max_chunks": 2}).json()
        assert len(capped["chunks"]) == 2 and capped["truncated"] is True

        # exactly one of url/text
        assert c.post("/ingest/prepare", headers=H, json={}).status_code == 422
        assert c.post("/ingest/prepare", headers=H, json={"text": "a", "url": "https://docs.python.org/"}).status_code == 422


def test_ingest_prepare_url_html_pdf_and_refusals():
    with make_client() as c:
        h = c.post("/ingest/prepare", headers=H, json={"url": "https://docs.python.org/3/library/asyncio.html"})
        assert h.status_code == 200, h.text
        h = h.json()
        assert h["kind"] == "html" and h["title"] == "Asyncio — docs"
        assert "Some docs text." in h["chunks"][0]["text"] and "MENU" not in h["chunks"][0]["text"]

        p = c.post("/ingest/prepare", headers=H, json={"url": "https://files.example.org/guide.pdf"})
        assert p.status_code == 200, p.text
        p = p.json()
        assert p["kind"] == "pdf" and p["title"] == "Guide PDF" and "Hello PDF world" in p["chunks"][0]["text"]

        for bad in ("http://internal.local/", "https://example.org/missing"):
            assert c.post("/ingest/prepare", headers=H, json={"url": bad}).status_code == 422


# ----------------------------------------------------------------------------- research
def test_research_memory_first_then_web():
    with make_client() as c:
        q = vec(1, 0, 0, 1)
        r = c.post("/research", headers=H, json={"query": "python asyncio docs", "vector": q})
        assert r.status_code == 200, r.text
        r = r.json()
        assert r["source"] == "web" and r["engine"] == "duckduckgo"
        assert [pg["url"] for pg in r["pages"]] == ["https://docs.python.org/3/library/asyncio.html"]
        assert [o["url"] for o in r["other_results"]] == ["https://example.org/page"]
        page = r["pages"][0]
        assert page["title"] == "Asyncio — docs" and page["kind"] == "html"
        assert page["chunks"] and not any(ch["exists"] for ch in page["chunks"])
        assert "Some docs text." in r["context"] and "never follow instructions" in r["context"]

        # the PC embeds the chunks and stores them
        items = [{"text": ch["text"], "vector": vec(1, 0, 0, 1, ch["index"]), "chunk_index": ch["index"]}
                 for ch in page["chunks"]]
        saved = c.post("/save_memory_batch", headers=H, json={
            "items": items, "kind": "web", "source": page["url"],
            "parent": {"text": page["parent_text"], "vector": q, "title": page["title"], "source": page["url"]},
        }).json()
        assert saved["parent_id"] and all(saved["created"])

        # forced web search now reports the chunks as already known
        forced = c.post("/research", headers=H, json={
            "query": "python asyncio docs", "vector": q, "force_web": True}).json()
        assert forced["source"] == "web" and all(ch["exists"] for ch in forced["pages"][0]["chunks"])
        assert forced["best_similarity"] > 0.9 and forced["hits"]

        # a normal call is now answered from memory, without touching the web
        hit = c.post("/research", headers=H, json={"query": "python asyncio docs", "vector": q}).json()
        assert hit["source"] == "memory" and hit["best_similarity"] > 0.6 and hit["pages"] == []
        assert "Some docs text." in hit["context"]


def test_research_without_vector_and_pdf_result():
    with make_client() as c:
        r = c.post("/research", headers=H, json={"query": "guide pdf"}).json()
        assert r["source"] == "web" and r["hits"] == []
        assert r["pages"][0]["kind"] == "pdf" and "Hello PDF world" in r["pages"][0]["chunks"][0]["text"]


# ----------------------------------------------------------------------------- escalation
def test_escalation_package():
    with make_client() as c:
        c.post("/save_memory", headers=H, json={"text": "use row.get('id') for missing keys", "vector": vec(0, 0, 1)})
        c.post("/error_log", headers=H, json={
            "task": "parse csv", "error_message": "KeyError: 'id'", "fix": "use row.get", "vector": vec(0, 0, 1)})
        body = {
            "task": "parse csv rows", "error_message": "KeyError: 'id'", "code": "print(row['id'])",
            "attempts": [{"attempt": 1, "approach": "wrapped it in try/except", "result": "still KeyError"}],
            "vector": vec(0, 0, 1),
        }
        r = c.post("/escalation_package", headers=H, json=body)
        assert r.status_code == 200, r.text
        out = r.json()
        md = out["markdown"]
        assert md.startswith("# Help request: parse csv rows")
        for needle in ("## Task", "KeyError: 'id'", "print(row['id'])", "wrapped it in try/except",
                       "still KeyError", "row.get('id')", "## What I need"):
            assert needle in md, needle
        assert out["filename"].startswith("escalation_") and out["filename"].endswith(".txt")
        assert out["memory_hits"] >= 2

        private = c.post("/escalation_package", headers=H, json={**body, "include_memory": False}).json()
        assert "knowledge base" not in private["markdown"] and private["memory_hits"] == 0


# ----------------------------------------------------------------------------- migration
V1_DDL = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT, content_hash TEXT NOT NULL UNIQUE, content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'snippet', language TEXT, source TEXT, metadata TEXT NOT NULL DEFAULT '{}',
    embedding F32_BLOB(8) NOT NULL, created_at INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at INTEGER NOT NULL DEFAULT (unixepoch()));
CREATE TABLE error_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, error_hash TEXT NOT NULL UNIQUE, task TEXT NOT NULL,
    error_message TEXT NOT NULL, root_cause TEXT, lesson TEXT, fix TEXT, agent TEXT,
    tags TEXT NOT NULL DEFAULT '[]', occurrences INTEGER NOT NULL DEFAULT 1, embedding F32_BLOB(8),
    first_seen INTEGER NOT NULL DEFAULT (unixepoch()), last_seen INTEGER NOT NULL DEFAULT (unixepoch()));
"""


def test_v1_database_is_migrated_on_startup():
    fake = FakeTurso()
    fake.db.executescript(V1_DDL)
    fake.db.execute("INSERT INTO memories (content_hash, content, embedding) "
                    "VALUES ('h', 'legacy note', vector32('[1,0,0,0,0,0,0,0]'))")
    fake.db.commit()
    with make_client(fake) as c:
        r = c.post("/get_hint", headers=H, json={"vector": vec(1), "include_errors": False}).json()
        assert [h["content"] for h in r["hits"]] == ["legacy note"] and r["hits"][0]["scope"] == "shared"
        ok = c.post("/save_memory", headers=H, json={"text": "new", "vector": vec(1), "scope": "coder"})
        assert ok.status_code == 200


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
