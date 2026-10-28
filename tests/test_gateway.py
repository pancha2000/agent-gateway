"""
Offline end-to-end tests (no Turso account, no internet needed):

    pip install -r requirements.txt pytest
    python -m pytest -q          # or: python tests/test_gateway.py
"""
from __future__ import annotations

import math
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

DDG_HTML = """
<div class="result"><h2><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&rut=x">asyncio &mdash; <b>Asynchronous</b> I/O</a></h2>
<a class="result__snippet" href="x">Hello, <b>asyncio</b> is a library to write concurrent code.</a></div>
<div class="result result--ad"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">AD</a></div>
<div class="result"><a class="result__a" href="https://example.org/page">Example page</a>
<a class="result__snippet">Second snippet</a></div>
"""
PAGE_HTML = ("<html><head><style>x{}</style></head><body><nav>MENU</nav><h1>Title</h1>"
             "<p>Some docs text.</p><pre>def f():\n    return 1</pre><script>evil()</script></body></html>")


def web_handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == "html.duckduckgo.com":
        return httpx.Response(200, text=DDG_HTML)
    if request.url.host == "docs.python.org":
        return httpx.Response(200, text=PAGE_HTML, headers={"content-type": "text/html; charset=utf-8"})
    return httpx.Response(404)


def vec(*head: float) -> list[float]:
    v = list(head) + [0.0] * (8 - len(head))
    return v


def make_client() -> TestClient:
    fake = FakeTurso()

    async def fake_resolve(host: str, port: int):      # no real DNS in tests
        return ["127.0.0.1"] if host == "internal.local" else ["93.184.216.34"]

    websearch._resolve = fake_resolve
    app = create_app(turso_transport=httpx.MockTransport(fake.handler),
                     web_transport=httpx.MockTransport(web_handler))
    return TestClient(app)


def test_auth_and_health():
    with make_client() as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.post("/save_memory", json={}).status_code == 401
        assert c.post("/save_memory", json={}, headers={"X-API-Key": "wrong"}).status_code == 401


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


def test_vector_validation():
    with make_client() as c:
        bad = c.post("/save_memory", headers=H, json={"text": "x", "vector": [1.0, 2.0]})
        assert bad.status_code == 422 and "expected 8" in bad.json()["detail"]
        nan = c.post("/get_hint", headers=H, content='{"vector": [NaN,0,0,0,0,0,0,0]}',
                     )  # content-type missing -> 422 either way
        assert nan.status_code == 422


def test_error_log_reflexion():
    with make_client() as c:
        body = {"task": "parse csv", "error_message": "KeyError: 'id'", "lesson": "check headers first",
                "agent": "coder", "tags": ["csv"], "vector": vec(0, 0, 1)}
        first = c.post("/error_log", headers=H, json=body).json()
        assert first["occurrences"] == 1 and first["is_repeat"] is False
        # same mistake (whitespace/case differences normalised) -> repeat
        body["error_message"] = "keyerror:   'ID'"
        second = c.post("/error_log", headers=H, json=body).json()
        assert second["id"] == first["id"] and second["occurrences"] == 2 and second["is_repeat"]

        # surfaced by /get_hint for a similar task
        r = c.post("/get_hint", headers=H, json={"vector": vec(0, 0, 1), "min_similarity": 0.5}).json()
        assert r["past_errors"] and r["past_errors"][0]["lesson"] == "check headers first"
        assert "Past mistakes" in r["context"] and "repeated 2 times" in r["context"]

        # without a vector the error is still stored
        nov = c.post("/error_log", headers=H, json={"task": "t", "error_message": "boom"}).json()
        assert nov["occurrences"] == 1


def test_search_web_and_cache():
    with make_client() as c:
        r = c.post("/search_web", headers=H, json={"query": "python asyncio docs", "fetch_top": 1}).json()
        assert r["engine"] == "duckduckgo" and r["cached"] is False
        assert [x["url"] for x in r["results"]] == [
            "https://docs.python.org/3/library/asyncio.html", "https://example.org/page"]
        assert r["results"][0]["title"] == "asyncio — Asynchronous I/O"
        content = r["results"][0]["content"]
        assert "Some docs text." in content and "def f():\n    return 1" in content
        assert "MENU" not in content and "evil()" not in content
        assert r["results"][1]["content"] is None
        assert "asyncio" in r["context"]
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
