"""
A fake Turso server for offline tests: speaks the Hrana /v2/pipeline protocol and
executes the SQL on an in-memory SQLite database. Turso's vector functions are
emulated with Python (`vector32`, `vector_distance_cos`), so the REAL SQL used by
the gateway runs end to end.
"""
from __future__ import annotations

import base64
import json
import math
import sqlite3
import struct

import httpx


def _vector32(x):
    if x is None:
        return None
    if isinstance(x, bytes):
        return x
    nums = [float(v) for v in x.strip("[]").split(",") if v.strip()]
    return struct.pack(f"<{len(nums)}f", *nums)


def _cos_dist(a, b):
    if a is None or b is None:
        return None
    va = struct.unpack(f"<{len(a) // 4}f", a)
    vb = struct.unpack(f"<{len(b) // 4}f", b)
    dot = sum(x * y for x, y in zip(va, vb))
    na, nb = math.sqrt(sum(x * x for x in va)), math.sqrt(sum(y * y for y in vb))
    return 1.0 - dot / (na * nb) if na and nb else 1.0


class FakeTurso:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = None
        self.db.create_function("vector32", 1, _vector32)
        self.db.create_function("vector_distance_cos", 2, _cos_dist)
        self.calls = 0

    @staticmethod
    def _dec(a):
        t = a["type"]
        return (None if t == "null" else int(a["value"]) if t == "integer"
                else a["value"] if t in ("float", "text") else base64.b64decode(a["base64"]))

    @staticmethod
    def _enc(v):
        if v is None:
            return {"type": "null"}
        if isinstance(v, int):
            return {"type": "integer", "value": str(v)}
        if isinstance(v, float):
            return {"type": "float", "value": v}
        if isinstance(v, bytes):
            return {"type": "blob", "base64": base64.b64encode(v).decode()}
        return {"type": "text", "value": v}

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v2/pipeline"
        assert request.headers["authorization"] == "Bearer test-token"
        self.calls += 1
        results = []
        for req in json.loads(request.content)["requests"]:
            if req["type"] == "close":
                results.append({"type": "ok", "response": {"type": "close"}})
                continue
            stmt = req["stmt"]
            try:
                cur = self.db.execute(stmt["sql"], [self._dec(a) for a in stmt.get("args", [])])
                cols = [{"name": d[0]} for d in (cur.description or [])]
                rows = [[self._enc(v) for v in r] for r in cur.fetchall()]
                self.db.commit()
                results.append({"type": "ok", "response": {
                    "type": "execute", "result": {"cols": cols, "rows": rows}}})
            except sqlite3.Error as exc:
                results.append({"type": "error", "error": {"message": str(exc)}})
        return httpx.Response(200, json={"baton": None, "base_url": None, "results": results})
