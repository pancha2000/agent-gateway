"""
Minimal async Turso client over the Hrana-over-HTTP API (`POST /v2/pipeline`).

Why not the official `libsql-client` / `libsql` packages?
  * They pull in extra dependencies (websockets, native wheels) and RAM.
  * The HTTP pipeline API is ~100 lines with httpx, which we already need for
    web search, and several statements can share ONE network round trip.

Wire format reference: Turso "Hrana over HTTP" (v2 pipeline).
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Sequence

import httpx

log = logging.getLogger("gateway.turso")

Statement = tuple[str, Sequence[Any]]   # (sql, positional args)
Row = dict[str, Any]


class TursoError(RuntimeError):
    """Raised when Turso rejects a statement or cannot be reached."""


# --------------------------------------------------------------------------
# Value encoding / decoding (Hrana typed values)
# --------------------------------------------------------------------------
def _encode(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):                      # bool is an int subclass
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}   # ints travel as strings
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "blob", "base64": base64.b64encode(bytes(value)).decode()}
    return {"type": "text", "value": str(value)}


def _decode(cell: dict[str, Any]) -> Any:
    kind = cell.get("type")
    if kind == "null":
        return None
    if kind == "integer":
        return int(cell["value"])
    if kind == "float":
        return float(cell["value"])
    if kind == "blob":
        b64 = cell["base64"]
        return base64.b64decode(b64 + "=" * (-len(b64) % 4))
    return cell.get("value")                         # text


class Turso:
    def __init__(
        self,
        url: str,
        token: str,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,   # injected by tests
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            # Few sockets = few buffers. One keep-alive connection is plenty.
            limits=httpx.Limits(max_connections=6, max_keepalive_connections=2),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    async def execute(self, sql: str, args: Sequence[Any] = ()) -> list[Row]:
        """Run one statement, return rows as dicts."""
        return (await self.batch([(sql, args)]))[0]

    async def batch(self, statements: Sequence[Statement]) -> list[list[Row]]:
        """
        Run several statements in ONE HTTP round trip (sequentially, on the same
        connection). Returns one row list per statement. Not transactional.
        """
        requests: list[dict[str, Any]] = [
            {"type": "execute", "stmt": {"sql": sql, "args": [_encode(a) for a in args]}}
            for sql, args in statements
        ]
        requests.append({"type": "close"})
        data = await self._post(requests)

        results = data.get("results", [])
        out: list[list[Row]] = []
        for i in range(len(statements)):
            item = results[i]
            if item.get("type") == "error":
                raise TursoError(item["error"].get("message", "unknown Turso error"))
            result = item["response"]["result"]
            cols = [c["name"] for c in result["cols"]]
            out.append([dict(zip(cols, map(_decode, row))) for row in result["rows"]])
        return out

    # ------------------------------------------------------------------
    async def _post(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in (1, 2):                       # one retry for transient failures
            try:
                resp = await self._client.post("/v2/pipeline", json={"requests": requests})
                if resp.status_code in (401, 403):
                    raise TursoError("Turso rejected the auth token (check TURSO_AUTH_TOKEN)")
                if resp.status_code >= 500 and attempt == 1:
                    last_exc = TursoError(f"Turso HTTP {resp.status_code}")
                    continue
                if resp.status_code >= 400:
                    raise TursoError(f"Turso HTTP {resp.status_code}: {resp.text[:300]}")
                return resp.json()
            except httpx.TransportError as exc:      # timeouts, resets, DNS...
                last_exc = exc
                log.warning("Turso transport error (attempt %d): %s", attempt, exc)
        raise TursoError(f"Turso unreachable: {last_exc}")
