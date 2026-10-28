# Agent Memory Gateway (FastAPI + Turso)

Tiny VPS backend for a local multi-agent setup. The VPS never runs models: your PC
computes embeddings and sends float arrays; Turso stores them and does the cosine search.

```
agent-gateway/
├── app/
│   ├── main.py          # app factory, startup (schema init), /health
│   ├── config.py        # env vars -> Settings
│   ├── security.py      # X-API-Key auth
│   ├── turso.py         # async Turso client (Hrana over HTTP, httpx only)
│   ├── repo.py          # ALL SQL: schema bootstrap, memories, error log
│   ├── vectors.py       # vector validation/serialisation (no numpy)
│   ├── websearch.py     # DuckDuckGo/Brave + page text extraction (stdlib parser)
│   ├── schemas.py       # request/response models
│   └── routers/         # memory.py (/save_memory, /get_hint), errors.py, web.py
├── schema.sql           # Turso tables (F32_BLOB vectors)
├── requirements.txt     # fastapi, uvicorn, httpx
├── .env.example
├── deploy/agent-gateway.service   # systemd unit, MemoryMax=300M
└── tests/               # offline tests with a fake Turso
```

## Deploy on the Oracle VPS

```bash
sudo mkdir -p /opt/agent-gateway && sudo chown $USER /opt/agent-gateway
# copy the project there, then:
cd /opt/agent-gateway
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env        # Turso URL/token, API key, EMBED_DIM
sudo cp deploy/agent-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now agent-gateway
curl localhost:8000/health               # -> {"status":"ok","rss_mb": ...}
```

The service binds to 127.0.0.1. Expose it with a TLS reverse proxy (Caddy is the
lightest: `your.domain { reverse_proxy 127.0.0.1:8000 }`), or keep it private with
Tailscale / an SSH tunnel. On Oracle Cloud also open the port in the VCN security
list and in the instance firewall (`sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT`).
A 1 GB swap file is cheap insurance on a 1 GB box.

## API (header `X-API-Key: <GATEWAY_API_KEY>` on everything except /health)

| Endpoint | Body | Returns |
|---|---|---|
| `POST /save_memory` | `text`, `vector`, optional `kind`, `language`, `source`, `metadata` | `{id, created}` |
| `POST /get_hint` | `vector`, optional `top_k`, `min_similarity`, `kind`, `language`, `include_errors`, `max_chars` | `{hits, past_errors, context}` |
| `POST /error_log` | `task`, `error_message`, optional `root_cause`, `lesson`, `fix`, `agent`, `tags`, `vector` | `{id, occurrences, is_repeat}` |
| `POST /search_web` | `query`, optional `max_results`, `fetch_top` (0-3) | `{engine, results, context}` |

`context` in `/get_hint` and `/search_web` is a ready-to-paste block for the local LLM prompt.

Local PC example:

```python
import httpx
c = httpx.Client(base_url="https://your.domain", headers={"X-API-Key": KEY})
c.post("/save_memory", json={"text": code, "vector": embed(code), "language": "python"})
hint = c.post("/get_hint", json={"vector": embed(task), "top_k": 4}).json()["context"]
```

## Notes

- **EMBED_DIM must match your embedding model** (MiniLM/bge-small = 384, nomic = 768).
  Changing it later means recreating the tables (the gateway logs an error if they disagree).
- `USE_VECTOR_INDEX=false` does an exact scan; fine to ~50k rows. Switch to `true`
  for larger tables (Turso DiskANN index, approximate results).
- DuckDuckGo often blocks datacenter IPs. If `/search_web` returns 502, set `BRAVE_API_KEY`.
- Fetched pages are SSRF-guarded (public IPs only, ports 80/443, redirects re-checked).
- Offline tests: `pip install pytest && python -m pytest -q tests`.
