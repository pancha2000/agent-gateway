# Agent Memory Gateway v2

A tiny FastAPI service for a **1 GB VPS**. It is the "memory + research server" of a local AI setup:
the LLM and the embedding model run on your laptop; the VPS does web search, downloading,
PDF/HTML cleaning, chunking, duplicate checks and vector search against **Turso**.

```
 laptop (Ollama: chat model + small embedding model)
   │  JSON over HTTPS / Tailscale, header X-API-Key
   ▼
 VPS: this gateway (~50-70 MB RAM)  ──►  Turso (libSQL, native vector columns)
   └─► DuckDuckGo / Brave, web pages, PDFs
```

The VPS has **no ML libraries**. Embeddings are computed on the laptop (`laptop/gwclient.py`).

## What's new in v2

| Feature | How |
|---|---|
| Per-agent memory | `scope` column. `shared` = everyone. A role (`coder`, `security`...) sees shared + its own. The main agent sends `all_scopes=true` and sees all. |
| Projects | `project_id`: searches return that project + general (untagged) knowledge. |
| Documents | `kind=doc` record + `kind=chunk/web` rows pointing at it with `parent_id`. |
| Tiny "hints" | `/get_hint` with `snippet_chars`, then `GET /recall/{id}` for the full text. |
| Ingestion | `POST /ingest/prepare`: URL (HTML or **PDF**) or raw text -> clean chunks; flags chunks already stored. |
| Research | `POST /research`: memory first, otherwise web search + page download + chunking. |
| Learning | `error_log` tracks `attempts` and `resolved`; `POST /error_log/{id}/resolve`. |
| Escalation | `POST /escalation_package`: one markdown brief (task, error, code, attempts, relevant memory) for Claude/Gemini, or to save as `.txt`. |
| Safety | Retrieved web text is wrapped with an "untrusted data" note. |
| Upgrade-safe | v1 tables are migrated automatically (`ALTER TABLE ... ADD COLUMN`). |

## Deploy (Ubuntu VPS)

```bash
sudo mkdir -p /opt/agent-gateway && sudo chown $USER /opt/agent-gateway
git clone <your repo> /opt/agent-gateway && cd /opt/agent-gateway
python3 -m venv venv && venv/bin/pip install -r requirements.txt
cp .env.example .env && nano .env            # Turso URL/token, GATEWAY_API_KEY, EMBED_DIM
sudo cp deploy/agent-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now agent-gateway
curl http://127.0.0.1:8000/health
```

Check `deploy/agent-gateway.service` (user, paths) first. Add 1-2 GB of swap on a 1 GB box.
Expose it only through HTTPS (Nginx + certbot) or Tailscale/WireGuard; the API key must never travel over plain HTTP.

**EMBED_DIM must equal your embedding model's size** (`nomic-embed-text` = 768, `all-minilm` = 384).
If the Turso tables already exist with another size, recreate them (drop `memories` and `error_log`).

## Laptop

```bash
cd laptop && pip install -r requirements.txt
ollama pull nomic-embed-text
cp .env.example .env && nano .env
python chat.py
```

`gwclient.py` is the library you will build the agents on (`remember`, `hint`, `research`,
`ingest_url`, `ingest_file`, `log_error`, `escalation`, ...). Embeddings run on the CPU by default
(`EMBED_ON_CPU=1`) so they do not fight the chat model for VRAM.

## API (all need `X-API-Key`, except `/health`)

| Endpoint | Purpose |
|---|---|
| `POST /save_memory` | One text + vector. Same text again = update. |
| `POST /save_memory_batch` | Up to 40 chunks, optional `parent` (document record). |
| `POST /get_hint` | Vector search (+ similar past mistakes) -> hits + ready-to-paste `context`. |
| `GET /recall/{id}?children=true` | Full memory (and its chunks). |
| `DELETE /memory/{id}` | Delete a memory and its chunks. |
| `GET /stats` | Counts per scope/kind. |
| `POST /error_log`, `POST /error_log/{id}/resolve` | Reflexion log. |
| `POST /escalation_package` | Brief for a stronger model / `.txt` for you. |
| `POST /search_web` | DuckDuckGo/Brave search, optional page text. |
| `POST /ingest/prepare` | URL/PDF/text -> chunks (nothing stored). |
| `POST /research` | Memory first, then web; returns chunks to embed. |
| `GET /health` | Liveness, RSS, `embed_dim`. |

### The research flow

1. Laptop embeds the query and calls `/research`.
2. VPS searches memory. A strong hit (`>= web_threshold`, default 0.6) is returned, nothing goes to the web.
3. Otherwise the VPS searches the web, downloads the top pages/PDFs, extracts text, chunks it and marks known chunks (`exists`).
4. Laptop embeds only the new chunks and sends them to `/save_memory_batch`. Next time the answer comes from memory.

### Sizing notes

- Search is an exact scan by default. Turso bills rows read, so with many thousands of rows set `USE_VECTOR_INDEX=true` (approximate, cheaper). Check Turso's current free-tier limits.
- PDFs larger than `PDF_MAX_BYTES` and scanned PDFs without a text layer are skipped.
- Cloud models (Claude/Gemini) may refuse offensive-security content: only escalate general code problems, and use `include_memory=false` when database content must not leave your machine.

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest -q
```
The suite runs fully offline against a fake Turso (SQLite) and a mocked web.
