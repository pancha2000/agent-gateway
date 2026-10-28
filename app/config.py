"""
Runtime configuration.

Everything comes from environment variables (systemd's EnvironmentFile= in
production). A tiny built-in .env loader is included for local development so
we don't need the python-dotenv dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from `path` without overriding real env vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Turso ---------------------------------------------------------
    turso_url: str          # https:// form of the libsql:// URL
    turso_token: str
    # --- Gateway auth --------------------------------------------------
    api_key: str
    # --- Vectors -------------------------------------------------------
    embed_dim: int
    use_vector_index: bool
    # --- Web search ----------------------------------------------------
    web_timeout: float
    web_max_page_bytes: int
    web_max_page_chars: int
    brave_api_key: str | None   # optional: more reliable than scraping DDG
    # --- Misc ----------------------------------------------------------
    enable_docs: bool
    max_body_bytes: int
    schema_path: Path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")

    url = os.getenv("TURSO_DATABASE_URL", "").strip()
    token = os.getenv("TURSO_AUTH_TOKEN", "").strip()
    api_key = os.getenv("GATEWAY_API_KEY", "").strip()
    missing = [
        n for n, v in (
            ("TURSO_DATABASE_URL", url),
            ("TURSO_AUTH_TOKEN", token),
            ("GATEWAY_API_KEY", api_key),
        ) if not v
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if len(api_key) < 16:
        raise RuntimeError("GATEWAY_API_KEY is too short (use at least 16 random characters).")

    # Turso hands out libsql:// URLs; the HTTP API lives on the https:// twin.
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://"):]
    url = url.rstrip("/")

    return Settings(
        turso_url=url,
        turso_token=token,
        api_key=api_key,
        embed_dim=int(os.getenv("EMBED_DIM", "384")),
        use_vector_index=_bool("USE_VECTOR_INDEX", False),
        web_timeout=float(os.getenv("WEB_TIMEOUT_SECONDS", "8")),
        web_max_page_bytes=int(os.getenv("WEB_MAX_PAGE_BYTES", "400000")),
        web_max_page_chars=int(os.getenv("WEB_MAX_PAGE_CHARS", "4000")),
        brave_api_key=os.getenv("BRAVE_API_KEY", "").strip() or None,
        enable_docs=_bool("ENABLE_DOCS", False),
        max_body_bytes=int(os.getenv("MAX_BODY_BYTES", str(1_000_000))),
        schema_path=Path(os.getenv("SCHEMA_PATH", str(PROJECT_ROOT / "schema.sql"))),
    )
