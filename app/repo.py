"""
All SQL lives here, so the routers stay thin and the queries are easy to audit.

Cosine similarity = 1 - vector_distance_cos(a, b)   (Turso returns a DISTANCE).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .config import Settings
from .turso import Turso
from .vectors import sha256, to_text

log = logging.getLogger("gateway.repo")

VECTOR_INDEX_NAME = "idx_memories_embedding"


# =========================================================================
# Schema bootstrap
# =========================================================================
def _schema_statements(settings: Settings) -> list[str]:
    """Read schema.sql, patch the vector size to EMBED_DIM, split into statements."""
    sql = settings.schema_path.read_text(encoding="utf-8")
    sql = re.sub(r"F32_BLOB\(\d+\)", f"F32_BLOB({settings.embed_dim})", sql)
    sql = re.sub(r"--[^\n]*", "", sql)                       # drop comments
    return [s.strip() for s in sql.split(";") if s.strip()]


async def init_schema(db: Turso, settings: Settings) -> None:
    """Idempotent: every statement is CREATE ... IF NOT EXISTS."""
    stmts = _schema_statements(settings)
    if settings.use_vector_index:
        stmts.append(
            f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX_NAME} "
            "ON memories (libsql_vector_idx(embedding, 'metric=cosine'))"
        )
    await db.batch([(s, ()) for s in stmts])

    # Guard against the classic mistake: changing EMBED_DIM after tables exist.
    rows = await db.execute("SELECT sql FROM sqlite_master WHERE name = 'memories'")
    match = re.search(r"F32_BLOB\((\d+)\)", rows[0]["sql"]) if rows else None
    if match and int(match.group(1)) != settings.embed_dim:
        log.error(
            "EMBED_DIM=%d but the existing `memories` table uses F32_BLOB(%s). "
            "Inserts will fail until you recreate the tables or fix EMBED_DIM.",
            settings.embed_dim, match.group(1),
        )


# =========================================================================
# Memories
# =========================================================================
async def save_memory(
    db: Turso, *, text: str, vector: list[float], kind: str,
    language: str | None, source: str | None, metadata: dict[str, Any],
) -> tuple[int, bool]:
    """Upsert by content hash. Returns (id, created)."""
    h = sha256(text.strip())
    existing, upserted = await db.batch([
        ("SELECT id FROM memories WHERE content_hash = ?", (h,)),
        (
            """
            INSERT INTO memories (content_hash, content, kind, language, source, metadata, embedding)
            VALUES (?, ?, ?, ?, ?, ?, vector32(?))
            ON CONFLICT(content_hash) DO UPDATE SET
                kind       = excluded.kind,
                language   = excluded.language,
                source     = excluded.source,
                metadata   = excluded.metadata,
                embedding  = excluded.embedding,
                updated_at = unixepoch()
            RETURNING id
            """,
            (h, text, kind, language, source, json.dumps(metadata, ensure_ascii=False), to_text(vector)),
        ),
    ])
    return upserted[0]["id"], not existing


_MEMORY_COLS = "id, content, kind, language, source, metadata"


def _memory_search_stmt(
    settings: Settings, vector: list[float], top_k: int, kind: str | None, language: str | None,
) -> tuple[str, list[Any]]:
    vec = to_text(vector)
    filters = "(? IS NULL OR kind = ?) AND (? IS NULL OR language = ?)"
    fargs = [kind, kind, language, language]

    if settings.use_vector_index:
        # Approximate nearest neighbours via Turso's DiskANN index. The index
        # knows nothing about our filters, so over-fetch and filter afterwards.
        fetch = top_k * 5 if (kind or language) else top_k
        sql = f"""
            SELECT m.id, m.content, m.kind, m.language, m.source, m.metadata,
                   vector_distance_cos(m.embedding, vector32(?)) AS dist
            FROM vector_top_k('{VECTOR_INDEX_NAME}', vector32(?), {int(fetch)}) AS t
            JOIN memories m ON m.rowid = t.id
            WHERE {filters.replace('kind', 'm.kind').replace('language', 'm.language')}
            ORDER BY dist ASC LIMIT {int(top_k)}
        """
        return sql, [vec, vec, *fargs]

    # Exact brute-force scan: perfect recall, fine up to ~50k rows.
    sql = f"""
        SELECT {_MEMORY_COLS}, vector_distance_cos(embedding, vector32(?)) AS dist
        FROM memories
        WHERE {filters}
        ORDER BY dist ASC LIMIT {int(top_k)}
    """
    return sql, [vec, *fargs]


def _error_search_stmt(vector: list[float], limit: int) -> tuple[str, list[Any]]:
    sql = f"""
        SELECT id, task, error_message, root_cause, lesson, fix, occurrences,
               vector_distance_cos(embedding, vector32(?)) AS dist
        FROM error_log
        WHERE embedding IS NOT NULL
        ORDER BY dist ASC LIMIT {int(limit)}
    """
    return sql, [to_text(vector)]


async def get_hint(
    db: Turso, settings: Settings, *, vector: list[float], top_k: int, min_similarity: float,
    kind: str | None, language: str | None, include_errors: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (memory_hits, past_errors) - both searched in ONE round trip."""
    stmts = [_memory_search_stmt(settings, vector, top_k, kind, language)]
    if include_errors:
        stmts.append(_error_search_stmt(vector, 3))
    results = await db.batch(stmts)

    hits = []
    for r in results[0]:
        sim = 1.0 - float(r["dist"])
        if sim >= min_similarity:
            r["metadata"] = _loads(r["metadata"], {})
            r["similarity"] = round(sim, 4)
            hits.append(r)

    errors = []
    if include_errors:
        for r in results[1]:
            sim = 1.0 - float(r["dist"])
            if sim >= min_similarity:
                r["similarity"] = round(sim, 4)
                errors.append(r)
    return hits, errors


# =========================================================================
# Reflexion error log
# =========================================================================
def _normalise(s: str) -> str:
    return " ".join(s.lower().split())


async def log_error(
    db: Turso, *, task: str, error_message: str, root_cause: str | None, lesson: str | None,
    fix: str | None, agent: str | None, tags: list[str], vector: list[float] | None,
) -> tuple[int, int]:
    """Upsert a mistake. Repeats bump `occurrences`. Returns (id, occurrences)."""
    h = sha256(_normalise(task) + "\n" + _normalise(error_message))
    emb_expr = "vector32(?)" if vector else "NULL"
    args: list[Any] = [h, task, error_message, root_cause, lesson, fix, agent,
                       json.dumps(tags, ensure_ascii=False)]
    if vector:
        args.append(to_text(vector))

    rows = await db.execute(
        f"""
        INSERT INTO error_log
            (error_hash, task, error_message, root_cause, lesson, fix, agent, tags, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, {emb_expr})
        ON CONFLICT(error_hash) DO UPDATE SET
            occurrences = occurrences + 1,
            last_seen   = unixepoch(),
            root_cause  = COALESCE(excluded.root_cause, root_cause),
            lesson      = COALESCE(excluded.lesson, lesson),
            fix         = COALESCE(excluded.fix, fix),
            agent       = COALESCE(excluded.agent, agent),
            tags        = excluded.tags,
            embedding   = COALESCE(excluded.embedding, embedding)
        RETURNING id, occurrences
        """,
        args,
    )
    return rows[0]["id"], rows[0]["occurrences"]


# =========================================================================
def _loads(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default
