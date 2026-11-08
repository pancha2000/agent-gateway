"""
All SQL lives here, so the routers stay thin and the queries are easy to audit.

Cosine similarity = 1 - vector_distance_cos(a, b)   (Turso returns a DISTANCE).

Visibility model
----------------
Every memory has a `scope`. 'shared' rows are visible to everyone. A caller with
scope "coder" sees 'shared' + 'coder'. The main agent passes all_scopes=True and
sees everything.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from .config import Settings
from .turso import Turso
from .vectors import sha256, to_text

log = logging.getLogger("gateway.repo")

VECTOR_INDEX_NAME = "idx_memories_embedding"

MEMORY_COLS = (
    "id", "content", "kind", "language", "source", "metadata",
    "scope", "project_id", "parent_id", "chunk_index", "title",
)

# Columns added after v1. init_schema() ALTERs old tables to include them.
_NEW_COLUMNS: dict[str, dict[str, str]] = {
    "memories": {
        "scope": "TEXT NOT NULL DEFAULT 'shared'",
        "project_id": "TEXT",
        "parent_id": "INTEGER",
        "chunk_index": "INTEGER",
        "title": "TEXT",
    },
    "error_log": {
        "attempts": "INTEGER NOT NULL DEFAULT 1",
        "resolved": "INTEGER NOT NULL DEFAULT 0",
    },
}


def _cols(prefix: str = "") -> str:
    return ", ".join(prefix + c for c in MEMORY_COLS)


# =========================================================================
# Schema bootstrap + migration
# =========================================================================
def _schema_statements(settings: Settings) -> list[str]:
    """Read schema.sql, patch the vector size to EMBED_DIM, split into statements."""
    sql = settings.schema_path.read_text(encoding="utf-8")
    sql = re.sub(r"F32_BLOB\(\d+\)", f"F32_BLOB({settings.embed_dim})", sql)
    sql = re.sub(r"--[^\n]*", "", sql)                       # drop comments
    return [s.strip() for s in sql.split(";") if s.strip()]


async def _migrate(db: Turso) -> None:
    """Add v2 columns to tables that were created by the v1 gateway."""
    for table, columns in _NEW_COLUMNS.items():
        rows = await db.execute(f"PRAGMA table_info({table})")
        have = {r["name"] for r in rows}
        for name, ddl in columns.items():
            if name not in have:
                log.info("migrating: ALTER TABLE %s ADD COLUMN %s", table, name)
                await db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


async def init_schema(db: Turso, settings: Settings) -> None:
    """Idempotent: tables first, then column migration, then indexes."""
    stmts = _schema_statements(settings)
    tables = [s for s in stmts if s.upper().startswith("CREATE TABLE")]
    others = [s for s in stmts if not s.upper().startswith("CREATE TABLE")]

    await db.batch([(s, ()) for s in tables])
    await _migrate(db)                       # indexes below need the migrated columns

    if settings.use_vector_index:
        others.append(
            f"CREATE INDEX IF NOT EXISTS {VECTOR_INDEX_NAME} "
            "ON memories (libsql_vector_idx(embedding, 'metric=cosine'))"
        )
    if others:
        await db.batch([(s, ()) for s in others])

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
# Helpers
# =========================================================================
def memory_hash(scope: str, text: str) -> str:
    """
    Dedupe key. 'shared' keeps the v1 formula (sha256 of the text) so old rows keep
    matching; private scopes mix the scope in, so the same text may exist per scope.
    """
    t = text.strip()
    return sha256(t if scope == "shared" else f"{scope}\x00{t}")


def _visibility(prefix: str, scope: str, all_scopes: bool) -> tuple[list[str], list[Any]]:
    if all_scopes:
        return [], []
    return [f"{prefix}scope IN ('shared', ?)"], [scope]


def _loads(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _row(r: dict[str, Any]) -> dict[str, Any]:
    r["metadata"] = _loads(r.get("metadata"), {})
    return r


# =========================================================================
# Memories
# =========================================================================
_UPSERT_SQL = """
    INSERT INTO memories
        (content_hash, content, kind, language, source, metadata,
         scope, project_id, parent_id, chunk_index, title, embedding)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, vector32(?))
    ON CONFLICT(content_hash) DO UPDATE SET
        kind        = excluded.kind,
        language    = COALESCE(excluded.language, language),
        source      = COALESCE(excluded.source, source),
        metadata    = excluded.metadata,
        project_id  = COALESCE(excluded.project_id, project_id),
        parent_id   = COALESCE(excluded.parent_id, parent_id),
        chunk_index = COALESCE(excluded.chunk_index, chunk_index),
        title       = COALESCE(excluded.title, title),
        embedding   = excluded.embedding,
        updated_at  = unixepoch()
    RETURNING id
"""


def _upsert_args(
    h: str, text: str, *, kind: str, language: str | None, source: str | None,
    metadata: dict[str, Any], scope: str, project_id: str | None, parent_id: int | None,
    chunk_index: int | None, title: str | None, vector: Sequence[float],
) -> tuple[Any, ...]:
    return (h, text, kind, language, source, json.dumps(metadata, ensure_ascii=False),
            scope, project_id, parent_id, chunk_index, title, to_text(vector))


async def save_memory(
    db: Turso, *, text: str, vector: list[float], kind: str,
    language: str | None, source: str | None, metadata: dict[str, Any],
    scope: str = "shared", project_id: str | None = None, parent_id: int | None = None,
    chunk_index: int | None = None, title: str | None = None,
) -> tuple[int, bool]:
    """Upsert by content hash. Returns (id, created)."""
    t = text.strip()
    h = memory_hash(scope, t)
    existing, upserted = await db.batch([
        ("SELECT id FROM memories WHERE content_hash = ?", (h,)),
        (_UPSERT_SQL, _upsert_args(
            h, t, kind=kind, language=language, source=source, metadata=metadata, scope=scope,
            project_id=project_id, parent_id=parent_id, chunk_index=chunk_index,
            title=title, vector=vector)),
    ])
    return upserted[0]["id"], not existing


async def save_memories(
    db: Turso, items: Sequence[dict[str, Any]], *, scope: str, project_id: str | None,
    parent_id: int | None, kind: str,
) -> list[tuple[int, bool]]:
    """
    Upsert many memories in ONE round trip. Each item is a dict with `text`, `vector`
    and optional kind/language/source/metadata/chunk_index/title.
    Returns [(id, created), ...] in the same order as `items`.
    """
    prepared = []
    for it in items:
        t = it["text"].strip()
        prepared.append((memory_hash(scope, t), t, it))

    hashes = sorted({h for h, _, _ in prepared})
    stmts: list[tuple[str, Sequence[Any]]] = [(
        f"SELECT content_hash FROM memories WHERE content_hash IN ({','.join('?' * len(hashes))})",
        hashes,
    )]
    for h, t, it in prepared:
        stmts.append((_UPSERT_SQL, _upsert_args(
            h, t, kind=it.get("kind") or kind, language=it.get("language"), source=it.get("source"),
            metadata=it.get("metadata") or {}, scope=scope, project_id=project_id,
            parent_id=parent_id, chunk_index=it.get("chunk_index"), title=it.get("title"),
            vector=it["vector"])))

    results = await db.batch(stmts)
    have = {r["content_hash"] for r in results[0]}
    return [(rows[0]["id"], h not in have) for (h, _, _), rows in zip(prepared, results[1:])]


async def existing_hashes(db: Turso, hashes: Sequence[str]) -> set[str]:
    """Which of these content hashes are already stored? (lets clients skip re-embedding)"""
    unique = sorted(set(hashes))
    if not unique:
        return set()
    rows = await db.execute(
        f"SELECT content_hash FROM memories WHERE content_hash IN ({','.join('?' * len(unique))})",
        unique,
    )
    return {r["content_hash"] for r in rows}


def _memory_search_stmt(
    settings: Settings, vector: list[float], top_k: int, *, kind: str | None,
    language: str | None, scope: str, all_scopes: bool, project_id: str | None,
) -> tuple[str, list[Any]]:
    vec = to_text(vector)
    use_idx = settings.use_vector_index
    p = "m." if use_idx else ""

    where, fargs = _visibility(p, scope, all_scopes)
    if kind:
        where.append(f"{p}kind = ?")
        fargs.append(kind)
    if language:
        where.append(f"{p}language = ?")
        fargs.append(language)
    if project_id:                       # this project's notes + general (untagged) knowledge
        where.append(f"({p}project_id = ? OR {p}project_id IS NULL)")
        fargs.append(project_id)
    cond = " AND ".join(where) or "1 = 1"

    if use_idx:
        # Approximate nearest neighbours via Turso's DiskANN index. The index
        # knows nothing about our filters, so over-fetch and filter afterwards.
        fetch = top_k * 5 if where else top_k
        sql = f"""
            SELECT {_cols('m.')}, vector_distance_cos(m.embedding, vector32(?)) AS dist
            FROM vector_top_k('{VECTOR_INDEX_NAME}', vector32(?), {int(fetch)}) AS t
            JOIN memories m ON m.rowid = t.id
            WHERE {cond}
            ORDER BY dist ASC LIMIT {int(top_k)}
        """
        return sql, [vec, vec, *fargs]

    # Exact brute-force scan: perfect recall, fine up to ~50k rows.
    sql = f"""
        SELECT {_cols()}, vector_distance_cos(embedding, vector32(?)) AS dist
        FROM memories
        WHERE {cond}
        ORDER BY dist ASC LIMIT {int(top_k)}
    """
    return sql, [vec, *fargs]


def _error_search_stmt(vector: list[float], limit: int) -> tuple[str, list[Any]]:
    sql = f"""
        SELECT id, task, error_message, root_cause, lesson, fix, occurrences, attempts, resolved,
               vector_distance_cos(embedding, vector32(?)) AS dist
        FROM error_log
        WHERE embedding IS NOT NULL
        ORDER BY dist ASC LIMIT {int(limit)}
    """
    return sql, [to_text(vector)]


async def get_hint(
    db: Turso, settings: Settings, *, vector: list[float], top_k: int, min_similarity: float,
    kind: str | None, language: str | None, include_errors: bool,
    scope: str = "shared", all_scopes: bool = False, project_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (memory_hits, past_errors) - both searched in ONE round trip."""
    stmts = [_memory_search_stmt(
        settings, vector, top_k, kind=kind, language=language,
        scope=scope, all_scopes=all_scopes, project_id=project_id)]
    if include_errors:
        stmts.append(_error_search_stmt(vector, 3))
    results = await db.batch(stmts)

    hits = []
    for r in results[0]:
        sim = 1.0 - float(r["dist"])
        if sim >= min_similarity:
            _row(r)
            r["similarity"] = round(sim, 4)
            hits.append(r)

    errors = []
    if include_errors:
        for r in results[1]:
            sim = 1.0 - float(r["dist"])
            if sim >= min_similarity:
                r["similarity"] = round(sim, 4)
                r["resolved"] = bool(r["resolved"])
                errors.append(r)
    return hits, errors


async def get_memory(
    db: Turso, memory_id: int, *, scope: str, all_scopes: bool,
) -> dict[str, Any] | None:
    """Full row by id (respecting scope visibility), or None."""
    where, args = _visibility("", scope, all_scopes)
    cond = " AND ".join(["id = ?", *where])
    rows = await db.execute(
        f"SELECT {_cols()}, created_at, updated_at FROM memories WHERE {cond}",
        [memory_id, *args],
    )
    return _row(rows[0]) if rows else None


async def get_children(
    db: Turso, parent_id: int, *, scope: str, all_scopes: bool, limit: int,
) -> list[dict[str, Any]]:
    """Chunks that belong to a document row, in document order."""
    where, args = _visibility("", scope, all_scopes)
    cond = " AND ".join(["parent_id = ?", *where])
    rows = await db.execute(
        f"""SELECT {_cols()}, created_at, updated_at FROM memories
            WHERE {cond} ORDER BY chunk_index, id LIMIT {int(limit)}""",
        [parent_id, *args],
    )
    return [_row(r) for r in rows]


async def delete_memory(
    db: Turso, memory_id: int, *, scope: str, all_scopes: bool,
) -> list[int]:
    """Delete a memory AND the chunks that point at it. Returns the deleted ids."""
    where, args = _visibility("", scope, all_scopes)
    cond = " AND ".join(["(id = ? OR parent_id = ?)", *where])
    rows = await db.execute(
        f"DELETE FROM memories WHERE {cond} RETURNING id",
        [memory_id, memory_id, *args],
    )
    return [r["id"] for r in rows]


async def stats(db: Turso) -> dict[str, Any]:
    mem, err = await db.batch([
        ("SELECT scope, kind, COUNT(*) AS n FROM memories GROUP BY scope, kind ORDER BY scope, kind", ()),
        ("SELECT COUNT(*) AS total, "
         "COALESCE(SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END), 0) AS unresolved FROM error_log", ()),
    ])
    return {
        "memories": [{"scope": r["scope"], "kind": r["kind"], "n": r["n"]} for r in mem],
        "errors": {"total": err[0]["total"], "unresolved": err[0]["unresolved"]},
    }


# =========================================================================
# Reflexion error log
# =========================================================================
def _normalise(s: str) -> str:
    return " ".join(s.lower().split())


async def log_error(
    db: Turso, *, task: str, error_message: str, root_cause: str | None, lesson: str | None,
    fix: str | None, agent: str | None, tags: list[str], vector: list[float] | None,
    attempt: int = 1, resolved: bool = False,
) -> dict[str, Any]:
    """
    Upsert a mistake. Repeats bump `occurrences`; `attempts` keeps the highest retry
    number seen; `resolved` follows the latest report.
    Returns {id, occurrences, attempts, resolved}.
    """
    h = sha256(_normalise(task) + "\n" + _normalise(error_message))
    emb_expr = "vector32(?)" if vector else "NULL"
    args: list[Any] = [h, task, error_message, root_cause, lesson, fix, agent,
                       json.dumps(tags, ensure_ascii=False), attempt, int(resolved)]
    if vector:
        args.append(to_text(vector))

    rows = await db.execute(
        f"""
        INSERT INTO error_log
            (error_hash, task, error_message, root_cause, lesson, fix, agent, tags,
             attempts, resolved, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {emb_expr})
        ON CONFLICT(error_hash) DO UPDATE SET
            occurrences = occurrences + 1,
            last_seen   = unixepoch(),
            root_cause  = COALESCE(excluded.root_cause, root_cause),
            lesson      = COALESCE(excluded.lesson, lesson),
            fix         = COALESCE(excluded.fix, fix),
            agent       = COALESCE(excluded.agent, agent),
            tags        = excluded.tags,
            attempts    = MAX(attempts, excluded.attempts),
            resolved    = excluded.resolved,
            embedding   = COALESCE(excluded.embedding, embedding)
        RETURNING id, occurrences, attempts, resolved
        """,
        args,
    )
    r = rows[0]
    return {"id": r["id"], "occurrences": r["occurrences"],
            "attempts": r["attempts"], "resolved": bool(r["resolved"])}


async def resolve_error(
    db: Turso, error_id: int, *, fix: str | None, lesson: str | None,
) -> dict[str, Any] | None:
    """Mark a logged mistake as fixed (optionally storing the fix / lesson)."""
    rows = await db.execute(
        """
        UPDATE error_log
        SET resolved = 1, fix = COALESCE(?, fix), lesson = COALESCE(?, lesson),
            last_seen = unixepoch()
        WHERE id = ?
        RETURNING id, occurrences, attempts, resolved
        """,
        [fix, lesson, error_id],
    )
    if not rows:
        return None
    r = rows[0]
    return {"id": r["id"], "occurrences": r["occurrences"],
            "attempts": r["attempts"], "resolved": bool(r["resolved"])}
