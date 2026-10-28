-- =====================================================================
-- Turso (libSQL) schema for the Local Multi-Agent Framework gateway.
--
-- Vectors are stored in Turso's native F32_BLOB(N) column type.
-- N MUST equal your local embedding model's output size (default 384).
-- The gateway applies this file automatically on startup (replacing 384
-- with EMBED_DIM), but you can also paste it into `turso db shell <db>`.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Long-term memory: code snippets, docs, notes.
-- content_hash (sha256 of the text) makes saves idempotent: saving the
-- same text twice updates the row instead of duplicating it.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT    NOT NULL UNIQUE,
    content      TEXT    NOT NULL,
    kind         TEXT    NOT NULL DEFAULT 'snippet',   -- snippet | doc | note | ...
    language     TEXT,                                 -- python, js, ...
    source       TEXT,                                 -- file path / URL / agent name
    metadata     TEXT    NOT NULL DEFAULT '{}',        -- JSON string
    embedding    F32_BLOB(384) NOT NULL,
    created_at   INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at   INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_memories_kind_lang ON memories (kind, language);

-- ---------------------------------------------------------------------
-- Reflexion log: mistakes the AI made and the lesson learned.
-- error_hash dedupes repeats; `occurrences` counts how often the same
-- mistake recurred (a strong signal that the lesson isn't sticking).
-- embedding is optional: if provided, past errors are surfaced by
-- /get_hint when a similar task comes up.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS error_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    error_hash    TEXT    NOT NULL UNIQUE,
    task          TEXT    NOT NULL,
    error_message TEXT    NOT NULL,
    root_cause    TEXT,
    lesson        TEXT,      -- "what to do differently next time"
    fix           TEXT,      -- the working fix, if found
    agent         TEXT,      -- orchestrator | coder | ...
    tags          TEXT    NOT NULL DEFAULT '[]',       -- JSON array string
    occurrences   INTEGER NOT NULL DEFAULT 1,
    embedding     F32_BLOB(384),
    first_seen    INTEGER NOT NULL DEFAULT (unixepoch()),
    last_seen     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_error_log_last_seen ON error_log (last_seen);
