-- =====================================================================
-- Turso (libSQL) schema for the Local Multi-Agent Framework gateway (v2).
--
-- Vectors are stored in Turso's native F32_BLOB(N) column type.
-- N MUST equal your local embedding model's output size
-- (nomic-embed-text = 768, all-minilm / bge-small = 384).
-- The gateway applies this file automatically on startup (replacing the
-- number with EMBED_DIM) and ALTERs older v1 tables to add the new columns.
-- Do not put semicolons inside comments or strings in this file: statements
-- are split on them.
-- =====================================================================

-- ---------------------------------------------------------------------
-- Long-term memory: code snippets, docs, web chunks, summaries, notes.
--
-- scope        'shared' = visible to every agent. Any other value is a
--              private agent/role scope (coder, security, ...). The main
--              agent searches with all_scopes=true and sees everything.
-- project_id   optional project tag (NULL = general knowledge).
-- parent_id    a chunk points at the row of its document (kind='doc').
-- chunk_index  position of the chunk inside its document.
-- title        short one-line label, handy for tiny "hints" kept in RAM.
-- content_hash sha256 of the text (shared scope) or of scope + text, so
--              saving the same text twice updates the row instead of duplicating.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT    NOT NULL UNIQUE,
    content      TEXT    NOT NULL,
    kind         TEXT    NOT NULL DEFAULT 'snippet',   -- snippet | doc | chunk | web | summary | note
    language     TEXT,                                 -- python, js, ...
    source       TEXT,                                 -- file path / URL / agent name
    metadata     TEXT    NOT NULL DEFAULT '{}',        -- JSON string
    scope        TEXT    NOT NULL DEFAULT 'shared',
    project_id   TEXT,
    parent_id    INTEGER,
    chunk_index  INTEGER,
    title        TEXT,
    embedding    F32_BLOB(768) NOT NULL,
    created_at   INTEGER NOT NULL DEFAULT (unixepoch()),
    updated_at   INTEGER NOT NULL DEFAULT (unixepoch())
);

-- ---------------------------------------------------------------------
-- Reflexion log: mistakes the AI made and the lesson learned.
-- error_hash dedupes repeats. occurrences counts how often the same
-- mistake recurred (a strong signal that the lesson isn't sticking).
-- attempts = highest retry number reached for this mistake.
-- resolved = 1 once a working fix is confirmed.
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
    attempts      INTEGER NOT NULL DEFAULT 1,
    resolved      INTEGER NOT NULL DEFAULT 0,
    embedding     F32_BLOB(768),
    first_seen    INTEGER NOT NULL DEFAULT (unixepoch()),
    last_seen     INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_memories_kind_lang ON memories (kind, language);

CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (scope, kind);

CREATE INDEX IF NOT EXISTS idx_memories_parent ON memories (parent_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_memories_project ON memories (project_id);

CREATE INDEX IF NOT EXISTS idx_error_log_last_seen ON error_log (last_seen);
