-- Conversation state and the retrieval audit trail.
--
-- chat_context_chunks records exactly which chunks were put in front of the
-- model for each answer. Without it, a wrong answer cannot be diagnosed: you
-- cannot tell whether retrieval missed the evidence or generation ignored it.

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT,
    -- Memory is maintained locally rather than by an LLM call; see chat/memory.py.
    summary     TEXT        NOT NULL DEFAULT '',
    facts       JSONB       NOT NULL DEFAULT '[]'::jsonb,
    subject     TEXT,
    turns       INT         NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id BIGINT NOT NULL REFERENCES chat_sessions (id) ON DELETE CASCADE,
    role       TEXT   NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content    TEXT   NOT NULL,
    -- Interpretation, answer mode, guard verdict and citation audit, so a turn
    -- can be replayed and explained after the fact.
    meta       JSONB  NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chat_context_chunks (
    id         BIGSERIAL PRIMARY KEY,
    message_id BIGINT NOT NULL REFERENCES chat_messages (id) ON DELETE CASCADE,
    -- No FK to chunks: re-chunking replaces rows, and the audit trail must
    -- survive that. The chunk text is denormalised for the same reason.
    chunk_id    BIGINT,
    document_id BIGINT,
    ordinal     INT    NOT NULL,
    title       TEXT,
    url         TEXT,
    score       DOUBLE PRECISION,
    was_cited   BOOLEAN NOT NULL DEFAULT FALSE,
    text        TEXT
);

CREATE INDEX IF NOT EXISTS chat_messages_session_idx
    ON chat_messages (session_id, created_at);
CREATE INDEX IF NOT EXISTS chat_context_message_idx
    ON chat_context_chunks (message_id);

CREATE OR REPLACE FUNCTION touch_updated_at() RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chat_sessions_touch ON chat_sessions;
CREATE TRIGGER chat_sessions_touch
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
