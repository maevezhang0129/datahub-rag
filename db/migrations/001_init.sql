-- datahub-rag base schema: documents + chunks.
-- Embedding tables are created per model by ensure_embedding_table() in
-- src/datahub_rag/store.py, so several models can be indexed side by side.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT        NOT NULL,
    source_id     TEXT        NOT NULL,
    url           TEXT,
    title         TEXT        NOT NULL DEFAULT '',
    body          TEXT        NOT NULL DEFAULT '',
    published_at  DATE,
    -- sha256 of title+body; lets ingest skip unchanged documents and lets
    -- chunking detect that a document was revised and needs re-chunking.
    content_hash  TEXT        NOT NULL,
    meta          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_id)
);

CREATE INDEX IF NOT EXISTS documents_published_at_idx ON documents (published_at DESC);

-- ------------------------------------------------------------------- chunks
-- title is denormalised from documents so that the full-text vector can be a
-- STORED generated column (generated columns cannot reference other tables).
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal      INT    NOT NULL,
    title        TEXT   NOT NULL DEFAULT '',
    text         TEXT   NOT NULL,
    token_count  INT    NOT NULL,
    -- content_hash of the parent document at the time this chunk was cut.
    -- Chunking skips a document whose chunks already carry its current hash.
    source_hash  TEXT   NOT NULL,
    fts tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(text,  '')), 'B')
    ) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, source_hash, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_fts_idx         ON chunks USING GIN (fts);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_source_hash_idx ON chunks (source_hash);
