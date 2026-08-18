"""Database access: connections, migrations, and per-model embedding tables."""

from __future__ import annotations

import hashlib
import pathlib
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from . import config

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parents[2] / "db" / "migrations"


def content_hash(*parts: str) -> str:
    """Stable content fingerprint used for change detection and rechunk skipping."""
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x00")  # delimiter, so ("ab","c") != ("a","bc")
    return h.hexdigest()


@contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(config.database_url(), row_factory=dict_row)
    conn.autocommit = autocommit
    try:
        yield conn
        if not autocommit:
            conn.commit()
    except Exception:
        if not autocommit:
            conn.rollback()
        raise
    finally:
        conn.close()


def run_migrations() -> list[str]:
    """Apply every .sql file in db/migrations in name order.

    The migrations are written to be idempotent (IF NOT EXISTS throughout), so
    re-running is safe and no migration-tracking table is needed at this size.
    """
    applied = []
    with connect() as conn:
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            conn.execute(path.read_text())
            applied.append(path.name)
    return applied


def ensure_embedding_table(model_key: str) -> str:
    """Create the embedding table and HNSW index for `model_key` if absent.

    Returns the table name. Idempotent.
    """
    spec = config.get_model(model_key)
    table = config.embedding_table(model_key)
    dim = spec["dim"]

    with connect() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                chunk_id  BIGINT PRIMARY KEY
                          REFERENCES chunks (id) ON DELETE CASCADE,
                embedding vector({dim}) NOT NULL
            )
            """
        )
        # HNSW over cosine distance. Built after the table exists but before
        # bulk load; at this corpus size the build cost is negligible and it
        # keeps the demo path a single command.
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx
            ON {table} USING hnsw (embedding vector_cosine_ops)
            """
        )
    return table


def corpus_stats() -> dict:
    """Row counts used by the CLI and the eval report header."""
    with connect() as conn:
        docs = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
        chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        out = {"documents": docs, "chunks": chunks, "embeddings": {}}
        for key in config.MODELS:
            table = config.embedding_table(key)
            row = conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS present", (table,)
            ).fetchone()
            if row["present"]:
                n = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
                out["embeddings"][key] = n
        return out
