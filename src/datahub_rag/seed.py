"""Load the frozen JSONL corpus into Postgres.

The corpus lives in the repository rather than being fetched at runtime, so a
clone reproduces exactly the documents the published evaluation numbers were
measured on, with no network access and no API keys.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterator

from . import store

CORPUS_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval" / "corpus"


def read_corpus(directory: pathlib.Path = CORPUS_DIR) -> Iterator[dict]:
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no .jsonl corpus files in {directory}. "
            f"Run: python scripts/fetch_corpus.py"
        )
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def seed(directory: pathlib.Path = CORPUS_DIR, truncate: bool = False) -> dict:
    """Upsert every corpus document. Idempotent: unchanged documents are
    left alone, revised ones get a new content_hash which invalidates their
    chunks on the next chunking run."""
    inserted = updated = unchanged = 0

    with store.connect() as conn:
        if truncate:
            conn.execute("TRUNCATE documents RESTART IDENTITY CASCADE")

        for doc in read_corpus(directory):
            digest = store.content_hash(doc.get("title", ""), doc.get("body", ""))
            row = conn.execute(
                """
                INSERT INTO documents
                    (source, source_id, url, title, body, published_at, content_hash, meta)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source, source_id) DO UPDATE SET
                    url          = EXCLUDED.url,
                    title        = EXCLUDED.title,
                    body         = EXCLUDED.body,
                    published_at = EXCLUDED.published_at,
                    content_hash = EXCLUDED.content_hash,
                    meta         = EXCLUDED.meta
                WHERE documents.content_hash IS DISTINCT FROM EXCLUDED.content_hash
                RETURNING (xmax = 0) AS is_insert
                """,
                (
                    doc["source"], doc["source_id"], doc.get("url"),
                    doc.get("title", ""), doc.get("body", ""),
                    doc.get("published_at") or None, digest,
                    json.dumps(doc.get("meta", {})),
                ),
            ).fetchone()

            if row is None:
                unchanged += 1      # conflict hit, WHERE suppressed the update
            elif row["is_insert"]:
                inserted += 1
            else:
                updated += 1

    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}


def main() -> None:
    ap = argparse.ArgumentParser(description="Load the frozen corpus into Postgres.")
    ap.add_argument("--truncate", action="store_true", help="wipe documents first")
    args = ap.parse_args()

    store.run_migrations()
    stats = seed(truncate=args.truncate)
    print(
        f"seeded corpus: {stats['inserted']} new, {stats['updated']} revised, "
        f"{stats['unchanged']} unchanged"
    )


if __name__ == "__main__":
    main()
