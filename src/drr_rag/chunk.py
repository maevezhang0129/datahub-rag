"""Token-based chunking with configurable overlap.

The text is tokenised once and the chunker slices windows out of that single
token list. Encoding per window instead would re-tokenise every overlapped
region, which at 50% overlap means encoding the whole corpus twice.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterator, List

import tiktoken
from bs4 import BeautifulSoup

from . import config, store

# cl100k_base is the tokeniser behind the OpenAI embedding models. Chunking on
# the same vocabulary the embedder uses means a "512 token" chunk really is 512
# tokens to the model, so chunks never get silently truncated mid-window.
_ENCODER = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str
    token_count: int


def clean_text(raw: str) -> str:
    """Strip HTML and normalise whitespace.

    Also removes NUL bytes, which Postgres rejects in text columns and which
    turn up in text extracted from PDFs.
    """
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        raw = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
    raw = raw.replace("\x00", "")
    return " ".join(raw.split())


def window_starts(n_tokens: int, max_tokens: int, overlap_pct: int) -> Iterator[int]:
    """Yield the start offset of each token window.

    overlap_pct is clamped to 0..90; at 100% the window would never advance.
    """
    if n_tokens <= 0:
        return
    overlap_pct = max(0, min(int(overlap_pct), 90))
    step = max_tokens if overlap_pct == 0 else max(1, max_tokens - max_tokens * overlap_pct // 100)

    start = 0
    while True:
        yield start
        # Stop once this window reached the end, rather than letting range()
        # emit a further start that would produce a duplicate tail window.
        if start + max_tokens >= n_tokens:
            return
        start += step


def chunk_text(
    text: str,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    overlap_pct: int = config.DEFAULT_OVERLAP_PCT,
) -> List[Chunk]:
    """Split `text` into overlapping token windows."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    tokens = _ENCODER.encode(cleaned)
    chunks: List[Chunk] = []
    for ordinal, start in enumerate(window_starts(len(tokens), max_tokens, overlap_pct)):
        window = tokens[start : start + max_tokens]
        if not window:
            break
        chunks.append(
            Chunk(
                ordinal=ordinal,
                text=_ENCODER.decode(window),
                token_count=len(window),
            )
        )
    return chunks


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text or ""))


# --------------------------------------------------------------------------
# Pipeline stage
# --------------------------------------------------------------------------

def chunk_corpus(
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    overlap_pct: int = config.DEFAULT_OVERLAP_PCT,
    rechunk: bool = False,
) -> dict:
    """Chunk every document that does not already have chunks at its current hash.

    A document whose chunks carry its current content_hash is skipped, so the
    stage is cheap to re-run and only revised documents are reprocessed.
    """
    stats = {"documents": 0, "skipped": 0, "chunks": 0}

    with store.connect() as conn:
        if rechunk:
            conn.execute("TRUNCATE chunks RESTART IDENTITY CASCADE")

        docs = conn.execute(
            """
            SELECT d.id, d.title, d.body, d.content_hash
            FROM documents d
            WHERE NOT EXISTS (
                SELECT 1 FROM chunks c
                WHERE c.document_id = d.id AND c.source_hash = d.content_hash
            )
            ORDER BY d.id
            """
        ).fetchall()

        total = conn.execute("SELECT count(*) AS n FROM documents").fetchone()["n"]
        stats["skipped"] = total - len(docs)

        for doc in docs:
            # Drop chunks from a previous revision of this document.
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc["id"],))

            pieces = chunk_text(doc["body"], max_tokens, overlap_pct)
            if not pieces:
                continue

            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, ordinal, title, text, token_count, source_hash)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (document_id, source_hash, ordinal) DO NOTHING
                    """,
                    [
                        (doc["id"], c.ordinal, doc["title"], c.text,
                         c.token_count, doc["content_hash"])
                        for c in pieces
                    ],
                )
            stats["documents"] += 1
            stats["chunks"] += len(pieces)

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Chunk documents into token windows.")
    ap.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS)
    ap.add_argument("--overlap", type=int, default=config.DEFAULT_OVERLAP_PCT,
                    help="overlap percentage, 0-90 (default: %(default)s)")
    ap.add_argument("--rechunk", action="store_true",
                    help="discard existing chunks and rebuild from scratch")
    args = ap.parse_args()

    stats = chunk_corpus(args.max_tokens, args.overlap, args.rechunk)
    print(
        f"chunked {stats['documents']} documents "
        f"({stats['skipped']} already current) -> {stats['chunks']} chunks "
        f"[{args.max_tokens} tokens, {args.overlap}% overlap]"
    )


if __name__ == "__main__":
    main()
