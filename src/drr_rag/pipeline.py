"""Run the full pipeline: migrate -> seed -> chunk -> embed.

This is what `docker compose up` executes, so a clean clone reaches a queryable
state in one command.  Every stage is idempotent, so re-running is cheap and
only does the work that is actually outstanding.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import chunk, config, embed, seed, store


def run(model_key: str | None = None,
        max_tokens: int = config.DEFAULT_MAX_TOKENS,
        overlap_pct: int = config.DEFAULT_OVERLAP_PCT) -> None:
    started = time.perf_counter()

    print("[1/4] migrations")
    for name in store.run_migrations():
        print(f"      applied {name}")

    print("[2/4] seeding frozen corpus")
    stats = seed.seed()
    print(f"      {stats['inserted']} new, {stats['updated']} revised, "
          f"{stats['unchanged']} unchanged")

    print(f"[3/4] chunking ({max_tokens} tokens, {overlap_pct}% overlap)")
    stats = chunk.chunk_corpus(max_tokens, overlap_pct)
    print(f"      {stats['documents']} documents -> {stats['chunks']} chunks "
          f"({stats['skipped']} already current)")

    spec = config.get_model(model_key)
    print(f"[4/4] embedding with {spec['key']} ({spec['dim']}d, {spec['backend']})")
    result = embed.embed_corpus(spec["key"])
    print(f"      {result['embedded']} chunks embedded into {result['table']}")

    census = store.corpus_stats()
    print(f"\nready in {time.perf_counter() - started:.1f}s: "
          f"{census['documents']} documents, {census['chunks']} chunks, "
          f"{census['embeddings']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-tokens", type=int, default=config.DEFAULT_MAX_TOKENS)
    ap.add_argument("--overlap", type=int, default=config.DEFAULT_OVERLAP_PCT)
    args = ap.parse_args()
    try:
        run(args.model, args.max_tokens, args.overlap)
    except Exception as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
