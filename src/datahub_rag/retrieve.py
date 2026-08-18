"""Retrieval: dense vector, lexical full-text, and Reciprocal Rank Fusion.

Three modes share one filter and result shape so they can be compared directly
by the evaluation harness:

  vector  cosine ANN over pgvector (HNSW)
  lexical Postgres full-text search, ts_rank_cd, title weighted above body
  hybrid  RRF over the two candidate sets (default)

Why RRF rather than a weighted sum of the two scores: cosine similarity and
ts_rank_cd live on different, query-dependent scales, so any fixed weighting
needs per-corpus tuning and still drifts. RRF discards magnitudes and fuses on
rank alone, which needs no calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, List, Optional, Sequence

import numpy as np
from pgvector.psycopg import register_vector

from . import config, embed, store

MODES = ("vector", "lexical", "hybrid")


@dataclass
class Result:
    chunk_id: int
    document_id: int
    title: str
    url: Optional[str]
    published_at: Optional[str]
    score: float
    text: str

    @classmethod
    def from_row(cls, row: dict) -> "Result":
        published = row.get("published_at")
        return cls(
            chunk_id=row["id"],
            document_id=row["document_id"],
            title=row["title"],
            url=row.get("url"),
            published_at=published.isoformat() if isinstance(published, date) else published,
            score=float(row["score"]),
            text=row["text"],
        )


@dataclass
class Filters:
    source: Optional[str] = None
    published_from: Optional[str] = None
    published_to: Optional[str] = None

    def sql(self, alias: str = "d") -> tuple[str, list]:
        """Render as a SQL fragment plus its parameters.

        Returned as a fragment rather than interpolated so both arms of the
        hybrid query apply exactly the same predicate.
        """
        clauses, params = [], []
        if self.source:
            clauses.append(f"{alias}.source = %s")
            params.append(self.source)
        if self.published_from:
            clauses.append(f"{alias}.published_at >= %s")
            params.append(self.published_from)
        if self.published_to:
            clauses.append(f"{alias}.published_at <= %s")
            params.append(self.published_to)
        return ("".join(f" AND {c}" for c in clauses), params)


# --------------------------------------------------------------------------
# Query builders
# --------------------------------------------------------------------------

def _vector_sql(table: str, filters: Filters) -> tuple[str, list]:
    where, params = filters.sql()
    sql = f"""
        SELECT c.id, c.document_id, c.title, c.text,
               d.url, d.published_at,
               1 - (e.embedding <=> %s) AS score
        FROM {table} e
        JOIN chunks c    ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE TRUE{where}
        ORDER BY e.embedding <=> %s
        LIMIT %s
    """
    return sql, params


def _lexical_sql(filters: Filters) -> tuple[str, list]:
    where, params = filters.sql()
    sql = f"""
        WITH q AS (
            -- websearch_to_tsquery ANDs every term, so a natural-language
            -- question ("how do communities prepare for flooding") matches
            -- only documents containing all of its words -- in practice, none.
            -- Let Postgres do the stemming and stopword removal, then relax
            -- the conjunctions to disjunctions so ts_rank_cd ranks partial
            -- matches instead of the predicate discarding them. Phrase
            -- operators (<->) from quoted input are left intact.
            SELECT replace(
                       websearch_to_tsquery('english', %s)::text, ' & ', ' | '
                   )::tsquery AS tsq
        )
        SELECT c.id, c.document_id, c.title, c.text,
               d.url, d.published_at,
               ts_rank_cd(c.fts, q.tsq) AS score
        FROM chunks c
        CROSS JOIN q
        JOIN documents d ON d.id = c.document_id
        WHERE c.fts @@ q.tsq
          AND ts_rank_cd(c.fts, q.tsq) >= %s{where}
        ORDER BY score DESC
        LIMIT %s
    """
    return sql, params


def _hybrid_sql(table: str, filters: Filters) -> tuple[str, list]:
    """Weighted RRF over the two arms."""
    where, fparams = filters.sql()
    sql = f"""
        WITH q AS (
            -- websearch_to_tsquery ANDs every term, so a natural-language
            -- question ("how do communities prepare for flooding") matches
            -- only documents containing all of its words -- in practice, none.
            -- Let Postgres do the stemming and stopword removal, then relax
            -- the conjunctions to disjunctions so ts_rank_cd ranks partial
            -- matches instead of the predicate discarding them. Phrase
            -- operators (<->) from quoted input are left intact.
            SELECT replace(
                       websearch_to_tsquery('english', %s)::text, ' & ', ' | '
                   )::tsquery AS tsq
        ),
        vec AS (
            SELECT e.chunk_id AS id,
                   ROW_NUMBER() OVER (ORDER BY e.embedding <=> %s) AS rank
            FROM {table} e
            JOIN chunks c    ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE TRUE{where}
            ORDER BY e.embedding <=> %s
            LIMIT %s
        ),
        lex AS (
            SELECT c.id,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(c.fts, q.tsq) DESC) AS rank
            FROM chunks c
            CROSS JOIN q
            JOIN documents d ON d.id = c.document_id
            WHERE c.fts @@ q.tsq
              AND ts_rank_cd(c.fts, q.tsq) >= %s{where}
            ORDER BY rank
            LIMIT %s
        ),
        fused AS (
            -- FULL OUTER JOIN: a chunk found by only one arm still scores,
            -- it just collects a single reciprocal-rank term.
            SELECT COALESCE(v.id, l.id) AS id,
                   COALESCE(%s / (%s + v.rank), 0)
                 + COALESCE(%s / (%s + l.rank), 0) AS score
            FROM vec v
            FULL OUTER JOIN lex l ON l.id = v.id
        )
        SELECT c.id, c.document_id, c.title, c.text,
               d.url, d.published_at, f.score
        FROM fused f
        JOIN chunks c    ON c.id = f.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY f.score DESC
        LIMIT %s
    """
    return sql, fparams


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    model_key: str | None = None,
    filters: Filters | None = None,
    candidates: int = config.HYBRID_CANDIDATES,
    weights: tuple[float, float] | None = None,
) -> List[Result]:
    """Run `query` in `mode` and return up to `top_k` ranked chunks.

    `weights` is the (vector, lexical) RRF weighting; it only applies to hybrid
    mode and defaults to the configured values.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    filters = filters or Filters()
    if weights is None:
        weights = (config.RRF_WEIGHT_VECTOR, config.RRF_WEIGHT_LEXICAL)
    spec = config.get_model(model_key)
    table = config.embedding_table(spec["key"])

    # The lexical arm needs no embedding, so skip the model load for it.
    qvec = None
    if mode in ("vector", "hybrid"):
        # As a float32 ndarray, not a list: psycopg adapts a plain list to
        # float8[], which has no <=> operator against vector. Inserts get away
        # with it because the column type drives the cast; an operator
        # expression has nothing to infer from.
        qvec = np.asarray(
            embed.get_embedder(spec["key"]).embed_query(query), dtype=np.float32
        )

    if mode == "vector":
        sql, fparams = _vector_sql(table, filters)
        params = [qvec, *fparams, qvec, top_k]

    elif mode == "lexical":
        sql, fparams = _lexical_sql(filters)
        params = [query, config.BM25_MIN_RANK, *fparams, top_k]

    else:  # hybrid
        sql, fparams = _hybrid_sql(table, filters)
        params = [
            query,                                   # q CTE
            qvec, *fparams, qvec, candidates,        # vec arm
            config.BM25_MIN_RANK, *fparams, candidates,  # lex arm
            weights[0], config.RRF_K,                # vector arm weight + k
            weights[1], config.RRF_K,                # lexical arm weight + k
            top_k,
        ]

    with store.connect() as conn:
        register_vector(conn)
        rows = conn.execute(sql, params).fetchall()

    return [Result.from_row(r) for r in rows]


def top_similarity(query: str, model_key: str | None = None) -> float:
    """Best dense cosine similarity in the corpus for `query`.

    Used as a relevance gate. Deliberately the dense score rather than the
    hybrid one: RRF produces reciprocal-rank sums with no absolute meaning
    (~0.05 regardless of match quality), so a threshold on them is
    uninterpretable. Cosine similarity is comparable across queries.
    """
    hits = search(query, mode="vector", top_k=1, model_key=model_key)
    return hits[0].score if hits else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the DRR knowledge base.")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=MODES, default="hybrid")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--from", dest="published_from", default=None)
    ap.add_argument("--to", dest="published_to", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    results = search(
        args.query,
        mode=args.mode,
        top_k=args.top_k,
        model_key=args.model,
        filters=Filters(args.source, args.published_from, args.published_to),
    )

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    if not results:
        print("no results")
        return
    for i, r in enumerate(results, 1):
        snippet = " ".join(r.text.split())[:180]
        print(f"\n{i}. [{r.score:.4f}] {r.title}")
        print(f"   {r.url or '-'}  ({r.published_at or 'undated'})")
        print(f"   {snippet}...")


if __name__ == "__main__":
    main()
