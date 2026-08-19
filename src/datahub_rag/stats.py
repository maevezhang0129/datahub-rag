"""Aggregate views over the corpus, the index, and past turns.

These answer operational questions that the per-answer trace cannot: is every
chunk actually embedded, did a document get revised without being re-chunked,
and -- the one that matters most -- is `MIN_RELEVANCE` still separating
answerable questions from unanswerable ones on this corpus?

Everything here is read-only and computed on demand. At this corpus size the
scans are trivial; none of it is cached, so nothing can go stale.
"""

from __future__ import annotations

from typing import List, Optional

from . import config, store


def corpus_health() -> dict:
    """Per-source census with chunk-size statistics.

    Split by source because the two halves of this corpus behave differently:
    OpenAlex records are abstracts that rarely fill one chunk, Wikipedia
    articles are long-form and split into many. An average over both hides
    that, and it is the kind of asymmetry that changes how the lexical and
    dense arms behave.
    """
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT d.source,
                   count(DISTINCT d.id)                         AS documents,
                   count(c.id)                                  AS chunks,
                   coalesce(round(avg(c.token_count))::int, 0)   AS tokens_mean,
                   coalesce(percentile_disc(0.5) WITHIN GROUP (
                       ORDER BY c.token_count)::int, 0)          AS tokens_median,
                   coalesce(min(c.token_count), 0)               AS tokens_min,
                   coalesce(max(c.token_count), 0)               AS tokens_max,
                   coalesce(sum(c.token_count), 0)               AS tokens_total
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.source
            ORDER BY d.source
            """
        ).fetchall()

    sources = []
    for row in rows:
        record = dict(row)
        record["chunks_per_document"] = (
            round(record["chunks"] / record["documents"], 2)
            if record["documents"] else 0.0
        )
        sources.append(record)

    return {
        "sources": sources,
        "totals": {
            "documents": sum(s["documents"] for s in sources),
            "chunks": sum(s["chunks"] for s in sources),
            "tokens": sum(s["tokens_total"] for s in sources),
        },
        "target_tokens": config.DEFAULT_MAX_TOKENS,
    }


def chunk_size_histogram(width: int = 64) -> dict:
    """Distribution of chunk token counts, bucketed, per source."""
    with store.connect() as conn:
        rows = conn.execute(
            """
            SELECT d.source,
                   (c.token_count / %s) * %s AS bucket,
                   count(*)                  AS n
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            GROUP BY d.source, bucket
            ORDER BY bucket
            """,
            (width, width),
        ).fetchall()

    by_source: dict[str, dict[int, int]] = {}
    for row in rows:
        by_source.setdefault(row["source"], {})[row["bucket"]] = row["n"]

    buckets = sorted({row["bucket"] for row in rows})
    return {
        "width": width,
        "buckets": buckets,
        "series": [
            {"source": source, "counts": [counts.get(b, 0) for b in buckets]}
            for source, counts in sorted(by_source.items())
        ],
    }


def index_coverage() -> List[dict]:
    """Embedding coverage per registered model.

    A model with a missing count above zero means `make pipeline` has work
    outstanding -- which is the normal state after adding documents, and a bug
    at any other time.
    """
    out = []
    with store.connect() as conn:
        total = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
        for key, spec in config.MODELS.items():
            table = config.embedding_table(key)
            present = conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS present", (table,)
            ).fetchone()["present"]
            embedded = 0
            if present:
                embedded = conn.execute(
                    f"SELECT count(*) AS n FROM {table}"
                ).fetchone()["n"]
            out.append({
                "model": key,
                "dimensions": spec["dim"],
                "table": table,
                "indexed": bool(present),
                "embedded": embedded,
                "chunks": total,
                "missing": max(total - embedded, 0),
                "coverage": round(embedded / total, 4) if total else 0.0,
                "is_default": key == config.DEFAULT_MODEL,
            })
    return out


def staleness() -> dict:
    """Documents whose chunks no longer match their current content.

    `chunks.source_hash` records the document's `content_hash` at the moment it
    was cut, so a mismatch means the document was revised and re-chunking has
    not caught up. This is the check that makes the idempotent pipeline
    verifiable rather than merely claimed.
    """
    with store.connect() as conn:
        never = conn.execute(
            """
            SELECT count(*) AS n FROM documents d
            WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
            """
        ).fetchone()["n"]
        stale = conn.execute(
            """
            SELECT count(*) AS n FROM documents d
            WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.document_id = d.id)
              AND NOT EXISTS (
                  SELECT 1 FROM chunks c
                  WHERE c.document_id = d.id AND c.source_hash = d.content_hash
              )
            """
        ).fetchone()["n"]
        orphan = conn.execute(
            """
            SELECT count(*) AS n FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.source_hash <> d.content_hash
            """
        ).fetchone()["n"]
    return {
        "documents_never_chunked": never,
        "documents_stale": stale,
        "chunks_from_a_superseded_revision": orphan,
        "clean": not (never or stale or orphan),
    }


def _turn_traces() -> List[dict]:
    """Every persisted assistant turn that carries a trace."""
    with store.connect() as conn:
        return conn.execute(
            """
            SELECT meta, created_at FROM chat_messages
            WHERE role = 'assistant' AND meta ? 'trace'
            ORDER BY created_at
            """
        ).fetchall()


def relevance_calibration(bins: int = 20) -> dict:
    """Distribution of measured relevance against the gate threshold.

    ADR 005 sets `MIN_RELEVANCE` from the gap between answerable and
    unanswerable questions on this corpus. That calibration is only valid while
    the gap stays where it was, and the corpus or the embedding model changing
    can move it. This is the view that shows whether it has.

    Reads persisted turns only. With no traffic it reports empty rather than a
    plausible-looking zero -- an empty calibration and a broken one must not
    look alike.
    """
    threshold = config.MIN_RELEVANCE
    answered: List[float] = []
    refused: List[float] = []

    for row in _turn_traces():
        trace = (row["meta"] or {}).get("trace") or {}
        score = trace.get("relevance")
        if score is None:
            continue
        (answered if trace.get("gate_passed") else refused).append(float(score))

    edges = [i / bins for i in range(bins + 1)]

    def histogram(values: List[float]) -> List[int]:
        counts = [0] * bins
        for value in values:
            index = min(int(value * bins), bins - 1)
            counts[max(index, 0)] += 1
        return counts

    return {
        "threshold": threshold,
        "bins": bins,
        "edges": edges,
        "answered": {"n": len(answered), "counts": histogram(answered),
                     "min": min(answered, default=None),
                     "max": max(answered, default=None)},
        "refused": {"n": len(refused), "counts": histogram(refused),
                    "min": min(refused, default=None),
                    "max": max(refused, default=None)},
        # The separation the threshold sits in. Negative means the two groups
        # overlap and no single threshold can divide them -- at which point
        # MIN_RELEVANCE needs recalibrating, not nudging.
        "margin": _margin(answered, refused),
        "total_turns": len(answered) + len(refused),
    }


def _margin(answered: List[float], refused: List[float]) -> Optional[float]:
    if not answered or not refused:
        return None
    return round(min(answered) - max(refused), 4)


def answer_quality() -> dict:
    """Citation-audit summary over persisted turns."""
    groundedness: List[float] = []
    refusals = fabricated = total = 0

    for row in _turn_traces():
        meta = row["meta"] or {}
        total += 1
        refusals += int(bool(meta.get("refused")))
        citations = meta.get("citations") or {}
        if meta.get("refused"):
            continue
        if citations.get("invalid_markers"):
            fabricated += 1
        if citations.get("groundedness") is not None:
            groundedness.append(float(citations["groundedness"]))

    return {
        "turns": total,
        "refusals": refusals,
        "refusal_rate": round(refusals / total, 4) if total else 0.0,
        "answers_with_fabricated_citations": fabricated,
        "mean_groundedness": (
            round(sum(groundedness) / len(groundedness), 4) if groundedness else None
        ),
    }
