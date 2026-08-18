"""Retrieval evaluation: compare vector, lexical and hybrid on the gold set.

Metrics are document-level. Retrieval returns chunks, so chunks are collapsed
to their parent documents in rank order before scoring -- otherwise a mode that
returns five chunks of one document would look identical to a mode that
returned five distinct relevant documents.

    python -m eval.run_eval              # compare the three modes
    python -m eval.run_eval --sweep      # also sweep chunk overlap
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from typing import Dict, List, Sequence, Set

import yaml

from datahub_rag import chunk as chunk_stage
from datahub_rag import config, embed, retrieve, store

from .metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank

EVAL_DIR = pathlib.Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
K = 10
# Recall is reported at several depths: at k=10 on a corpus this size every
# mode saturates, and the differences between them only show up near the top.
RECALL_KS = (1, 3, 10)
# Chunks fetched per query before collapsing to documents. Deep enough that a
# document with many matching chunks cannot crowd the top-K document list.
CHUNK_DEPTH = K * 5


def load_queries() -> List[dict]:
    data = yaml.safe_load((EVAL_DIR / "queries.yaml").read_text())
    return data["queries"]


def resolve_labels(queries: Sequence[dict]) -> Dict[str, int]:
    """Map "source:source_id" labels to database document ids.

    Fails loudly on an unknown label: a typo silently scoring as "never
    retrieved" would quietly depress every mode's numbers.
    """
    wanted = {ref for q in queries for ref in q["relevant"]}
    mapping: Dict[str, int] = {}
    with store.connect() as conn:
        for ref in sorted(wanted):
            source, _, source_id = ref.partition(":")
            row = conn.execute(
                "SELECT id FROM documents WHERE source = %s AND source_id = %s",
                (source, source_id),
            ).fetchone()
            if row is None:
                raise SystemExit(f"gold label {ref!r} is not in the corpus")
            mapping[ref] = row["id"]
    return mapping


def ranked_documents(query: str, mode: str,
                     weights: tuple[float, float] | None = None
                     ) -> tuple[List[int], float]:
    """Return document ids in rank order, plus latency in milliseconds."""
    started = time.perf_counter()
    hits = retrieve.search(query, mode=mode, top_k=CHUNK_DEPTH, weights=weights)
    elapsed = (time.perf_counter() - started) * 1000

    seen: List[int] = []
    for hit in hits:
        if hit.document_id not in seen:
            seen.append(hit.document_id)
    return seen, elapsed


def evaluate(mode: str, queries: Sequence[dict], labels: Dict[str, int],
             weights: tuple[float, float] | None = None,
             label: str | None = None) -> dict:
    recalls: Dict[int, List[float]] = {k: [] for k in RECALL_KS}
    rrs, ndcgs, latencies = [], [], []
    per_tag: Dict[str, List[float]] = {}
    misses: List[str] = []

    for item in queries:
        relevant: Set[int] = {labels[ref] for ref in item["relevant"]}
        ranked, elapsed = ranked_documents(item["query"], mode, weights)

        for k in RECALL_KS:
            recalls[k].append(recall_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
        ndcgs.append(ndcg_at_k(ranked, relevant, K))
        latencies.append(elapsed)

        top1 = recall_at_k(ranked, relevant, 1)
        for tag in item.get("tags", []):
            per_tag.setdefault(tag, []).append(top1)
        if recall_at_k(ranked, relevant, K) == 0:
            misses.append(item["id"])

    result = {
        "mode": label or mode,
        "queries": len(queries),
        "mrr": round(statistics.fmean(rrs), 4),
        f"ndcg@{K}": round(statistics.fmean(ndcgs), 4),
        "latency_p50_ms": round(percentile(latencies, 50), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
        "misses": misses,
        "recall_by_tag": {
            tag: round(statistics.fmean(vals), 4) for tag, vals in sorted(per_tag.items())
        },
    }
    for k in RECALL_KS:
        result[f"recall@{k}"] = round(statistics.fmean(recalls[k]), 4)
    return result


def markdown_table(rows: Sequence[dict]) -> str:
    recall_cols = " | ".join(f"recall@{k}" for k in RECALL_KS)
    header = (
        f"| mode | {recall_cols} | MRR | nDCG@{K} | p50 ms | p95 ms |\n"
        + "|---" * (len(RECALL_KS) + 5) + "|\n"
    )
    body = ""
    for r in rows:
        recalls = " | ".join(f"{r[f'recall@{k}']:.3f}" for k in RECALL_KS)
        body += (
            f"| {r['mode']} | {recalls} | {r['mrr']:.3f} | "
            f"{r[f'ndcg@{K}']:.3f} | {r['latency_p50_ms']:.0f} | "
            f"{r['latency_p95_ms']:.0f} |\n"
        )
    return header + body


def tag_table(rows: Sequence[dict]) -> str:
    tags = sorted({t for r in rows for t in r["recall_by_tag"]})
    header = "| tag | " + " | ".join(r["mode"] for r in rows) + " |\n"
    header += "|---" * (len(rows) + 1) + "|\n"
    body = ""
    for tag in tags:
        cells = " | ".join(f"{r['recall_by_tag'].get(tag, 0):.3f}" for r in rows)
        body += f"| {tag} | {cells} |\n"
    return header + body


def run_comparison(queries, labels) -> List[dict]:
    rows = []
    for mode in ("vector", "lexical", "hybrid"):
        print(f"  evaluating {mode} ...", flush=True)
        rows.append(evaluate(mode, queries, labels))
    return rows


def run_weight_sweep(queries, labels) -> List[dict]:
    """Sweep the RRF arm weights.

    Equal weighting is the textbook default but assumes both arms are equally
    trustworthy. Sweeping shows how much of hybrid's deficit against pure dense
    retrieval is the fusion weighting rather than the fusion itself.
    """
    rows = []
    for vector_w, lexical_w in ((1, 1), (2, 1), (3, 1), (5, 1)):
        label = f"hybrid {vector_w}:{lexical_w}"
        print(f"  evaluating {label} ...", flush=True)
        rows.append(evaluate("hybrid", queries, labels,
                             weights=(float(vector_w), float(lexical_w)), label=label))
    return rows


def run_sweep(queries, labels, overlaps=(0, 25, 50)) -> List[dict]:
    """Re-chunk and re-embed at each overlap, then evaluate hybrid retrieval.

    Changing overlap changes the chunk set, so the embedding table has to be
    rebuilt for each setting; this is the slow path by design.
    """
    spec = config.get_model()
    table = config.embedding_table(spec["key"])
    rows = []

    for overlap in overlaps:
        print(f"\n  overlap {overlap}%: rechunking ...", flush=True)
        stats = chunk_stage.chunk_corpus(overlap_pct=overlap, rechunk=True)
        with store.connect() as conn:
            conn.execute(f"TRUNCATE {table}")
        print(f"    {stats['chunks']} chunks, embedding ...", flush=True)
        embed.embed_corpus(spec["key"])

        for mode in ("vector", "hybrid"):
            row = evaluate(mode, queries, labels)
            row["overlap_pct"] = overlap
            row["chunks"] = stats["chunks"]
            row["mode"] = f"{mode} @ {overlap}% overlap"
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", action="store_true",
                    help="also sweep chunk overlap (rebuilds the index each time)")
    ap.add_argument("--weights", action="store_true",
                    help="also sweep the RRF arm weights")
    args = ap.parse_args()

    queries = load_queries()
    labels = resolve_labels(queries)
    census = store.corpus_stats()

    print(f"corpus: {census['documents']} documents, {census['chunks']} chunks")
    print(f"gold set: {len(queries)} queries, {len(labels)} labelled documents")
    print(f"model: {config.DEFAULT_MODEL}\n")

    rows = run_comparison(queries, labels)
    output = "## Retrieval comparison\n\n" + markdown_table(rows)
    output += "\n### top-1 accuracy by query type\n\n" + tag_table(rows)

    missed = {r["mode"]: r["misses"] for r in rows if r["misses"]}
    if missed:
        output += "\n### queries with nothing relevant in the top 10\n\n"
        for mode, ids in missed.items():
            output += f"- **{mode}**: {', '.join(ids)}\n"

    if args.weights:
        weight_rows = run_weight_sweep(queries, labels)
        output += "\n## RRF weight sweep (vector:lexical)\n\n" + markdown_table(weight_rows)
        rows += weight_rows

    if args.sweep:
        sweep_rows = run_sweep(queries, labels)
        output += "\n## Chunk overlap sweep\n\n" + markdown_table(sweep_rows)
        rows += sweep_rows

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(json.dumps(rows, indent=2))
    (RESULTS_DIR / "latest.md").write_text(output)

    print("\n" + output)
    print(f"written to {RESULTS_DIR}/latest.md")


if __name__ == "__main__":
    main()
