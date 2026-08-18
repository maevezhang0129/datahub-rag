"""Ranking metrics. Kept separate from the runner so they are unit-testable."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, Set


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Fraction of the relevant set that appears in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    """1/rank of the first relevant item, 0 if none was retrieved."""
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Binary-gain nDCG.

    Labels are binary (relevant / not), so gain is 1 or 0 and the ideal ranking
    puts every relevant document first.
    """
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(index + 1)
        for index in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def percentile(values: Iterable[float], pct: float) -> float:
    """Nearest-rank percentile; avoids a numpy dependency for one number."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
