# ADR 002: Weighted Reciprocal Rank Fusion, not weighted score fusion

## Status
Accepted, with a measured weighting.

## Context
Hybrid retrieval must combine a dense arm scoring cosine similarity (roughly
0.6–0.9 on this corpus) with a lexical arm scoring `ts_rank_cd` (unbounded
above, observed 0.01–11). Combining them requires a common footing.

## Decision
Fuse on **rank**, not score:

```
score(d) = w_v / (k + rank_v(d)) + w_l / (k + rank_l(d)),  k = 60
```

Score fusion — normalising both to 0..1 and taking a weighted sum — needs the
normalisation to hold across queries. It does not: `ts_rank_cd` depends on term
frequency and document length, so its range moves per query, and any fixed
min-max mapping silently rescales as the corpus changes. RRF discards
magnitudes entirely, so nothing needs calibrating.

`k = 60` is from the original RRF paper (Cormack et al., 2009). It damps the
influence of the very top ranks enough that one arm's confident-but-wrong first
result cannot dominate the fusion.

## The weighting is measured, not assumed
Textbook RRF sets `w_v = w_l = 1`, which assumes both arms are equally good.
On this corpus they are not, and equal weighting scores **below pure dense
retrieval** (MRR 0.796 vs 0.847). Sweeping the weights showed 3:1 recovers the
loss and overtakes dense retrieval on recall@10. See `evaluation.md`.

The general lesson is the point worth keeping: RRF is only fair when the arms
are comparably strong, and whether they are is an empirical question about your
corpus, not a property of the algorithm.

## Consequences
The weighting is tuned to this corpus and does not transfer. Anyone reusing
this should re-run `make eval-weights` against their own gold set. The tuning
knob is exposed as `DRR_RRF_WEIGHT_VECTOR` / `DRR_RRF_WEIGHT_LEXICAL` rather
than hard-coded, precisely because the right value is local.
