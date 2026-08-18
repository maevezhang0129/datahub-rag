# Evaluation

All numbers below are reproducible on a clean clone with `make up && make eval-weights`.
No API keys are involved: the embedding model runs locally and the corpus is
committed to the repository.

- **Corpus**: 473 documents / 1,213 chunks (128 Wikipedia articles, 345 OpenAlex abstracts)
- **Gold set**: 60 hand-authored queries, 65 labelled documents
- **Embedding model**: `bge-small-en-v1.5` (384d, local, CPU)
- **Metrics**: document-level; chunks are collapsed to parent documents in rank
  order before scoring

## Method

Queries were written by hand after reading each target document, and
deliberately **paraphrase** rather than reuse title words — otherwise both arms
would trivially match the title and the comparison would measure nothing. Each
query carries `tags` recording what it is meant to probe, which is what makes
the per-type breakdown below possible.

### Stated limitations

- **Labels are sparse.** A query may have unlabelled relevant documents in the
  corpus, so absolute recall is a lower bound. Comparison between modes stays
  valid because every mode is scored against the same labels.
- **Targets are Wikipedia documents**, which are longer and more definitional
  than the OpenAlex abstracts making up most of the corpus. The abstracts act
  mainly as distractors.
- **60 queries is a small sample.** Differences below roughly 0.03 are inside
  the noise and should not be read as real. This is why the overlap conclusion
  below is stated as "no difference" rather than as a winner.
- Queries and labels were written by the same person who wrote the retriever,
  which is a real bias. The mitigation is that the gold set was fixed before
  the weight and overlap sweeps were run.

## Mode comparison

| mode | recall@1 | recall@3 | recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| vector | 0.717 | 0.917 | 0.967 | 0.847 | 0.874 | 36 | 53 |
| lexical | 0.475 | 0.642 | 0.908 | 0.607 | 0.676 | 17 | 23 |
| hybrid (3:1) | 0.708 | 0.933 | **0.983** | 0.838 | **0.875** | 46 | 69 |

Hybrid retrieves the most relevant documents overall (recall@10 0.983) and
ranks best by nDCG, at roughly 1.3x the latency of the dense arm alone. Pure
dense retrieval is marginally better at recall@1. The lexical arm is clearly
the weakest alone, but it is not redundant — see the weight sweep.

### Top-1 accuracy by query type

| tag | vector | lexical | hybrid |
|---|---|---|---|
| definition | 0.423 | 0.154 | **0.462** |
| entity | 0.833 | 0.444 | 0.833 |
| historical | 0.833 | 0.500 | 0.833 |
| lexical-favoured | **1.000** | 0.778 | 0.778 |
| mechanism | 0.542 | 0.500 | **0.625** |
| paraphrase | 0.600 | 0.283 | **0.617** |
| procedure | 0.909 | 0.636 | 0.909 |
| rare-term | **1.000** | 0.818 | 0.818 |
| recency | 1.000 | 1.000 | 1.000 |
| regional | 0.857 | 0.571 | 0.857 |

The interesting row is `lexical-favoured` — queries built from rare technical
strings (`QuakeML`, `Keetch-Byram index`, `radius of outermost closed isobar`),
exactly the case hybrid retrieval is supposed to rescue. **Dense retrieval wins
it outright at 1.000, and hybrid is worse at 0.778.**

That is the opposite of the usual story, and the reason is corpus size: with
473 documents, a distinctive term is distinctive enough that the dense model
already nails it, while the lexical arm — which ORs its terms — pulls in
documents matching only the common words and drags them up through fusion.
Hybrid's advantage would be expected to grow with corpus size and with
vocabulary the embedding model has not seen; this corpus is too small and too
mainstream to show it.

## RRF weight sweep

Textbook RRF weights both arms equally. That assumes they are comparably good.

| weighting (vector:lexical) | recall@1 | recall@3 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| 1:1 (textbook) | 0.658 | 0.892 | 0.967 | 0.796 | 0.838 |
| 2:1 | 0.675 | 0.917 | 0.967 | 0.816 | 0.854 |
| **3:1 (default)** | 0.708 | **0.933** | **0.983** | 0.838 | 0.875 |
| 5:1 | **0.725** | 0.917 | **0.983** | **0.844** | **0.879** |
| dense only | 0.717 | 0.917 | 0.967 | 0.847 | 0.874 |

**Unweighted RRF scores below pure dense retrieval** (MRR 0.796 vs 0.847). Equal
weighting gives a measurably weaker lexical ranking an equal vote, and it drags
a good dense ranking down.

At 3:1 hybrid overtakes dense on recall@10 (0.983 vs 0.967) and matches it on
nDCG. 5:1 edges further on MRR but is approaching pure dense retrieval — by
then the lexical arm barely participates, so 3:1 is preferred as the setting
that still gets value from both arms.

This is the finding I would most want to discuss: *hybrid retrieval is not
automatically better than dense retrieval, and the textbook fusion default is
what made it worse here.* Whether the arms deserve equal weight is an empirical
question about the corpus, not a property of RRF.

## Chunk overlap sweep

| overlap | chunks | vector MRR | vector recall@10 | hybrid recall@10 |
|---|---|---|---|---|
| 0% | 1,038 | 0.850 | 0.967 | 0.983 |
| 25% | 1,213 | 0.847 | 0.967 | 0.983 |
| 50% | 1,550 | 0.821 | 0.950 | 0.950 |

0% and 25% are indistinguishable within noise. **50% overlap is a genuine
regression** and costs 49% more chunks to store and embed — plausibly because
near-duplicate chunks of one passage crowd each other through the top ranks,
spending result slots that would otherwise hold distinct documents.

See [ADR 001](adr/001-chunk-size-512-tokens.md) for why 25% is nonetheless kept
as the default.

## Failure analysis

Queries where nothing relevant appeared in the top 10:

| mode | missed |
|---|---|
| vector | q002, q004 |
| lexical | q004, q009, q016, q041, q060 |
| hybrid | q004 |

`q004` — *"a neighbourhood's capacity to absorb a shock and bounce back"*,
targeting **Community resilience** — fails in every mode. The corpus contains
345 OpenAlex abstracts about community resilience that are better lexical and
semantic matches for that phrasing than the general Wikipedia article is. This
is really a labelling artifact: the retrieved documents are relevant, they are
just not the one document labelled relevant. It is the clearest illustration of
the sparse-label limitation stated above, and I left it in rather than quietly
relabelling it.

Fixing it properly means denser labels, which means either multiple annotators
or pooled judgements across systems — the standard approach, and out of scope
at this size.

## Reproducing

```bash
make up            # postgres + pipeline + api
make eval          # mode comparison
make eval-weights  # + RRF weight sweep
make eval-sweep    # + overlap sweep (rebuilds the index three times, ~10 min)
```

Results are written to `eval/results/latest.md` and `latest.json`.
