# ADR 001: 512-token chunks, and overlap that did not pay off

## Status
Accepted, with the overlap default kept deliberately low after measurement
contradicted the initial assumption.

## Context
Chunk size trades context against precision. Too small and a chunk loses the
context that makes it interpretable; too large and one chunk spans several
topics, so its embedding is an average that matches everything weakly and
nothing strongly.

Overlap is the conventional remedy for a second problem: a fixed grid will
sometimes cut a sentence — or an entity and its definition — in half, and
neither resulting chunk then states the fact cleanly.

## Decision
**512 tokens**, tokenised with `cl100k_base`, at **25% overlap**.

512 sits comfortably inside the input limit of every model in the registry and
matches the granularity at which a passage still states one idea. Tokenising
with `cl100k_base` rather than by characters or words means a "512 token" chunk
is 512 tokens to the model too, so chunks are never silently truncated at the
embedding boundary.

## What the measurement actually showed

I expected overlap to help. It did not. Sweeping 0 / 25 / 50% (`make eval-sweep`):

| overlap | chunks | vector MRR | hybrid recall@10 |
|---|---|---|---|
| 0%  | 1,038 | 0.850 | 0.983 |
| 25% | 1,213 | 0.847 | 0.983 |
| 50% | 1,550 | 0.821 | 0.950 |

0% and 25% are indistinguishable — the gaps are far inside the noise of a
60-query gold set — while **50% overlap is a real regression** on every metric,
and costs 49% more chunks to store and embed.

The likely reason 50% hurts: heavy overlap fills the index with near-duplicate
chunks of the same passage. They crowd each other through the top ranks, so a
single document occupies several result slots that would otherwise hold
distinct documents. Overlap buys boundary robustness and pays for it in result
diversity, and past some point the trade stops being worth it.

25% is kept over 0% as a cheap hedge against boundary effects that this gold
set is too small to detect, at a 17% chunk-count cost. That is a judgement
call, not a measured win, and it is recorded here as such — 0% would be an
equally defensible default on this evidence.

## Consequences
The text is encoded once per document and windows are sliced from that single
token list. Encoding per window would re-tokenise every overlapped region — at
50% overlap, tokenising the corpus twice.

The general lesson worth carrying: overlap is widely recommended as a default
and is not free. On a corpus of self-contained, well-structured documents it
may buy nothing, and at high settings it measurably costs. It is worth sweeping
rather than assuming.
