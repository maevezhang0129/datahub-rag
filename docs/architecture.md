# Architecture

## Pipeline

```
public APIs                frozen corpus            Postgres 16 + pgvector
┌──────────────┐          ┌──────────────┐         ┌──────────────────────┐
│ Wikipedia    │─┐        │ eval/corpus/ │         │ documents            │
│ (long-form)  │ │ fetch  │  *.jsonl     │  seed   │  content_hash        │
├──────────────┤ ├───────▶│  (committed) │────────▶│                      │
│ OpenAlex     │─┘        └──────────────┘         │ chunks               │
│ (abstracts)  │            scripts/               │  text, token_count   │
└──────────────┘            fetch_corpus.py        │  source_hash         │
                                                   │  fts  (tsvector)     │
                                                   │                      │
                                                   │ chunk_embeddings_*   │
                                                   │  vector(384) + HNSW  │
                                                   └──────────┬───────────┘
                                                              │
                   ┌──────────────────────────────────────────┤
                   │                                          │
            ┌──────▼───────┐  ┌──────────────┐   ┌────────────▼─────────┐
            │ dense arm    │  │ lexical arm  │   │ FastAPI /search      │
            │ cosine <=>   │  │ ts_rank_cd   │   │  mode=vector|lexical │
            │ HNSW ANN     │  │ GIN on fts   │   │       |hybrid        │
            └──────┬───────┘  └──────┬───────┘   └──────────────────────┘
                   │                 │
                   └────────┬────────┘
                            │
                   weighted RRF fusion
                   score = wv/(k+rank_v) + wl/(k+rank_l)
```

The fetch step runs once and its output is committed. Everything downstream of
`eval/corpus/` runs offline with no network and no credentials.

## Stages

| Stage | Module | Idempotency |
|---|---|---|
| seed | `datahub_rag/seed.py` | upsert keyed on `(source, source_id)`; the update is suppressed when `content_hash` is unchanged |
| chunk | `datahub_rag/chunk.py` | a document whose chunks carry its current `content_hash` is skipped |
| embed | `datahub_rag/embed.py` | only chunks absent from the model's embedding table are processed |
| retrieve | `datahub_rag/retrieve.py` | read-only |

Every stage is re-runnable, so `make pipeline` after a corpus refresh does only
the outstanding work. `datahub_rag/pipeline.py` chains all four, and is what
`docker compose up` executes.

## Schema notes

**`chunks.fts` is a STORED generated column**, not a trigger-maintained one, so
the full-text vector cannot drift out of sync with the text. Title is
denormalised onto `chunks` because a generated column cannot reference another
table, and the title needs to be in the vector at weight `A` so that a chunk
from a document whose title matches the query outranks an incidental body
match.

**Embeddings live in one table per model** (`chunk_embeddings_<model_slug>`).
pgvector columns are fixed-width, so a 384-dimension and a 1536-dimension model
cannot share a column; per-model tables also mean a new model can be indexed
alongside the old one and compared without a migration.

**`FOR UPDATE ... SKIP LOCKED`** in the embedding claim query lets several
workers run against one database concurrently: each claims a distinct batch and
none blocks on another's rows.

## Retrieval

The two arms retrieve `DRR_HYBRID_CANDIDATES` (default 200) chunks each, then
fuse. Filters are rendered once as a SQL fragment and applied identically to
both arms, so a filtered hybrid query cannot leak documents through one side.

Chunk-level results are collapsed to documents in rank order before scoring in
the evaluation harness; without that, a mode returning five chunks of one
document would score the same as one returning five distinct relevant
documents.

## Conversational layer

```
question
   │
   ├─▶ guard          regulated-advice check (deterministic, pre-retrieval)
   │
   ├─▶ interpret      one model call -> {mode, query_terms, entities, is_followup}
   │
   ├─▶ build_query    follow-ups merged with the running subject
   │
   ├─▶ relevance gate top-1 dense cosine vs MIN_RELEVANCE ──▶ refuse if below
   │
   ├─▶ retrieve       hybrid, over-fetched then capped per document
   │
   ├─▶ generate       grounded answer, citations required
   │
   ├─▶ audit          validate every [n]; strip fabricated ones; score groundedness
   │
   └─▶ remember       subject + window + term counts, no model call
```

Two stages are deliberately **not** the model's job:

- The **guard** is keyword-driven and runs before retrieval. A guardrail that
  depends on the model behaving is not a guardrail, and one that fails open on
  an API error is worse than none.
- The **citation audit** verifies rather than trusts. Asking a model to cite is
  not the same as it having cited: markers pointing at sources that do not
  exist are stripped, and the proportion of sentences carrying a citation is
  recorded per answer.

`chat_context_chunks` stores exactly which chunks were shown for each answer,
with a `was_cited` flag. Without that, a wrong answer cannot be diagnosed —
there is no way to tell whether retrieval missed the evidence or generation
ignored it.

The `stub` LLM backend is a first-class implementation, not a test double. It
makes interpretation and memory fully deterministic, so the entire
conversational chain runs, demos and unit-tests with no API key.

## Web layer

```
web/  (static, no build step)          served at /ui by the same FastAPI app
┌───────────────────────────┐  ┌────────────────────────────────────────────┐
│ conversation              │  │ inspector — the `trace` for one turn       │
│                           │  │                                            │
│  question                 │  │  1 guard      verdict + domains            │
│  answer with [n] chips ───┼─▶│  2 interpret  mode, terms, rewritten query │
│  guard notice (separate)  │  │  3 gate       cosine ──┬── MIN_RELEVANCE   │
│                           │  │  4 retrieve   over-fetch → kept → docs     │
│  mode · sources · cited   │  │  5 audit      groundedness, stripped [n]   │
│  · latency                │  │  ─────────────────────────────────────────│
│                           │  │  context      each chunk, score, cited?    │
└───────────────────────────┘  └────────────────────────────────────────────┘
```

`Turn.trace` is a record of decisions the pipeline had already made; nothing
depends on it. Two of its fields are worth naming because a plausible-looking
version of each would be wrong:

- **`relevance` vs `threshold`** is the gate rendered as a meter. On a refusal
  the pipeline visibly stops at stage 3, which is the whole of ADR 005 in one
  screen.
- **`displaced`** is how many chunks a plain top-k would have kept that the
  per-document cap pushed out. It is deliberately *not* `candidates - kept`:
  that difference is dominated by the over-fetch being truncated back to
  `top_k`, which the cap had nothing to do with, and reporting it would credit
  the cap with roughly three times its actual effect.

The guard notice travels on the verdict rather than only appended to the answer
text, so the UI can render it as a system element instead of recovering it by
string-matching the answer's tail.

The UI does not stream; see [ADR 007](adr/007-no-token-streaming.md) for why
that follows from the audit being post-hoc rather than being a missing feature.

See [`evaluation.md`](evaluation.md) for measured behaviour and
[`adr/`](adr/) for the reasoning behind the main choices.
