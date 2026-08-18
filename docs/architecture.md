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

See [`evaluation.md`](evaluation.md) for measured behaviour and
[`adr/`](adr/) for the reasoning behind the main choices.
