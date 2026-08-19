# datahub-rag

A grounded question-answering system over disaster risk reduction literature:
token-aware chunking, pgvector dense search, PostgreSQL full-text lexical
search, weighted Reciprocal Rank Fusion, a conversational layer with verified
citations, and an evaluation harness that measures whether any of it actually
helps.

Runs end to end with **one command and no API keys**.

```bash
git clone https://github.com/<you>/datahub-rag && cd datahub-rag
make up          # postgres + pgvector, seed, chunk, embed, serve
open http://localhost:8000/docs
```

---

## Provenance

I worked as part of a team on a production RAG pipeline for the **UNDRR Datum
platform** (UN Office for Disaster Risk Reduction), a system indexing hundreds
of thousands of disaster risk documents into roughly 2.7 million embedded
chunks. My contributions there were the enrichment pipeline monitoring
dashboard, the pipeline documentation set, and code review of the chunking and
embedding stages.

That codebase is private and owned by UNDRR. **No part of it appears here.**
This repository is an independent reimplementation of the architecture, on
public data, built so that I have something I can show in full and discuss in
full technical depth. It was written with AI coding assistance; see
[ATTRIBUTION.md](ATTRIBUTION.md).

---

## Results

Measured on 60 hand-authored queries against the committed 473-document corpus,
reproducible with `make eval-weights`.

| mode | recall@1 | recall@3 | recall@10 | MRR | nDCG@10 | p50 ms |
|---|---|---|---|---|---|---|
| vector (dense) | **0.717** | 0.917 | 0.967 | **0.847** | 0.874 | 36 |
| lexical (full-text) | 0.475 | 0.642 | 0.908 | 0.607 | 0.676 | 17 |
| hybrid RRF 3:1 | 0.708 | **0.933** | **0.983** | 0.838 | **0.875** | 46 |

### Four findings worth the space

**1. Textbook RRF made retrieval worse.** Equal-weight fusion — the standard
default — scored *below* pure dense retrieval (MRR 0.796 vs 0.847), because it
gives a measurably weaker lexical arm an equal vote. Weighting the fusion 3:1
toward the dense arm recovered the loss and pushed recall@10 past dense-only
(0.983 vs 0.967). Hybrid retrieval is not automatically better than dense
retrieval; the fusion weighting decides it.

**2. Chunk overlap bought nothing, and 50% actively hurt.** 0% and 25% overlap
are indistinguishable within noise; 50% is a real regression (MRR 0.821) while
costing 49% more chunks to store and embed. Overlap is a near-universal RAG
default and it is not free.

**3. Dense retrieval beat hybrid on exactly the queries hybrid is meant to
win.** On rare-technical-string queries (`QuakeML`, `Keetch-Byram index`),
dense scored 1.000 top-1 and hybrid 0.778 — because at 473 documents a
distinctive term is already unambiguous to the embedding model, while the
lexical arm drags up documents matching only the common words.

**4. The chat layer refused 0% of unanswerable questions — while being
explicitly prompted to refuse.** Ask it to configure a Kubernetes ingress and
it returned a confidently cited answer, because dense retrieval always returns
its top-k however distant, and the answer stage cannot tell far neighbours from
good context. The prompt instruction was obeyed to the letter; the sources
genuinely looked relevant. A relevance floor calibrated against the measured
score gap (answerable ≥ 0.743, unanswerable ≤ 0.629) took refusal to 100% with
no effect on in-domain answers. **A behaviour you need is not a behaviour you
can prompt for.**

Full methodology, per-query-type breakdowns, stated limitations, and a failure
analysis of the one query every mode misses: **[docs/evaluation.md](docs/evaluation.md)**.

---

## What it does

```
Wikipedia + OpenAlex  ──fetch──▶  frozen JSONL corpus  ──seed──▶  Postgres 16
   (public, no key)               (committed to repo)             + pgvector
                                                                       │
                            ┌──────────────────────────────────────────┤
                            │                                          │
                   chunk (tiktoken 512tok)                    embed (bge-small,
                   → tsvector, title weight A                  local, 384d, HNSW)
                            │                                          │
                            └──────────────┬───────────────────────────┘
                                           │
                    dense (cosine <=>) ────┴──── lexical (ts_rank_cd)
                                           │
                              weighted RRF fusion
                                           │
                                  FastAPI /search
```

- **A web UI that shows its work** — every answer ships with the trace that
  produced it: the guard verdict, the interpretation, the measured relevance
  against the threshold, how many candidates the per-document cap displaced,
  and which retrieved chunks the answer actually cited
- **Grounded chat** with a verified citation audit: every `[n]` marker is
  validated against the sources actually supplied, fabricated ones are
  stripped, and per-answer groundedness is recorded
- **Regulated-advice guardrail** (financial / legal / medical / emergency),
  deterministic and pre-retrieval, so it cannot fail open on a model error
- **Retrieval audit trail** — `chat_context_chunks` stores exactly which chunks
  were shown for each answer and which were cited
- **Multi-source ingest** from two public APIs requiring no credentials, frozen
  into a committed corpus so results are reproducible offline
- **Token-aware chunking** on `cl100k_base` with configurable overlap; text is
  encoded once and windows sliced from that token list
- **Idempotent pipeline** — content hashes let every stage skip work already done
- **Two embedding backends** behind one interface: local by default, hosted
  OpenAI/Azure opt-in
- **Three retrieval modes** sharing one filter path, so hybrid cannot leak
  documents past a filter through one arm
- **Parallel-safe embedding** via `FOR UPDATE ... SKIP LOCKED`

## Design decisions

Each of these is a question I have been asked in interviews, answered with the
measurement rather than the received wisdom:

- [ADR 001 — 512-token chunks, and overlap that did not pay off](docs/adr/001-chunk-size-512-tokens.md)
- [ADR 002 — Weighted RRF, not weighted score fusion](docs/adr/002-rrf-over-score-fusion.md)
- [ADR 003 — Postgres + pgvector rather than a dedicated vector database](docs/adr/003-postgres-not-a-vector-database.md)
- [ADR 004 — Local embeddings by default](docs/adr/004-local-embeddings-by-default.md)
- [ADR 005 — A calibrated relevance gate, not a prompt instruction](docs/adr/005-relevance-gate-before-answering.md)
- [ADR 006 — Conversation memory without an LLM call](docs/adr/006-memory-without-an-llm.md)
- [ADR 007 — Why the UI does not stream tokens](docs/adr/007-no-token-streaming.md)

System overview: [docs/architecture.md](docs/architecture.md).

## Usage

After `make up`, the UI is at **http://localhost:8000/ui/** and the OpenAPI docs
at `/docs`. The left pane is the conversation; the right pane is the pipeline
that produced the selected answer, stage by stage. Clicking a `[n]` marker jumps
to the chunk it cites.

Four buttons on the landing screen demonstrate the four behaviours worth seeing:
a synthesis answer, an enumeration, an out-of-domain refusal (the relevance
meter goes red and the pipeline visibly stops at the gate), and a
regulated-advice question that triggers the guard.

```bash
make chat          # interactive grounded chat (no API key needed)

# One-shot, with the full audit as JSON
docker compose run --rm pipeline \
  python -m datahub_rag.chat.cli --ask "what causes flash droughts" --json

# Retrieval only
docker compose run --rm pipeline \
  python -m datahub_rag.retrieve "how do communities prepare for flooding" --mode hybrid

# HTTP
curl 'localhost:8000/search?q=early+warning+systems&mode=hybrid&top_k=5'
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"question":"what makes wildfires spread faster"}'
curl localhost:8000/health

make test          # 126 cases: unit + integration against the live index
make eval          # reproduce the retrieval results table
make eval-chat     # chat layer: citations, guard, refusal
make eval-weights  # RRF weight sweep
make eval-sweep    # chunk overlap sweep
make clean         # tear down, including the database volume
```

Tuning is via environment variables (see [.env.example](.env.example)):
`DRR_CHUNK_TOKENS`, `DRR_CHUNK_OVERLAP_PCT`, `DRR_RRF_K`,
`DRR_RRF_WEIGHT_VECTOR`, `DRR_RRF_WEIGHT_LEXICAL`, `DRR_BM25_MIN_RANK`,
`DATAHUB_MIN_RELEVANCE`.

The chat layer defaults to a deterministic **stub** backend so it runs with no
API key. Set `DATAHUB_CHAT_BACKEND=openai` plus `OPENAI_API_KEY` (or the
`AZURE_OPENAI_*` trio) for a real model.

To use hosted embeddings instead of the local model, set
`DRR_EMBED_MODEL=text-embedding-3-small` plus `OPENAI_API_KEY` (or the
`AZURE_OPENAI_*` trio). Both models can be indexed simultaneously — embeddings
live in one table per model.

## Layout

```
src/datahub_rag/     chunk · embed · retrieve · api · seed · store · pipeline
  chat/            llm · interpret · guard · answer · memory · session · cli
web/             dependency-free UI, served by the API from the same image
db/migrations/   schema (generated tsvector column, HNSW indexes)
eval/            gold query set, metrics, harness, frozen corpus
docs/            architecture, evaluation, ADRs
scripts/         corpus fetcher
tests/           unit + integration
```

## Limitations

Honest list, since a portfolio project that claims none is not credible:

- **473 documents is small.** Several conclusions above are corpus-size
  artifacts and would likely reverse at scale — that is stated where it applies
  rather than glossed over.
- **60 queries, single annotator.** Differences under ~0.03 are noise. Labels
  are sparse, so absolute recall is a lower bound.
- **No asset enrichment.** The original system extracted text from PDFs, DOCX
  and video transcripts. Here every document arrives as clean text, so that
  stage has no analogue.
- **Groundedness is not yet a claim about a real model.** Under the default
  stub backend the answer text is extractive by construction, so groundedness
  is trivially 1.000 — that measures the audit machinery, not model honesty.
  Run with a hosted backend to make it meaningful.
- **The relevance threshold is calibrated on the same 16 questions it is
  reported against.** The gap is wide and clean, but this is not a held-out
  evaluation.
- **The UI does not stream.** Answers appear complete rather than
  progressively, because the citation audit runs over the finished text —
  see [ADR 007](docs/adr/007-no-token-streaming.md). With a hosted model that
  wait would be seconds, not milliseconds.
- **No asset enrichment UI.** The pipeline dashboard from the original system
  has no analogue here yet.

## License

MIT. Corpus documents remain the property of their publishers: Wikipedia
content is CC BY-SA 4.0, OpenAlex metadata is CC0. Not affiliated with or
endorsed by UNDRR, OCHA, or the United Nations.
