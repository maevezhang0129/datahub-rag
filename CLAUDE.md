# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project constraint (read first)

This repository is an **independent reimplementation**. It exists because the
user's original work — a production RAG pipeline for the UNDRR Datum platform —
is private, unlicensed, org-owned, and mostly authored by colleagues, so it
cannot be published. See `ATTRIBUTION.md`.

**No code, configuration, prompts, internal endpoints, dataset definitions, or
infrastructure details from those upstream repos may enter this one.** Write
from the design, never by transcribing. Before publishing changes, the boundary
check must come back clean except for the deliberate UNDRR references in
`README.md` and `ATTRIBUTION.md`:

```bash
grep -rInE 'unisdr|z3r0101|ryanzkizen|szekerb|bounajma|preventionweb-all|host\.docker\.internal|DATUM_API|undrr\.org|\bdatum\b' . \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=corpus \
  --exclude=CLAUDE.md --exclude=README.md --exclude=ATTRIBUTION.md
```

(The three excluded files legitimately contain these terms: this pattern, and
the provenance statements.)

The repo is public and serves as an interview artifact, so documented claims
must stay true to what the code actually does and what the evaluation actually
measured. When a measurement contradicts a design assumption, change the
document (see ADR 001, which was rewritten after its sweep disproved it) rather
than the framing.

## Commands

Everything runs in Docker. Local Python is 3.9, too old for this code; the
gitignored `.venv` exists only to run `scripts/fetch_corpus.py`.

```bash
make up            # build, start postgres, run pipeline, serve API on :8000
make down          # stop (keeps the db volume);  make clean  also drops it
make pipeline      # re-run migrate -> seed -> chunk -> embed (idempotent)
make test          # full suite (118 cases from 102 test functions)
make chat          # interactive grounded chat CLI
make eval          # retrieval comparison;  eval-weights / eval-sweep / eval-chat
make shell         # psql into the database
```

Single test or file:

```bash
docker compose run --rm --entrypoint pytest pipeline tests/test_chunk.py -v
docker compose run --rm --entrypoint pytest pipeline \
  tests/test_chat_citations.py::TestAudit::test_fabricated_marker_is_stripped_and_reported
```

Ad-hoc code against the live index:

```bash
docker compose run --rm pipeline python -c "from datahub_rag import retrieve; ..."
docker compose run --rm pipeline python -m datahub_rag.chat.cli --ask "..." --json
```

After editing `src/`, rebuild before running: `docker compose build -q pipeline`.
The `api` service needs `docker compose up -d --build api`.

`make eval-sweep` re-chunks and re-embeds three times (~10 min); avoid running
builds concurrently with it, since embedding is CPU-bound and they compete.

## Architecture

Two layers over one Postgres 16 + pgvector instance. `docs/architecture.md` has
the diagrams; the reasoning behind each major choice is in `docs/adr/`.

**Pipeline** (`seed → chunk → embed`, chained by `pipeline.py`, which is what
`docker compose up` runs). Every stage is idempotent through content hashes:
`documents.content_hash` suppresses unchanged upserts, and a document whose
chunks already carry its current hash is skipped by chunking. Re-running does
only outstanding work.

**Retrieval** (`retrieve.py`) has three modes sharing one `Filters.sql()`
fragment, so hybrid applies an identical predicate to both arms — a filter that
leaked on one side would silently pass foreign documents through fusion.

**Chat** (`chat/`) is `guard → interpret → gate → retrieve → ground → audit →
remember`. Two stages are deliberately *not* the model's job: the guardrail is
keyword-driven and runs pre-retrieval so it cannot fail open, and the citation
audit verifies markers rather than trusting the prompt.

Cross-cutting things worth knowing before editing:

- **Embeddings live in one table per model** (`chunk_embeddings_<slug>`, built
  by `store.ensure_embedding_table`). pgvector columns are fixed-width, so
  models of different dimensionality cannot share one.
- **`chunks.fts` is a STORED generated column**, so it cannot drift from the
  text. `title` is denormalised onto `chunks` because generated columns cannot
  reference another table, and title needs weight `A` in the vector.
- **The `stub` LLM backend is a first-class implementation, not a test double.**
  It is what lets the whole chat chain run, demo, and unit-test with no API key.
  Changes to `interpret`/`answer` must keep it working.
- **`chat_context_chunks` is the audit trail** — the exact chunks shown per
  answer, with `was_cited`. It has no FK to `chunks` on purpose: re-chunking
  replaces those rows and the trail must survive it.

## Non-obvious behaviour

Each of these was a real bug found by running the system, and each has a
regression test. Do not "simplify" them away.

- **Query vectors must be `np.float32` ndarrays, not lists.** psycopg adapts a
  plain list to `float8[]`, which has no `<=>` operator against `vector`.
  Inserts get away with it because the column type drives the cast; an operator
  expression has nothing to infer from.
- **The lexical arm ORs its terms** by rewriting `websearch_to_tsquery` output
  (`' & '` → `' | '`). Left as-is it ANDs every term, so a natural-language
  question matches nothing at all.
- **The citation audit normalises marker placement** before splitting sentences.
  Models emit both `...weeks. [1]` and `...weeks [1].`; without normalisation
  the first form makes a correctly cited answer audit as 0.0 groundedness.
- **`diversify()`'s per-document cap is hard, with no backfill.** An earlier
  backfill defeated the cap in exactly the single-document case it existed to
  prevent. A short context is the correct outcome.
- **The relevance gate uses dense cosine, not the hybrid score.** RRF produces
  reciprocal-rank sums with no absolute meaning (~0.05 regardless of match
  quality). `MIN_RELEVANCE` (0.68) is calibrated against the measured gap in
  `eval/chat_queries.yaml` and is specific to this corpus *and* embedding model
  — changing either requires re-running `make eval-chat`.
- **MediaWiki returns a full (non-intro) extract for only one page per request**,
  regardless of `exlimit`. `scripts/fetch_corpus.py` fetches one title at a
  time; batching silently yields one article per batch.

## Configuration

Runs with no `.env` at all: the default embedding model is local and baked into
the image, and the chat layer defaults to the stub backend.

Tuning: `DRR_CHUNK_TOKENS`, `DRR_CHUNK_OVERLAP_PCT`, `DRR_RRF_K`,
`DRR_RRF_WEIGHT_VECTOR`, `DRR_RRF_WEIGHT_LEXICAL`, `DRR_BM25_MIN_RANK`,
`DRR_HYBRID_CANDIDATES`, `DATAHUB_MIN_RELEVANCE`.

Hosted models are opt-in: `DRR_EMBED_MODEL=text-embedding-3-small` for
embeddings, `DATAHUB_CHAT_BACKEND=openai` for chat, each with `OPENAI_API_KEY`
or the `AZURE_OPENAI_*` trio.

## Corpus

`eval/corpus/*.jsonl` is frozen and committed so the demo and the published
evaluation numbers are reproducible offline. Regenerating it with
`make corpus` invalidates the numbers in `README.md` and `docs/evaluation.md`;
re-run the evals and update both if you do.

Sources are Wikipedia (CC BY-SA) and OpenAlex (CC0), both key-free. ReliefWeb
was evaluated and rejected: its v2 API requires a registered `appname`, which
would break the clone-and-run property.
