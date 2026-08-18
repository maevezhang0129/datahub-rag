# Attribution and Provenance

## What this repository is

`datahub-rag` is an **independent reimplementation** of a retrieval-augmented
generation architecture for disaster risk reduction (DRR) literature, built by
me (Siwei Zhang) on public data from Wikipedia and OpenAlex.

It was developed with AI coding assistance (Claude Code), which is recorded in
the commit history rather than left implicit. The architecture, the evaluation
methodology, and the engineering decisions documented in `docs/adr/` are ones I
own and can defend in detail; the measurements in `docs/evaluation.md` are real
and reproducible by anyone who clones this.

Published under the MIT License.

## Relationship to my work at UNDRR

I worked as part of a team on a production RAG pipeline for the **UNDRR Datum
platform** (UN Office for Disaster Risk Reduction). That system indexed
hundreds of thousands of DRR documents into roughly 2.7 million embedded
chunks and served hybrid retrieval to downstream applications.

**My contributions to that project were:**

- Built the enrichment pipeline monitoring dashboard (Streamlit over
  PostgreSQL) used to track chunking, embedding, and per-collection
  enrichment status
- Authored the pipeline documentation set covering the ingest → enrichment →
  chunking → embedding → indexing stages
- Performed code review on the chunking and embedding stages, including an
  optimization proposal for redundant token encoding in overlapped
  ("shifted") chunking

**What this repository does NOT contain:**

That codebase is private and owned by UNDRR. **None of its source code,
configuration, prompts, internal endpoints, dataset definitions, or
infrastructure details appear anywhere in this repository.** Nothing here was
copied, transcribed, or adapted from it.

What I carried over is understanding, not code. The techniques this project
implements — token-based chunking with configurable overlap, Reciprocal Rank
Fusion over vector and lexical candidate sets, pgvector cosine search,
PostgreSQL full-text search for the lexical arm — are standard, publicly
documented information retrieval methods, not anyone's proprietary work.

## Why I built it

I wanted a portfolio artifact I could publish honestly and discuss in full
technical depth. Rebuilding the architecture alone, on public data, with an
evaluation harness the original never had, seemed a better demonstration of
what I understand than any codebase I could point at but not fully claim.

## Data

Documents in `eval/corpus/` are retrieved from the ReliefWeb API and remain
the property of their original publishers. ReliefWeb is a service of the UN
Office for the Coordination of Humanitarian Affairs (OCHA). This project is
not affiliated with, endorsed by, or representing UNDRR, OCHA, or the
United Nations.
