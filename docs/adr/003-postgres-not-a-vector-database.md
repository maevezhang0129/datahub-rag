# ADR 003: Postgres + pgvector rather than a dedicated vector database

## Status
Accepted.

## Context
The obvious alternatives are a purpose-built vector store (Qdrant, Weaviate,
Milvus) or a hosted search service.

## Decision
One Postgres 16 instance with pgvector.

The deciding factor is that hybrid retrieval needs **both** arms and a join
back to document metadata. In Postgres the dense arm (`<=>` over HNSW), the
lexical arm (`ts_rank_cd` over a GIN-indexed `tsvector`), the metadata filters,
and the fusion are one query against one consistent snapshot. Splitting the
arms across two systems means two round trips, two indexes to keep in sync, and
filters that must be reimplemented identically on both sides — the failure mode
being a filter that silently applies to only one arm.

It also means one container in `docker-compose.yml` instead of two, which is
what makes the one-command demo credible.

## Consequences
This holds at this corpus size and would hold well past it; pgvector's HNSW is
competitive into the millions of vectors. The point where it stops holding is
when vector search needs to scale independently of the relational workload, or
when index build time becomes operationally significant — at which point the
dense arm moves out and the fusion moves into the application layer.

Because embeddings live in their own per-model tables keyed by `chunk_id`, that
extraction is a contained change: the fusion query is the only thing that would
have to move.
