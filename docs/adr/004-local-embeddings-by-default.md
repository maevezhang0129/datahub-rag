# ADR 004: The default embedding model runs locally

## Status
Accepted.

## Context
The obvious default is a hosted embedding API — better quality per dimension
and no local compute. But a reviewer cloning this repository does not have my
API keys, and a portfolio project nobody can run does not do its job.

## Decision
`BAAI/bge-small-en-v1.5` (384d, sentence-transformers, CPU) is the default, and
is baked into the Docker image at build time so first run needs no download.
`text-embedding-3-small` (1536d) is available by setting `DRR_EMBED_MODEL` plus
credentials.

The backends sit behind one `Embedder` interface, so the choice is
configuration rather than a code path — and having two real implementations is
what keeps the interface honest.

The bge models are trained asymmetrically: queries take an instruction prefix,
passages do not. That prefix lives in the model spec in `config.py` rather than
at the call sites, since omitting it costs real retrieval quality and is easy
to forget in one of the two places it is needed.

## Consequences
Quality is lower than a hosted model would give. That is an acceptable trade
because the numbers in `evaluation.md` are *comparative* — the same model is
used for every mode, so the vector/lexical/hybrid comparison is unaffected by
the choice.

Per-model embedding tables mean both models can be indexed at once and compared
directly, rather than one replacing the other.
