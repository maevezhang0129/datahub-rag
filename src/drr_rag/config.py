"""Model registry and environment configuration.

Two embedding backends are supported. The default is a local sentence-transformers
model so that the whole project runs with no API key of any kind; the hosted
OpenAI/Azure model is opt-in via environment variables.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

# --------------------------------------------------------------------------
# Embedding model registry
# --------------------------------------------------------------------------
# `dim` must match the model's true output width: it becomes the pgvector
# column width, and a mismatch surfaces as an insert error rather than as
# silently wrong results.

MODELS: Dict[str, Dict[str, Any]] = {
    "bge-small-en-v1.5": {
        "backend": "local",
        "hf_name": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        # bge models are trained with an asymmetric prefix: queries get an
        # instruction, passages do not. Omitting this costs real retrieval
        # quality, so it is part of the model spec rather than call sites.
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "passage_prefix": "",
    },
    "text-embedding-3-small": {
        "backend": "openai",
        "deployment": "text-embedding-3-small",
        "dim": 1536,
        "query_prefix": "",
        "passage_prefix": "",
    },
}

DEFAULT_MODEL = os.getenv("DRR_EMBED_MODEL", "bge-small-en-v1.5")

# Chunking defaults. 512 tokens is a deliberate choice, see docs/adr/.
DEFAULT_MAX_TOKENS = int(os.getenv("DRR_CHUNK_TOKENS", "512"))
DEFAULT_OVERLAP_PCT = int(os.getenv("DRR_CHUNK_OVERLAP_PCT", "25"))

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF paper
# (Cormack et al., 2009) and the de facto default.
RRF_K = int(os.getenv("DRR_RRF_K", "60"))

# Per-arm RRF weights. Textbook RRF weights both arms equally, which assumes
# both are comparably good. Measured on this corpus they are not: equal
# weighting scores BELOW pure dense retrieval (MRR 0.796 vs 0.847), because a
# noisy lexical ranking gets an equal vote. At 3:1 hybrid overtakes dense on
# recall@10 (0.983 vs 0.967) while still contributing lexical precision.
# The number comes from `make eval-weights`, not from taste; set both to 1.0
# to reproduce unweighted RRF. See docs/evaluation.md.
RRF_WEIGHT_VECTOR = float(os.getenv("DRR_RRF_WEIGHT_VECTOR", "3.0"))
RRF_WEIGHT_LEXICAL = float(os.getenv("DRR_RRF_WEIGHT_LEXICAL", "1.0"))

# Candidate depth taken from each retrieval arm before fusion.
HYBRID_CANDIDATES = int(os.getenv("DRR_HYBRID_CANDIDATES", "200"))

# ts_rank_cd floor for the lexical arm. Because the arm ORs its terms (see
# retrieve.py), a chunk matching one term out of eight still passes the
# predicate; this floor drops that tail without touching genuine partial
# matches.
BM25_MIN_RANK = float(os.getenv("DRR_BM25_MIN_RANK", "0.01"))


def get_model(model_key: str | None = None) -> Dict[str, Any]:
    """Return the spec for `model_key`, defaulting to the configured model."""
    key = model_key or DEFAULT_MODEL
    if key not in MODELS:
        raise KeyError(f"unknown embedding model {key!r}; known: {sorted(MODELS)}")
    return {"key": key, **MODELS[key]}


def embedding_table(model_key: str) -> str:
    """Table name holding embeddings for `model_key`.

    One table per model rather than one wide table: pgvector columns are fixed
    width, so models of different dimensionality cannot share a column, and
    per-model tables let a new model be indexed without touching the old one.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", model_key.lower()).strip("_")
    return f"chunk_embeddings_{slug}"


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://drr:drr@localhost:5432/drr_rag",
    )
