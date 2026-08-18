"""FastAPI service exposing the three retrieval modes."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config, embed, retrieve, store

logger = logging.getLogger("drr_rag.api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the embedding model before accepting traffic.

    Without this the first request pays the model load -- about 16 seconds --
    which reads as "this system is slow" rather than "this process is cold".
    Warm requests are ~50ms.
    """
    started = time.perf_counter()
    try:
        embed.get_embedder().embed_query("warmup")
        logger.info("embedder warm in %.1fs", time.perf_counter() - started)
    except Exception as exc:
        # Not fatal: lexical mode needs no embedder, and /health should still
        # come up to report what is wrong.
        logger.warning("embedder warmup failed: %s", exc)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="drr-rag",
    version="0.1.0",
    description=(
        "Retrieval over disaster risk reduction literature. "
        "Three modes: dense vector, lexical full-text, and RRF hybrid."
    ),
)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    title: str
    url: Optional[str] = None
    published_at: Optional[str] = None
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    model: str
    took_ms: float
    count: int
    results: List[SearchHit]


class HealthResponse(BaseModel):
    status: str
    database: bool
    model: str
    corpus: dict = Field(default_factory=dict)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus a corpus census, so a failed seed is visible immediately."""
    try:
        stats = store.corpus_stats()
        database = True
    except Exception:
        stats, database = {}, False
    return HealthResponse(
        status="ok" if database else "degraded",
        database=database,
        model=config.DEFAULT_MODEL,
        corpus=stats,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=2, description="natural language query"),
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid",
    top_k: int = Query(10, ge=1, le=100),
    source: Optional[str] = Query(None, description="e.g. wikipedia, openalex"),
    published_from: Optional[str] = Query(None, description="ISO date, inclusive"),
    published_to: Optional[str] = Query(None, description="ISO date, inclusive"),
) -> SearchResponse:
    started = time.perf_counter()
    try:
        hits = retrieve.search(
            q,
            mode=mode,
            top_k=top_k,
            filters=retrieve.Filters(source, published_from, published_to),
        )
    except Exception as exc:  # surfaced rather than swallowed into an empty list
        raise HTTPException(status_code=500, detail=f"retrieval failed: {exc}") from exc

    return SearchResponse(
        query=q,
        mode=mode,
        model=config.DEFAULT_MODEL,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        count=len(hits),
        results=[SearchHit(**vars(h)) for h in hits],
    )


@app.get("/stats")
def stats() -> dict:
    return store.corpus_stats()
