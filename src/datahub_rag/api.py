"""FastAPI service exposing the three retrieval modes."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, embed, retrieve, store
from .chat.llm import get_llm
from .chat.session import ChatSession

logger = logging.getLogger("datahub_rag.api")


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
    title="datahub-rag",
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


# --------------------------------------------------------------------------
# Conversational endpoints
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    session_id: Optional[int] = Field(None, description="omit to start a new session")
    mode: Literal["vector", "lexical", "hybrid"] = "hybrid"


class ChatResponse(BaseModel):
    session_id: Optional[int]
    question: str
    answer: str
    refused: bool
    took_ms: float
    interpretation: dict
    guard: dict
    citations: dict
    trace: dict
    sources: List[dict]


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Ask a grounded question.

    The response carries the interpretation, the guard verdict and the citation
    audit alongside the answer, so a caller can see how the answer was produced
    rather than only what it says.
    """
    started = time.perf_counter()
    try:
        session = ChatSession(session_id=request.session_id, mode=request.mode)
        turn = session.ask(request.question)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat failed: {exc}") from exc

    return ChatResponse(
        session_id=session.session_id,
        took_ms=round((time.perf_counter() - started) * 1000, 2),
        **turn.as_dict(),
    )


@app.get("/chat/{session_id}/context/{message_id}")
def chat_context(session_id: int, message_id: int) -> dict:
    """The chunks shown to the model for one past answer.

    Read from `chat_context_chunks`, not by re-running retrieval: the point of
    the audit trail is that it records what was actually shown, which re-running
    would not reproduce after a re-chunk.
    """
    try:
        session = ChatSession(session_id=session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"session_id": session_id, "message_id": message_id,
            "chunks": [dict(row) for row in session.context_for(message_id)]}


@app.get("/chat/{session_id}/history")
def chat_history(session_id: int) -> dict:
    try:
        session = ChatSession(session_id=session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "memory": session.memory.as_dict(),
        "messages": [
            {"role": m["role"], "content": m["content"], "meta": m["meta"],
             "created_at": m["created_at"].isoformat()}
            for m in session.history()
        ],
    }


# --------------------------------------------------------------------------
# Web UI
# --------------------------------------------------------------------------
# Mounted under /ui rather than / so it can never shadow an API route, and
# served straight from the image: the UI is dependency-free static files, which
# keeps `docker compose up` the only thing a reader has to run.
_WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=_WEB_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/")
