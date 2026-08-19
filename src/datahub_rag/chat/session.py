"""Conversation orchestrator and persistence.

One turn:

    guard -> interpret -> rewrite query -> retrieve -> ground -> audit -> remember

The guard runs first, before anything is retrieved, so a regulated-advice
question is constrained no matter what retrieval returns. Memory is updated
last, from the interpretation, so a failed turn does not corrupt the subject.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from .. import config, retrieve, store
from ..retrieve import Result
from . import answer as answer_stage
from . import guard, interpret
from .llm import LLM, get_llm
from .memory import Memory

DEFAULT_TOP_K = 8
# A LIST answer needs to see more of the corpus than a synthesis: completeness
# is the whole point of an enumeration.
LIST_TOP_K = 16
# Chunks from one document that may occupy the context. Without a cap, a long
# article whose every section matches will fill all eight slots and the answer
# is grounded in a single source -- which reads as corroboration but is not.
MAX_CHUNKS_PER_DOCUMENT = 2


def diversify(results: List[Result], limit: int,
              max_per_document: int = MAX_CHUNKS_PER_DOCUMENT) -> List[Result]:
    """Cap how many chunks any one document contributes, preserving rank order.

    The cap is hard. An earlier version backfilled past it when the capped set
    came up short, which defeated the entire purpose: for the case this exists
    to handle -- one long article matching at every section -- the backfill
    restored exactly the single-document context it had just removed.

    Returning a short context is the correct outcome. If only one document is
    relevant, two chunks of it is the honest amount of evidence; padding with
    six near-duplicates makes a single source look like corroboration.
    """
    kept: List[Result] = []
    seen: dict[int, int] = {}
    for result in results:
        if len(kept) >= limit:
            break
        count = seen.get(result.document_id, 0)
        if count < max_per_document:
            kept.append(result)
            seen[result.document_id] = count + 1
    return kept


@dataclass
class Trace:
    """How a turn was produced, stage by stage.

    Every field here is a decision the pipeline already made; the trace only
    records it. It exists because the interesting part of a RAG answer is not
    the prose but the path -- which query was actually searched, how close the
    corpus came to the relevance threshold, how many candidates the
    per-document cap discarded. The web UI renders this; nothing depends on it.
    """

    mode: str = "hybrid"
    query: str = ""
    relevance: float = 0.0
    threshold: float = config.MIN_RELEVANCE
    gate_passed: bool = False
    top_k: int = 0
    candidates: int = 0
    kept: int = 0
    documents: int = 0
    # Chunks a plain top-k would have included that the per-document cap
    # displaced. This is the cap's actual effect. It is NOT `candidates - kept`:
    # that difference is mostly the over-fetch being truncated back to top_k,
    # which the cap had nothing to do with.
    displaced: int = 0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "query": self.query,
            "relevance": round(self.relevance, 4),
            "threshold": round(self.threshold, 4),
            "gate_passed": self.gate_passed,
            "top_k": self.top_k,
            "candidates": self.candidates,
            "kept": self.kept,
            "documents": self.documents,
            "displaced": self.displaced,
        }


@dataclass
class Turn:
    question: str
    answer: str
    sources: List[Result] = field(default_factory=list)
    interpretation: Optional[interpret.Interpretation] = None
    guard_verdict: Optional[guard.GuardVerdict] = None
    audit: Optional[answer_stage.CitationAudit] = None
    trace: Trace = field(default_factory=Trace)
    refused: bool = False
    message_id: Optional[int] = None

    @property
    def cited_ordinals(self) -> set:
        """1-based positions in `sources` that the audited answer cites.

        One definition, used by both the persisted audit trail and the API
        payload -- if these two ever disagreed, the stored `was_cited` column
        would stop matching what the UI shows for the same answer.
        """
        return self.audit.cited_sources if self.audit else set()

    def as_dict(self) -> dict:
        cited = self.cited_ordinals
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "interpretation": self.interpretation.as_dict() if self.interpretation else {},
            "guard": self.guard_verdict.as_dict() if self.guard_verdict else {},
            "citations": self.audit.as_dict() if self.audit else {},
            "trace": self.trace.as_dict(),
            "sources": [
                {
                    "n": i,
                    "chunk_id": s.chunk_id,
                    "document_id": s.document_id,
                    "title": s.title,
                    "url": s.url,
                    "published_at": s.published_at,
                    "score": round(s.score, 4),
                    "cited": i in cited,
                    # Truncated to what the prompt carried, so the panel shows
                    # the evidence the model saw rather than the whole chunk.
                    "text": s.text[:answer_stage.PROMPT_CHARS],
                }
                for i, s in enumerate(self.sources, start=1)
            ],
        }


class ChatSession:
    """A conversation. Persisted if `session_id` is set, in-memory otherwise."""

    def __init__(self, session_id: Optional[int] = None, llm: Optional[LLM] = None,
                 mode: str = "hybrid", persist: bool = True):
        self.llm = llm or get_llm()
        self.mode = mode
        self.persist = persist
        self.session_id = session_id
        self.memory = Memory()

        if session_id is not None:
            self._load()
        elif persist:
            self.session_id = self._create()

    # -- persistence ------------------------------------------------------
    def _create(self) -> int:
        with store.connect() as conn:
            row = conn.execute(
                "INSERT INTO chat_sessions DEFAULT VALUES RETURNING id"
            ).fetchone()
        return row["id"]

    def _load(self) -> None:
        with store.connect() as conn:
            row = conn.execute(
                "SELECT summary, facts, subject, turns FROM chat_sessions WHERE id = %s",
                (self.session_id,),
            ).fetchone()
        if row is None:
            raise LookupError(f"no chat session {self.session_id}")
        self.memory = Memory.from_row(row)

    def _save_memory(self) -> None:
        if not (self.persist and self.session_id):
            return
        with store.connect() as conn:
            conn.execute(
                """UPDATE chat_sessions
                   SET summary = %s, facts = %s, subject = %s, turns = %s
                   WHERE id = %s""",
                (self.memory.summary, json.dumps(self.memory.facts),
                 self.memory.subject, self.memory.turns, self.session_id),
            )

    def _save_turn(self, turn: Turn) -> Optional[int]:
        """Persist the exchange and the exact context the model was shown."""
        if not (self.persist and self.session_id):
            return None
        with store.connect() as conn:
            conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, meta) "
                "VALUES (%s, 'user', %s, %s)",
                (self.session_id, turn.question,
                 json.dumps(turn.interpretation.as_dict() if turn.interpretation else {})),
            )
            meta = {
                "guard": turn.guard_verdict.as_dict() if turn.guard_verdict else {},
                "citations": turn.audit.as_dict() if turn.audit else {},
                "refused": turn.refused,
                "retrieval_mode": self.mode,
                "trace": turn.trace.as_dict(),
            }
            row = conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, meta) "
                "VALUES (%s, 'assistant', %s, %s) RETURNING id",
                (self.session_id, turn.answer, json.dumps(meta)),
            ).fetchone()
            message_id = row["id"]

            cited = turn.cited_ordinals
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chat_context_chunks
                       (message_id, chunk_id, document_id, ordinal, title, url,
                        score, was_cited, text)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    [
                        (message_id, s.chunk_id, s.document_id, i, s.title,
                         s.url, s.score, i in cited, s.text[:2000])
                        for i, s in enumerate(turn.sources, start=1)
                    ],
                )
        return message_id

    # -- the turn ---------------------------------------------------------
    def ask(self, question: str, top_k: Optional[int] = None) -> Turn:
        question = question.strip()
        if not question:
            raise ValueError("question is empty")

        verdict = guard.check(question)
        context = self.memory.as_prompt_context()
        plan = interpret.interpret(question, context=context, llm=self.llm)
        query = interpret.build_query(question, plan, subject=self.memory.subject)

        # Relevance gate. Dense retrieval always returns its top-k however
        # weak the match, so without this the answer stage receives
        # plausible-looking context for questions the corpus cannot address
        # and dutifully writes a cited answer from it. Checked before
        # generation, so an out-of-scope question costs no answer tokens.
        relevance = retrieve.top_similarity(query)
        trace = Trace(mode=self.mode, query=query, relevance=relevance,
                      threshold=config.MIN_RELEVANCE)
        if relevance < config.MIN_RELEVANCE:
            turn = Turn(
                question=question,
                answer=answer_stage.REFUSAL,
                interpretation=plan,
                guard_verdict=verdict,
                audit=answer_stage.CitationAudit(),
                trace=trace,
                refused=True,
            )
            self.memory.observe(question, plan.is_followup, plan.entities)
            turn.message_id = self._save_turn(turn)
            self._save_memory()
            return turn

        limit = top_k or (LIST_TOP_K if plan.is_list else DEFAULT_TOP_K)
        # Over-fetch, then diversify: the cap discards chunks, so asking for
        # exactly `limit` would leave the context short.
        candidates = retrieve.search(query, mode=self.mode, top_k=limit * 4)
        results = diversify(candidates, limit, MAX_CHUNKS_PER_DOCUMENT)

        trace.gate_passed = True
        trace.top_k = limit
        trace.candidates = len(candidates)
        trace.kept = len(results)
        trace.documents = len({r.document_id for r in results})
        trace.displaced = len(
            {r.chunk_id for r in candidates[:limit]} - {r.chunk_id for r in results}
        )

        grounded = answer_stage.generate(
            question, results,
            is_list=plan.is_list,
            guard_notice=verdict.notice,
            memory=context,
            llm=self.llm,
        )

        turn = Turn(
            question=question,
            answer=grounded.text,
            sources=grounded.sources,
            interpretation=plan,
            guard_verdict=verdict,
            audit=grounded.audit,
            trace=trace,
            refused=grounded.refused,
        )

        self.memory.observe(question, plan.is_followup, plan.entities)
        turn.message_id = self._save_turn(turn)
        self._save_memory()
        return turn

    # -- history ----------------------------------------------------------
    def history(self) -> List[dict]:
        if not self.session_id:
            return []
        with store.connect() as conn:
            return conn.execute(
                "SELECT role, content, meta, created_at FROM chat_messages "
                "WHERE session_id = %s ORDER BY created_at, id",
                (self.session_id,),
            ).fetchall()

    def context_for(self, message_id: int) -> List[dict]:
        """The chunks shown to the model for one answer, cited flag included."""
        with store.connect() as conn:
            return conn.execute(
                "SELECT ordinal, title, url, score, was_cited, chunk_id, document_id "
                "FROM chat_context_chunks WHERE message_id = %s ORDER BY ordinal",
                (message_id,),
            ).fetchall()
