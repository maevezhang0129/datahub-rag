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
class Turn:
    question: str
    answer: str
    sources: List[Result] = field(default_factory=list)
    interpretation: Optional[interpret.Interpretation] = None
    guard_verdict: Optional[guard.GuardVerdict] = None
    audit: Optional[answer_stage.CitationAudit] = None
    refused: bool = False
    message_id: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "interpretation": self.interpretation.as_dict() if self.interpretation else {},
            "guard": self.guard_verdict.as_dict() if self.guard_verdict else {},
            "citations": self.audit.as_dict() if self.audit else {},
            "sources": [
                {"n": i, "title": s.title, "url": s.url, "score": round(s.score, 4)}
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
            }
            row = conn.execute(
                "INSERT INTO chat_messages (session_id, role, content, meta) "
                "VALUES (%s, 'assistant', %s, %s) RETURNING id",
                (self.session_id, turn.answer, json.dumps(meta)),
            ).fetchone()
            message_id = row["id"]

            cited = turn.audit.cited_sources if turn.audit else set()
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
        if relevance < config.MIN_RELEVANCE:
            turn = Turn(
                question=question,
                answer=answer_stage.REFUSAL,
                interpretation=plan,
                guard_verdict=verdict,
                audit=answer_stage.CitationAudit(),
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
