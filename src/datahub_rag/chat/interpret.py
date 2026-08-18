"""Concept interpretation and the LIST/SYNTHESIS mode gate.

One model call turns a raw question into a retrieval plan: search terms, the
answer shape, and whether the turn depends on the previous one. Keeping this
separate from answering means the retrieval query can be rewritten (resolving
"what about there?" into an actual subject) before anything is searched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from .llm import TASK_INTERPRET, LLM, get_llm

SYSTEM_PROMPT = """\
You analyse a question and produce a retrieval plan. You never answer it.

Return ONLY a JSON object with these keys:
  mode         "LIST" if the user wants an enumeration of items, otherwise
               "SYNTHESIS" for an explanation or summary.
  query_terms  3-8 concrete search keywords. Content words only: no filler,
               no question words. These are used for keyword search, so prefer
               the terms that would appear in a relevant document.
  entities     specific named things in the question (places, organisations,
               events, named indices). Empty list if there are none.
  is_followup  true if the question depends on the previous turn to be
               understood, e.g. it uses a pronoun with no antecedent, or asks
               to expand on something unnamed.
  confidence   0.0-1.0, your confidence that the question is answerable from a
               corpus of disaster risk reduction literature.

Rules:
- If the question is a follow-up, do NOT invent a new topic. Reuse the subject
  from the conversation context you are given.
- Be literal. Do not expand acronyms you are not sure about.
"""


@dataclass
class Interpretation:
    mode: str = "SYNTHESIS"
    query_terms: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    is_followup: bool = False
    confidence: float = 0.5
    raw: str = ""

    @property
    def is_list(self) -> bool:
        return self.mode == "LIST"

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "query_terms": self.query_terms,
            "entities": self.entities, "is_followup": self.is_followup,
            "confidence": self.confidence,
        }


def interpret(question: str, context: str = "", llm: LLM | None = None) -> Interpretation:
    llm = llm or get_llm()
    user = f"<context>{context}</context>\n<question>{question}</question>"
    raw = llm.complete(SYSTEM_PROMPT, user, task=TASK_INTERPRET, temperature=0.0)

    try:
        data = json.loads(_strip_fence(raw))
    except (json.JSONDecodeError, TypeError):
        # A malformed plan must not take the turn down: fall back to using the
        # question itself as the query. Degraded retrieval beats no answer.
        return Interpretation(query_terms=question.split()[:8], raw=raw)

    mode = str(data.get("mode", "SYNTHESIS")).upper()
    return Interpretation(
        mode=mode if mode in ("LIST", "SYNTHESIS") else "SYNTHESIS",
        query_terms=[str(t) for t in data.get("query_terms", []) if str(t).strip()],
        entities=[str(e) for e in data.get("entities", []) if str(e).strip()],
        is_followup=bool(data.get("is_followup", False)),
        confidence=_clamp(data.get("confidence", 0.5)),
        raw=raw,
    )


def build_query(question: str, interp: Interpretation, subject: str = "") -> str:
    """Turn the plan into the string actually sent to retrieval.

    A follow-up is merged with the running subject, because "what about there?"
    retrieves nothing on its own. The original wording is kept alongside the
    terms so the dense arm still sees natural language.
    """
    parts = [question]
    if interp.is_followup and subject and subject.lower() not in question.lower():
        parts.insert(0, subject)
    extra = [t for t in interp.query_terms if t.lower() not in question.lower()]
    parts.extend(extra)
    return " ".join(parts).strip()


def _strip_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
    return cleaned.strip()


def _clamp(value, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return 0.5
