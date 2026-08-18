"""Conversation memory, maintained without any LLM call.

The usual approach is to ask the model to summarise the conversation each turn.
That doubles the per-turn cost and latency for something a rolling window and a
term-frequency table do adequately, and it makes memory nondeterministic --
which in turn makes the whole conversational chain untestable.

What this keeps is what follow-up resolution actually needs: the running
subject (so "what about there?" can be rewritten), a short digest of recent
turns for prompt context, and the terms the conversation keeps returning to.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List

MAX_FACTS = 12
RECENT_TURNS = 4
SUMMARY_CHARS = 600

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "is", "are",
    "was", "were", "how", "what", "why", "when", "where", "which", "do",
    "does", "did", "can", "could", "should", "would", "about", "that", "this",
    "these", "those", "with", "from", "by", "at", "as", "it", "its", "be",
    "been", "there", "their", "them", "they", "me", "my", "you", "your", "we",
    "our", "us", "tell", "give", "some", "more", "also", "than", "then",
}


@dataclass
class Memory:
    summary: str = ""
    facts: List[str] = field(default_factory=list)
    subject: str = ""
    turns: int = 0
    _recent: List[str] = field(default_factory=list, repr=False)

    # -- reading ----------------------------------------------------------
    def as_prompt_context(self) -> str:
        """Compact digest handed to interpretation and answering."""
        parts = []
        if self.subject:
            parts.append(f"Current subject: {self.subject}")
        if self.facts:
            parts.append("Recurring terms: " + ", ".join(self.facts[:8]))
        if self.summary:
            parts.append("Recent turns: " + self.summary)
        return "\n".join(parts)

    def as_dict(self) -> dict:
        return {"summary": self.summary, "facts": self.facts,
                "subject": self.subject, "turns": self.turns}

    @classmethod
    def from_row(cls, row: dict) -> "Memory":
        memory = cls(
            summary=row.get("summary") or "",
            facts=list(row.get("facts") or []),
            subject=row.get("subject") or "",
            turns=int(row.get("turns") or 0),
        )
        # Rehydrate the window from the stored summary so a resumed session
        # keeps producing sensible digests rather than starting blank.
        memory._recent = [s for s in (memory.summary or "").split(" | ") if s]
        return memory

    # -- writing ----------------------------------------------------------
    def observe(self, question: str, is_followup: bool, entities: List[str]) -> None:
        """Fold one user turn into the memory state."""
        self.turns += 1

        self._recent.append(" ".join(question.split()))
        self._recent = self._recent[-RECENT_TURNS:]
        self.summary = " | ".join(self._recent)[:SUMMARY_CHARS]

        # A follow-up leans on the existing subject; a fresh question replaces
        # it. Without this, "what about Ghana?" would reset the topic to
        # "Ghana" alone and lose what was being asked about it.
        if not is_followup:
            self.subject = entities[0] if entities else _headline(question)

        self._update_facts(question, entities)

    def _update_facts(self, question: str, entities: List[str]) -> None:
        counts = Counter(self.facts)
        for term in entities:
            counts[term.lower()] += 2  # named entities outrank bare keywords
        for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{3,}", question.lower()):
            if word not in _STOPWORDS:
                counts[word] += 1
        self.facts = [term for term, _ in counts.most_common(MAX_FACTS)]


def _headline(question: str) -> str:
    """Best-effort subject when no entity was identified: the content words."""
    words = [
        w for w in re.findall(r"[a-zA-Z][a-zA-Z'-]+", question.lower())
        if w not in _STOPWORDS
    ]
    return " ".join(words[:4])
