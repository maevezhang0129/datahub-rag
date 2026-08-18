"""LLM backends.

Two implementations behind one interface. The stub is not a test fixture
bolted on afterwards -- it is a first-class backend, because it is what lets
the whole conversational chain be exercised, demonstrated and unit-tested with
no API key and no network. Everything except the wording of the final answer is
deterministic under it.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Protocol

# Tasks the pipeline asks of a model. The stub branches on these; real backends
# ignore them. Passing the task explicitly beats having the stub guess from the
# prompt text, which would silently break whenever a prompt was reworded.
TASK_INTERPRET = "interpret"
TASK_ANSWER = "answer"


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, *, task: str,
                 temperature: float = 0.0, max_tokens: int = 800) -> str: ...


class StubLLM:
    """Deterministic offline backend.

    Interpretation is done with real (if blunt) rule-based logic rather than a
    canned string, so the stub exercises the same downstream code paths a live
    model would: mode gating, entity handling and citation verification all
    receive plausible, varying input.
    """

    name = "stub"

    _LIST_MARKERS = (
        "list", "which", "what are", "name some", "examples of",
        "types of", "kinds of", "enumerate", "give me",
    )
    _STOPWORDS = {
        "the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "is",
        "are", "was", "were", "how", "what", "why", "when", "where", "which",
        "do", "does", "did", "can", "could", "should", "would", "about",
        "that", "this", "these", "those", "with", "from", "by", "at", "as",
        "it", "its", "be", "been", "there", "their", "them", "they", "me",
        "my", "i", "you", "your", "we", "our", "us", "tell", "give", "some",
    }

    def complete(self, system: str, user: str, *, task: str,
                 temperature: float = 0.0, max_tokens: int = 800) -> str:
        if task == TASK_INTERPRET:
            return self._interpret(user)
        if task == TASK_ANSWER:
            return self._answer(user)
        raise ValueError(f"stub backend has no behaviour for task {task!r}")

    # -- interpretation ---------------------------------------------------
    def _interpret(self, user: str) -> str:
        question = _extract_tagged(user, "question") or user
        lowered = question.lower()

        terms = [
            w for w in re.findall(r"[a-z0-9][a-z0-9'-]+", lowered)
            if w not in self._STOPWORDS and len(w) > 2
        ]
        # Longer words carry more retrieval signal than short common ones.
        ranked = sorted(dict.fromkeys(terms), key=lambda w: (-len(w), w))

        return json.dumps({
            "mode": "LIST" if any(m in lowered for m in self._LIST_MARKERS) else "SYNTHESIS",
            "query_terms": ranked[:8],
            "entities": [w for w in ranked if len(w) > 6][:5],
            "is_followup": _looks_like_followup(question),
            "confidence": 0.5,
        })

    # -- answering --------------------------------------------------------
    def _answer(self, user: str) -> str:
        """Compose an answer from the supplied sources.

        Deliberately extractive: it quotes the first sentence of the top few
        sources and cites them. That keeps it trivially grounded, so citation
        verification has real, correct input to check against -- and a test can
        assert that a *correct* answer passes the verifier, not only that a
        broken one fails.
        """
        sources = _parse_sources(user)
        if not sources:
            return "I don't have enough information in the knowledge base to answer that."

        parts: List[str] = []
        for index, body in sources[:3]:
            sentence = _first_sentence(body)
            if not sentence:
                continue
            # Marker before the terminal punctuation, the form the prompt asks
            # for. The audit normalises either form, but the stub should model
            # the one being requested.
            if sentence[-1] in ".!?":
                parts.append(f"{sentence[:-1]} [{index}]{sentence[-1]}")
            else:
                parts.append(f"{sentence} [{index}].")
        if not parts:
            return "I don't have enough information in the knowledge base to answer that."
        return " ".join(parts)


class OpenAILLM:
    """OpenAI or Azure OpenAI, selected by which environment variables are set."""

    def __init__(self) -> None:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if endpoint:
            from openai import AzureOpenAI

            self._model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
            self._client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=_require("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        else:
            from openai import OpenAI

            self._model = os.getenv("DATAHUB_CHAT_MODEL", "gpt-4o-mini")
            self._client = OpenAI(api_key=_require("OPENAI_API_KEY"))
        self.name = self._model

    def complete(self, system: str, user: str, *, task: str,
                 temperature: float = 0.0, max_tokens: int = 800) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=max_tokens,
            # Interpretation must parse as JSON; answering must not be constrained.
            response_format={"type": "json_object"} if task == TASK_INTERPRET else None,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is required for the hosted chat backend")
    return value


_CACHE: dict[str, LLM] = {}


def get_llm(backend: str | None = None) -> LLM:
    """Return the configured backend.

    Defaults to the stub, so the chat layer works out of the box. Set
    DATAHUB_CHAT_BACKEND=openai (plus credentials) for a real model.
    """
    choice = (backend or os.getenv("DATAHUB_CHAT_BACKEND", "stub")).lower()
    if choice not in _CACHE:
        _CACHE[choice] = StubLLM() if choice == "stub" else OpenAILLM()
    return _CACHE[choice]


# -- shared helpers -------------------------------------------------------

_FOLLOWUP_PATTERNS = (
    r"^(and|but|so|also)\b", r"\b(that|those|it|them|this)\b",
    r"^(what about|how about|tell me more|expand|go on|why|elaborate)",
    r"^(more|others|another)\b",
)


def _looks_like_followup(question: str) -> bool:
    lowered = question.strip().lower()
    # A very short question almost always leans on the previous turn.
    if len(lowered.split()) <= 3:
        return True
    return any(re.search(p, lowered) for p in _FOLLOWUP_PATTERNS)


def _extract_tagged(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_sources(text: str) -> List[tuple[int, str]]:
    """Pull '[n] title\\n body' blocks out of a rendered prompt."""
    out: List[tuple[int, str]] = []
    for match in re.finditer(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)",
                             text, re.DOTALL | re.MULTILINE):
        out.append((int(match.group(1)), match.group(2).strip()))
    return out


def _first_sentence(text: str) -> str:
    """First complete sentence, skipping a leading fragment.

    Chunk boundaries routinely cut mid-sentence, so a chunk often opens with a
    tail like " scarcity." Quoting that back is incoherent, so the first
    candidate that starts like a real sentence is used instead.
    """
    cleaned = " ".join(text.split())
    candidates = re.findall(r"[^.!?]*[.!?]", cleaned)
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) >= 40 and candidate[:1].isupper():
            return candidate[:300]
    return (candidates[0].strip() if candidates else cleaned)[:200]
