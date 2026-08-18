"""Grounded answer generation with verified citations.

Asking a model to cite its sources is not the same as it having done so. This
module renders the retrieved chunks as a numbered source list, instructs the
model to cite them, and then **checks the result**: every [n] marker is
validated against the sources that were actually supplied, markers pointing at
sources that do not exist are stripped, and the proportion of sentences
carrying a citation is recorded.

That audit is the point. It turns "the prompt says to cite" into a measurable
property of each answer, and it is what `chat_context_chunks.was_cited` and the
groundedness figure in the eval are computed from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set

from ..retrieve import Result
from .llm import TASK_ANSWER, LLM, get_llm

BASE_RULES = """\
You answer questions using ONLY the numbered sources provided. You are a
research assistant for disaster risk reduction literature.

Rules, in order of precedence:
1. Use only what the sources state. Do not add facts from your own knowledge,
   even if you are confident they are correct.
2. Cite the source for every claim with its bracketed number, like [2]. Put
   the citation at the end of the sentence it supports. A sentence may carry
   more than one, like [1][3].
3. If the sources do not answer the question, say so plainly and stop. Do not
   assemble a partial answer out of loosely related material.
4. Never cite a number that is not in the source list.
5. Do not mention "the sources" or "the context" as objects. Write the answer,
   with citations.
"""

LIST_RULES = """\
The user wants an enumeration. Answer as a list. Give one item per line,
starting with "- ", each with its own citation. Include every distinct item the
sources support, and no items they do not.
"""

SYNTHESIS_RULES = """\
The user wants an explanation. Answer in prose, at most two short paragraphs.
Lead with the direct answer, then the supporting detail.
"""

REFUSAL = "I don't have enough information in the knowledge base to answer that."

# A sentence shorter than this is treated as connective tissue rather than a
# claim, so it is not counted as an uncited assertion.
_MIN_CLAIM_CHARS = 25


@dataclass
class CitationAudit:
    total_sentences: int = 0
    cited_sentences: int = 0
    cited_sources: Set[int] = field(default_factory=set)
    invalid_markers: List[int] = field(default_factory=list)
    uncited_sentences: List[str] = field(default_factory=list)

    @property
    def groundedness(self) -> float:
        """Fraction of substantive sentences carrying at least one citation."""
        if not self.total_sentences:
            return 0.0
        return self.cited_sentences / self.total_sentences

    @property
    def is_clean(self) -> bool:
        return not self.invalid_markers and not self.uncited_sentences

    def as_dict(self) -> dict:
        return {
            "groundedness": round(self.groundedness, 4),
            "total_sentences": self.total_sentences,
            "cited_sentences": self.cited_sentences,
            "cited_sources": sorted(self.cited_sources),
            "invalid_markers": self.invalid_markers,
            "uncited_count": len(self.uncited_sentences),
        }


@dataclass
class GroundedAnswer:
    text: str
    sources: List[Result]
    audit: CitationAudit
    refused: bool = False


def render_sources(results: Sequence[Result], max_chars: int = 1200) -> str:
    """Render chunks as a numbered list. Numbering is 1-based and positional:
    source [n] is results[n-1], which is what the audit relies on."""
    blocks = []
    for index, result in enumerate(results, start=1):
        body = " ".join(result.text.split())[:max_chars]
        blocks.append(f"[{index}] {result.title}\n{body}")
    return "\n\n".join(blocks)


def normalise_citations(text: str) -> str:
    """Move citation markers inside the sentence they belong to.

    Models place markers on either side of the terminal period -- "...weeks.
    [1]" and "...weeks [1]." are both common, and the same model will produce
    both. Left alone, the first form makes the marker look like it opens the
    *next* sentence, so a correctly cited answer audits as entirely uncited.
    Normalising to "...weeks[1]." makes the audit independent of that styling.
    """
    return re.sub(r"([.!?])(\s*)((?:\[\d+\])+)(?=\s|$)", r"\3\1", text)


def split_sentences(text: str) -> List[str]:
    """Split prose into sentences, treating each list bullet as its own unit."""
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("- ", "* ", "• ")):
            out.append(line)
            continue
        # Split on sentence-final punctuation, but not on the period inside a
        # citation marker or a decimal.
        out.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", line)
            if part.strip()
        )
    return out


def audit_citations(text: str, source_count: int) -> tuple[str, CitationAudit]:
    """Validate every citation marker; strip the ones that point nowhere.

    Returns the cleaned text and the audit. Stripping rather than merely
    flagging matters: a marker like [7] against five sources is a fabricated
    reference, and leaving it in the answer would present a citation the reader
    cannot follow.
    """
    audit = CitationAudit()

    invalid = sorted({
        int(n) for n in re.findall(r"\[(\d+)\]", text)
        if not 1 <= int(n) <= source_count
    })
    audit.invalid_markers = invalid

    cleaned = text
    for number in invalid:
        cleaned = cleaned.replace(f"[{number}]", "")
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned).strip()
    cleaned = normalise_citations(cleaned)

    for sentence in split_sentences(cleaned):
        markers = [int(n) for n in re.findall(r"\[(\d+)\]", sentence)]
        stripped = re.sub(r"\[\d+\]", "", sentence).strip()
        if len(stripped) < _MIN_CLAIM_CHARS:
            continue
        audit.total_sentences += 1
        if markers:
            audit.cited_sentences += 1
            audit.cited_sources.update(markers)
        else:
            audit.uncited_sentences.append(sentence)

    return cleaned, audit


def generate(
    question: str,
    results: Sequence[Result],
    *,
    is_list: bool = False,
    guard_notice: str = "",
    memory: str = "",
    llm: LLM | None = None,
) -> GroundedAnswer:
    """Produce a grounded, citation-audited answer."""
    if not results:
        return GroundedAnswer(REFUSAL, [], CitationAudit(), refused=True)

    llm = llm or get_llm()
    system = BASE_RULES + "\n" + (LIST_RULES if is_list else SYNTHESIS_RULES)
    if guard_notice:
        # The guard reaches the model as an instruction, and is also appended
        # verbatim below -- the notice must survive even if the model ignores it.
        system += (
            "\nThis question touches regulated advice. Answer informationally "
            "from the sources only. Do not tell the user what they should do, "
            "and do not assess their personal situation.\n"
        )

    prompt_parts = []
    if memory:
        prompt_parts.append(f"<conversation>{memory}</conversation>")
    prompt_parts.append(f"<question>{question}</question>")
    prompt_parts.append("Sources:\n" + render_sources(results))

    raw = llm.complete(system, "\n\n".join(prompt_parts),
                       task=TASK_ANSWER, temperature=0.0)

    text, audit = audit_citations(raw.strip(), len(results))
    refused = text.strip().lower().startswith("i don't have enough information")

    if guard_notice and not refused:
        text = f"{text}\n\n{guard_notice}"

    return GroundedAnswer(text=text, sources=list(results), audit=audit, refused=refused)
