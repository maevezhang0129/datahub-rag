"""Regulated-advice guardrail.

A knowledge base of disaster-risk literature can be asked questions whose
answers are regulated advice -- what insurance to buy, whether to evacuate a
specific person, how to treat an injury. Retrieval will happily return relevant
passages, and a grounded answer built from them still reads as personal advice.

The guard runs before retrieval and downgrades those turns: the system still
answers informationally from the corpus, but is instructed not to issue a
recommendation, and the response carries a referral notice.

This is deliberately conservative and keyword-driven rather than model-driven:
a guardrail that depends on the model behaving is not a guardrail, and one that
fails open on an API error is worse than none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Domain -> (patterns, referral wording).
_DOMAINS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "medical": (
        (r"\b(should i|do i need to|can i) (take|use|apply|drink|eat)\b",
         r"\b(diagnos|prescrib|dosage|symptom|treat(ment)? for|medication)\b",
         r"\b(am i|is my \w+) (sick|infected|injured|safe to)\b"),
        "a qualified medical professional",
    ),
    "legal": (
        # No trailing \b on these: several alternatives are stems ("prosecut",
        # "require"), and a word boundary after a stem never matches its own
        # inflections -- "legally required" would silently fail to trigger.
        (r"\b(sue|sued|lawsuit|liable|liability|prosecut|negligen)",
         r"\blegal(ly)?\s+(obligat|require|entitled|responsib)",
         r"\b(can|should|must)\s+(i|we)\s+(sue|claim|file|refuse|report)\b",
         r"\b(my|our)\s+(rights|contract|lease|landlord|employer|insurer)\b"),
        "a qualified legal professional",
    ),
    "financial": (
        (r"\b(should i|worth) (buy|sell|invest|insure|purchase)\b",
         r"\b(insurance (policy|claim|payout)|premium|deductible|mortgage|investment)\b",
         r"\b(financial|tax) (advice|planning)\b"),
        "a licensed financial or insurance adviser",
    ),
    "emergency": (
        # Live danger is not an advice question; it is a routing question.
        (r"\b(right now|currently|happening now)\b.{0,40}\b(fire|flood|earthquake|trapped|injured)\b",
         r"\b(i am|we are|my family)\b.{0,30}\b(trapped|injured|drowning|evacuat)\b",
         r"\bshould (i|we) evacuate\b"),
        "your local emergency services",
    ),
}


@dataclass
class GuardVerdict:
    triggered: bool
    domains: List[str]
    referral: str = ""

    @property
    def notice(self) -> str:
        if not self.triggered:
            return ""
        return (
            f"This answer summarises published literature and is not "
            f"{'advice' if 'emergency' not in self.domains else 'emergency guidance'}. "
            f"For your situation, consult {self.referral}."
        )

    def as_dict(self) -> dict:
        # The notice travels with the verdict so a client can render it as a
        # distinct element instead of recovering it by string-matching the tail
        # of the answer it was appended to.
        return {
            "triggered": self.triggered,
            "domains": self.domains,
            "referral": self.referral,
            "notice": self.notice,
        }


def check(question: str) -> GuardVerdict:
    lowered = " " + question.lower().strip() + " "
    hits = [
        domain for domain, (patterns, _) in _DOMAINS.items()
        if any(re.search(p, lowered) for p in patterns)
    ]
    if not hits:
        return GuardVerdict(triggered=False, domains=[])

    # Emergency outranks the rest: if someone may be in danger, the referral
    # that matters is the one to emergency services.
    primary = "emergency" if "emergency" in hits else hits[0]
    return GuardVerdict(triggered=True, domains=hits, referral=_DOMAINS[primary][1])
