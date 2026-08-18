"""The guardrail must be deterministic and must not depend on the model."""

import pytest

from datahub_rag.chat import guard


@pytest.mark.parametrize("question,domain", [
    ("should i buy flood insurance for my house", "financial"),
    ("is this earthquake cover worth the premium", "financial"),
    ("can i sue my landlord over flood damage", "legal"),
    ("am i legally required to evacuate", "legal"),
    ("what dosage should i take after smoke inhalation", "medical"),
    ("should i take antibiotics for this", "medical"),
])
def test_regulated_questions_trigger(question, domain):
    verdict = guard.check(question)
    assert verdict.triggered
    assert domain in verdict.domains


@pytest.mark.parametrize("question", [
    "what is the palmer drought index",
    "how do tropical cyclones intensify",
    "list the main categories of drought",
    "what causes debris flows",
])
def test_informational_questions_pass(question):
    assert not guard.check(question).triggered


def test_emergency_outranks_other_domains():
    """Someone possibly in danger gets routed to emergency services, not to a
    financial adviser, even if both patterns match."""
    verdict = guard.check("we are trapped, should i claim on my insurance right now")
    assert verdict.triggered
    assert "emergency" in verdict.domains
    assert "emergency services" in verdict.referral


def test_notice_is_empty_when_not_triggered():
    assert guard.check("what is a megadrought").notice == ""


def test_notice_names_the_referral():
    verdict = guard.check("should i buy flood insurance")
    assert "adviser" in verdict.notice


def test_case_insensitive():
    assert guard.check("SHOULD I BUY FLOOD INSURANCE").triggered
