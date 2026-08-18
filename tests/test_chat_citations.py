"""Citation auditing is the grounding guarantee, so it is tested hardest."""

import pytest

from datahub_rag.chat.answer import (
    audit_citations, normalise_citations, render_sources, split_sentences,
)


class FakeResult:
    def __init__(self, title, text):
        self.title, self.text = title, text


class TestNormalisation:
    def test_marker_after_period_moves_inside(self):
        assert normalise_citations("Rain falls. [1]") == "Rain falls[1]."

    def test_marker_already_inside_is_untouched(self):
        assert normalise_citations("Rain falls [1].") == "Rain falls [1]."

    def test_multiple_markers_after_period(self):
        assert normalise_citations("Rain falls. [1][2]") == "Rain falls[1][2]."

    def test_does_not_touch_a_marker_mid_sentence(self):
        text = "Rain [1] falls hard."
        assert normalise_citations(text) == text


class TestAudit:
    def test_fully_cited_answer_scores_one(self):
        text = "Droughts reduce crop yields substantially [1]. Soil moisture falls too [2]."
        _, audit = audit_citations(text, source_count=2)
        assert audit.groundedness == 1.0
        assert audit.cited_sources == {1, 2}
        assert audit.is_clean

    def test_uncited_sentence_is_caught(self):
        text = ("Droughts reduce crop yields substantially [1]. "
                "Soil moisture declines sharply across the region.")
        _, audit = audit_citations(text, source_count=2)
        assert audit.groundedness == 0.5
        assert len(audit.uncited_sentences) == 1

    def test_fabricated_marker_is_stripped_and_reported(self):
        # [7] against 2 sources is a reference the reader cannot follow.
        text = "Droughts reduce crop yields substantially [7]."
        cleaned, audit = audit_citations(text, source_count=2)
        assert audit.invalid_markers == [7]
        assert "[7]" not in cleaned
        assert not audit.is_clean

    def test_valid_and_invalid_markers_together(self):
        text = "Yields fall in prolonged dry periods [1][9]."
        cleaned, audit = audit_citations(text, source_count=3)
        assert audit.invalid_markers == [9]
        assert "[1]" in cleaned and "[9]" not in cleaned
        assert audit.cited_sources == {1}

    def test_marker_after_period_still_counts_as_cited(self):
        """Regression: this styling made a correctly cited answer audit as 0.0."""
        text = "Droughts reduce crop yields substantially. [1]"
        _, audit = audit_citations(text, source_count=1)
        assert audit.groundedness == 1.0

    def test_short_connective_text_is_not_counted_as_a_claim(self):
        text = "In short. Droughts reduce crop yields substantially [1]."
        _, audit = audit_citations(text, source_count=1)
        assert audit.total_sentences == 1

    def test_list_bullets_are_audited_individually(self):
        text = "- Meteorological drought is a rainfall deficit [1]\n- Hydrological drought is a streamflow deficit [2]"
        _, audit = audit_citations(text, source_count=2)
        assert audit.total_sentences == 2
        assert audit.cited_sources == {1, 2}

    def test_empty_answer(self):
        _, audit = audit_citations("", source_count=3)
        assert audit.groundedness == 0.0
        assert audit.total_sentences == 0


class TestRenderSources:
    def test_numbering_is_one_based_and_positional(self):
        rendered = render_sources([FakeResult("A", "alpha"), FakeResult("B", "beta")])
        assert rendered.startswith("[1] A")
        assert "[2] B" in rendered

    def test_body_is_truncated(self):
        rendered = render_sources([FakeResult("A", "x" * 5000)], max_chars=100)
        assert len(rendered) < 300


class TestSplitSentences:
    def test_splits_prose(self):
        assert len(split_sentences("One thing here. Another thing there.")) == 2

    def test_bullets_stay_whole(self):
        parts = split_sentences("- first item\n- second item")
        assert parts == ["- first item", "- second item"]

    def test_blank_lines_ignored(self):
        assert split_sentences("\n\nOnly one.\n\n") == ["Only one."]
