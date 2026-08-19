"""Aggregate views behind the dashboard.

These need the seeded database; they skip rather than fail without it.
"""

import pytest

from datahub_rag import config, stats, store
from datahub_rag.chat.llm import StubLLM
from datahub_rag.chat.session import ChatSession


@pytest.fixture(scope="module")
def traffic(seeded):
    """Two persisted turns, one answerable and one not.

    Persisted deliberately: calibration reads the audit trail, so the test has
    to produce one rather than assert against whatever the database happens to
    be carrying.
    """
    ChatSession(llm=StubLLM(), persist=True).ask(
        "what causes flash droughts to develop quickly")
    ChatSession(llm=StubLLM(), persist=True).ask(
        "how do i configure a kubernetes ingress controller")


@pytest.fixture(scope="module")
def seeded():
    try:
        census = store.corpus_stats()
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    if not census["chunks"]:
        pytest.skip("corpus not seeded; run `make pipeline`")
    return census


class TestCorpusHealth:
    def test_totals_match_the_census(self, seeded):
        health = stats.corpus_health()
        assert health["totals"]["documents"] == seeded["documents"]
        assert health["totals"]["chunks"] == seeded["chunks"]

    def test_every_source_is_reported(self, seeded):
        sources = {s["source"] for s in stats.corpus_health()["sources"]}
        assert sources == {"openalex", "wikipedia"}

    def test_no_chunk_exceeds_the_token_target(self, seeded):
        """The chunker caps at the target; a chunk above it means the token
        windowing is off by more than a boundary."""
        health = stats.corpus_health()
        for source in health["sources"]:
            assert source["tokens_max"] <= health["target_tokens"]

    def test_histogram_accounts_for_every_chunk(self, seeded):
        sizes = stats.chunk_size_histogram()
        counted = sum(sum(series["counts"]) for series in sizes["series"])
        assert counted == seeded["chunks"]


class TestIndexCoverage:
    def test_default_model_is_fully_embedded(self, seeded):
        row = next(m for m in stats.index_coverage()
                   if m["model"] == config.DEFAULT_MODEL)
        assert row["indexed"]
        assert row["missing"] == 0
        assert row["coverage"] == 1.0

    def test_an_unbuilt_model_is_absent_rather_than_empty(self, seeded):
        """A model with no table has not been indexed; that is a different
        state from one indexed with zero rows, and the dashboard must not
        render them alike."""
        rows = [m for m in stats.index_coverage() if not m["indexed"]]
        for row in rows:
            assert row["embedded"] == 0
            assert row["missing"] == row["chunks"]

    def test_a_freshly_built_index_is_not_stale(self, seeded):
        assert stats.staleness()["clean"], stats.staleness()


class TestCalibration:
    """The dashboard's reason to exist: showing whether MIN_RELEVANCE still
    sits in the gap ADR 005 measured."""

    def test_the_two_groups_land_on_the_right_side_of_the_threshold(self, traffic):
        c = stats.relevance_calibration()
        assert c["answered"]["n"] and c["refused"]["n"]
        assert c["answered"]["min"] >= c["threshold"]
        assert c["refused"]["max"] < c["threshold"]

    def test_the_gap_is_still_positive(self, traffic):
        """A non-positive margin means no single threshold divides answerable
        from unanswerable questions, and MIN_RELEVANCE needs recalibrating
        rather than nudging."""
        assert stats.relevance_calibration()["margin"] > 0

    def test_histogram_bins_account_for_every_turn(self, traffic):
        c = stats.relevance_calibration()
        binned = sum(c["answered"]["counts"]) + sum(c["refused"]["counts"])
        assert binned == c["total_turns"] == c["answered"]["n"] + c["refused"]["n"]

    def test_margin_is_undefined_without_both_groups(self):
        assert stats._margin([0.9], []) is None
        assert stats._margin([], [0.4]) is None
        assert stats._margin([0.9], [0.4]) == 0.5

    def test_quality_counts_refusals(self, traffic):
        quality = stats.answer_quality()
        assert quality["turns"] >= 2
        assert quality["refusals"] >= 1
        assert quality["answers_with_fabricated_citations"] == 0
