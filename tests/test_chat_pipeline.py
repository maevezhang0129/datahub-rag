"""End-to-end conversational tests using the deterministic stub backend.

These need the seeded database; they skip rather than fail without it.
"""

import pytest

from datahub_rag import store
from datahub_rag.chat import interpret
from datahub_rag.chat.llm import StubLLM
from datahub_rag.chat.session import ChatSession, diversify


class Doc:
    def __init__(self, document_id, chunk_id=0):
        self.document_id, self.chunk_id = document_id, chunk_id


@pytest.fixture(scope="module")
def seeded():
    try:
        census = store.corpus_stats()
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    if not census["chunks"]:
        pytest.skip("corpus not seeded; run `make pipeline`")
    return census


class TestDiversify:
    def test_caps_chunks_per_document(self):
        results = [Doc(1, i) for i in range(10)]
        assert len(diversify(results, limit=8, max_per_document=2)) == 2

    def test_preserves_rank_order(self):
        results = [Doc(1, 0), Doc(2, 1), Doc(1, 2), Doc(3, 3)]
        kept = diversify(results, limit=4, max_per_document=1)
        assert [d.document_id for d in kept] == [1, 2, 3]

    def test_cap_holds_even_when_it_shortens_the_context(self):
        """Returning fewer chunks is correct: padding a single-document result
        up to the limit would recreate the false corroboration the cap
        exists to prevent."""
        results = [Doc(1, i) for i in range(6)]
        assert len(diversify(results, limit=4, max_per_document=2)) == 2

    def test_respects_the_limit(self):
        results = [Doc(i, i) for i in range(20)]
        assert len(diversify(results, limit=5, max_per_document=2)) == 5


class TestInterpretation:
    def test_list_question_is_gated_to_list_mode(self):
        plan = interpret.interpret("what are the main types of drought", llm=StubLLM())
        assert plan.is_list

    def test_explanatory_question_is_synthesis(self):
        plan = interpret.interpret("how do cyclones intensify", llm=StubLLM())
        assert not plan.is_list

    def test_short_question_reads_as_a_followup(self):
        assert interpret.interpret("tell me more", llm=StubLLM()).is_followup

    def test_query_terms_drop_filler_words(self):
        plan = interpret.interpret("what are the causes of drought", llm=StubLLM())
        assert "the" not in plan.query_terms
        assert any("drought" in t for t in plan.query_terms)

    def test_malformed_json_degrades_instead_of_raising(self):
        class Broken:
            name = "broken"
            def complete(self, *a, **k):
                return "not json at all"
        plan = interpret.interpret("what causes floods", llm=Broken())
        assert plan.query_terms  # fell back to the raw question

    def test_followup_query_is_merged_with_the_subject(self):
        plan = interpret.interpret("tell me more", llm=StubLLM())
        query = interpret.build_query("tell me more", plan, subject="flash drought")
        assert "flash drought" in query


class TestConversation:
    def test_single_turn_is_grounded_and_cited(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=False)
        turn = session.ask("what causes flash droughts")
        assert turn.sources
        assert not turn.refused
        assert turn.audit.groundedness == 1.0
        assert not turn.audit.invalid_markers

    def test_context_is_diversified_across_documents(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=False)
        turn = session.ask("what causes droughts")
        doc_ids = [s.document_id for s in turn.sources]
        assert len(set(doc_ids)) > 1, "context came from a single document"

    def test_list_mode_retrieves_more_context(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=False)
        listy = session.ask("what are the main types of drought")
        assert listy.interpretation.is_list
        assert len(listy.sources) > 8

    def test_guarded_question_carries_the_referral(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=False)
        turn = session.ask("should i buy flood insurance for my house")
        assert turn.guard_verdict.triggered
        assert "adviser" in turn.answer

    def test_multi_turn_keeps_the_subject(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=False)
        session.ask("what causes flash droughts")
        subject = session.memory.subject
        session.ask("tell me more about that")
        assert session.memory.subject == subject
        assert session.memory.turns == 2

    def test_empty_question_is_rejected(self, seeded):
        with pytest.raises(ValueError):
            ChatSession(llm=StubLLM(), persist=False).ask("   ")


class TestRelevanceGate:
    """The gate is what stops the system answering questions the corpus
    cannot address. Before it existed, out-of-domain refusal was 0%."""

    @pytest.mark.parametrize("question", [
        "how do i configure a kubernetes ingress controller",
        "what is the offside rule in association football",
        "what are the ingredients in a classic tiramisu",
    ])
    def test_out_of_domain_questions_are_refused(self, seeded, question):
        turn = ChatSession(llm=StubLLM(), persist=False).ask(question)
        assert turn.refused
        assert not turn.sources, "refused turns must not present sources"

    @pytest.mark.parametrize("question", [
        "what causes flash droughts to develop quickly",
        "how do tropical cyclones intensify over warm water",
        "what is business continuity planning",
    ])
    def test_in_domain_questions_are_answered(self, seeded, question):
        turn = ChatSession(llm=StubLLM(), persist=False).ask(question)
        assert not turn.refused
        assert turn.sources

    def test_gate_can_be_disabled_by_configuration(self, seeded, monkeypatch):
        from datahub_rag import config
        monkeypatch.setattr(config, "MIN_RELEVANCE", 0.0)
        turn = ChatSession(llm=StubLLM(), persist=False).ask(
            "what is the offside rule in association football")
        assert not turn.refused

    def test_dense_scores_separate_the_two_groups(self, seeded):
        """The threshold is only meaningful if the underlying signal separates.
        If this fails, MIN_RELEVANCE needs recalibrating, not nudging."""
        from datahub_rag import retrieve
        in_domain = retrieve.top_similarity("what causes flash droughts")
        out_of_domain = retrieve.top_similarity("how do i configure kubernetes ingress")
        assert in_domain > out_of_domain + 0.1


class TestPersistence:
    def test_turn_and_audit_trail_are_written(self, seeded):
        session = ChatSession(llm=StubLLM(), persist=True)
        turn = session.ask("what causes wildfires to spread")
        assert turn.message_id

        context = session.context_for(turn.message_id)
        assert len(context) == len(turn.sources)
        # The audit trail must record which sources the answer actually used.
        assert any(row["was_cited"] for row in context)

        history = session.history()
        assert [m["role"] for m in history] == ["user", "assistant"]

    def test_session_resumes_with_its_memory(self, seeded):
        first = ChatSession(llm=StubLLM(), persist=True)
        first.ask("how do tropical cyclones form")

        resumed = ChatSession(session_id=first.session_id, llm=StubLLM())
        assert resumed.memory.subject == first.memory.subject
        assert resumed.memory.turns == first.memory.turns

    def test_unknown_session_raises(self, seeded):
        with pytest.raises(LookupError):
            ChatSession(session_id=999_999_999, llm=StubLLM())
