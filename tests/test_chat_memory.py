"""Memory is LLM-free, so it is fully deterministic and testable."""

from datahub_rag.chat.memory import Memory


class TestSubjectTracking:
    def test_first_question_sets_the_subject(self):
        m = Memory()
        m.observe("how do tropical cyclones form", is_followup=False, entities=["cyclones"])
        assert m.subject == "cyclones"

    def test_followup_preserves_the_subject(self):
        """Without this, 'tell me more' would reset the topic and lose it."""
        m = Memory()
        m.observe("how do tropical cyclones form", is_followup=False, entities=["cyclones"])
        m.observe("tell me more about that", is_followup=True, entities=[])
        assert m.subject == "cyclones"

    def test_new_question_replaces_the_subject(self):
        m = Memory()
        m.observe("how do cyclones form", is_followup=False, entities=["cyclones"])
        m.observe("what causes wildfires", is_followup=False, entities=["wildfires"])
        assert m.subject == "wildfires"

    def test_subject_falls_back_to_content_words_without_entities(self):
        m = Memory()
        m.observe("what are the main types of drought", is_followup=False, entities=[])
        assert "drought" in m.subject
        assert "what" not in m.subject


class TestFacts:
    def test_entities_outrank_bare_keywords(self):
        m = Memory()
        m.observe("flooding and drainage in coastal cities", is_followup=False,
                  entities=["rotterdam"])
        assert m.facts[0] == "rotterdam"

    def test_repeated_terms_rise(self):
        m = Memory()
        for _ in range(3):
            m.observe("drought monitoring indices", is_followup=False, entities=[])
        assert m.facts[0] == "drought"

    def test_stopwords_excluded(self):
        m = Memory()
        m.observe("what about the thing that they were doing", is_followup=False, entities=[])
        assert "that" not in m.facts and "they" not in m.facts

    def test_facts_are_capped(self):
        m = Memory()
        for i in range(40):
            m.observe(f"term{i} hazard analysis method", is_followup=False, entities=[])
        assert len(m.facts) <= 12


class TestWindow:
    def test_turns_increment(self):
        m = Memory()
        m.observe("one question here", False, [])
        m.observe("another question here", False, [])
        assert m.turns == 2

    def test_summary_keeps_only_recent_turns(self):
        m = Memory()
        for i in range(10):
            m.observe(f"question number {i} about hazards", False, [])
        assert "question number 9" in m.summary
        assert "question number 0" not in m.summary

    def test_roundtrip_through_a_database_row(self):
        m = Memory()
        m.observe("how do cyclones form", False, ["cyclones"])
        restored = Memory.from_row(m.as_dict())
        assert restored.subject == m.subject
        assert restored.facts == m.facts
        assert restored.turns == m.turns

    def test_prompt_context_includes_subject_and_facts(self):
        m = Memory()
        m.observe("how do cyclones form", False, ["cyclones"])
        context = m.as_prompt_context()
        assert "cyclones" in context

    def test_empty_memory_yields_empty_context(self):
        assert Memory().as_prompt_context() == ""
