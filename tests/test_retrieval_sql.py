"""Integration tests against the seeded database.

These need the stack running (`make up`); they are skipped otherwise rather
than failing, so `pytest` is still useful without Postgres.
"""

import pytest

from datahub_rag import config, retrieve, store


@pytest.fixture(scope="module")
def seeded():
    try:
        census = store.corpus_stats()
    except Exception as exc:
        pytest.skip(f"database unavailable: {exc}")
    if not census["chunks"]:
        pytest.skip("corpus not seeded; run `make pipeline`")
    return census


@pytest.mark.parametrize("mode", ["vector", "lexical", "hybrid"])
def test_every_mode_returns_results(seeded, mode):
    hits = retrieve.search("earthquake early warning", mode=mode, top_k=5)
    assert hits, f"{mode} returned nothing"
    assert len(hits) <= 5


@pytest.mark.parametrize("mode", ["vector", "lexical", "hybrid"])
def test_results_are_ordered_by_descending_score(seeded, mode):
    hits = retrieve.search("flood risk management", mode=mode, top_k=10)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_honoured(seeded):
    assert len(retrieve.search("drought", mode="hybrid", top_k=3)) <= 3


def test_source_filter_applies_to_both_hybrid_arms(seeded):
    # A filter that leaked on only one arm would show up as foreign sources
    # surviving fusion.
    hits = retrieve.search("climate adaptation", mode="hybrid", top_k=20,
                           filters=retrieve.Filters(source="wikipedia"))
    assert hits
    with store.connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT d.source FROM documents d WHERE d.id = ANY(%s)",
            ([h.document_id for h in hits],),
        ).fetchall()
    assert {r["source"] for r in rows} == {"wikipedia"}


def test_date_filter_excludes_earlier_documents(seeded):
    hits = retrieve.search("disaster", mode="vector", top_k=20,
                           filters=retrieve.Filters(published_from="2015-01-01"))
    assert all(h.published_at >= "2015-01-01" for h in hits if h.published_at)


def test_unknown_mode_is_rejected(seeded):
    with pytest.raises(ValueError):
        retrieve.search("x", mode="bm25")


def test_rrf_weighting_changes_the_ranking(seeded):
    """The weights must actually reach the SQL, not sit unused in config."""
    dense_leaning = retrieve.search("flood risk", mode="hybrid", top_k=10,
                                    weights=(10.0, 1.0))
    lexical_leaning = retrieve.search("flood risk", mode="hybrid", top_k=10,
                                      weights=(1.0, 10.0))
    assert [h.chunk_id for h in dense_leaning] != [h.chunk_id for h in lexical_leaning]


def test_lexical_arm_matches_partial_term_overlap(seeded):
    """Regression: websearch_to_tsquery ANDs terms, so a long natural-language
    question used to match nothing at all."""
    hits = retrieve.search(
        "how do communities prepare for flooding before it happens",
        mode="lexical", top_k=5,
    )
    assert hits


def test_embedding_table_name_is_stable(seeded):
    assert config.embedding_table("bge-small-en-v1.5") == "chunk_embeddings_bge_small_en_v1_5"
