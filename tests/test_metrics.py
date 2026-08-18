"""Metrics are checked against hand-computed values, not against themselves."""

import math

import pytest

from eval.metrics import ndcg_at_k, percentile, recall_at_k, reciprocal_rank


class TestRecall:
    def test_all_relevant_found(self):
        assert recall_at_k([1, 2, 3], {1, 2}, 10) == 1.0

    def test_half_found(self):
        assert recall_at_k([1, 9], {1, 2}, 10) == 0.5

    def test_respects_the_cutoff(self):
        assert recall_at_k([9, 9, 1], {1}, 2) == 0.0
        assert recall_at_k([9, 9, 1], {1}, 3) == 1.0

    def test_empty_relevant_set(self):
        assert recall_at_k([1], set(), 10) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([7, 8], {7}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank([1, 2, 7], {7}) == pytest.approx(1 / 3)

    def test_not_found(self):
        assert reciprocal_rank([1, 2], {7}) == 0.0

    def test_uses_the_earliest_relevant_item(self):
        assert reciprocal_rank([5, 7], {5, 7}) == 1.0


class TestNDCG:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([1, 2, 3], {1, 2}, 10) == pytest.approx(1.0)

    def test_hand_computed_single_relevant_at_rank_two(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        assert ndcg_at_k([9, 1], {1}, 10) == pytest.approx(1 / math.log2(3))

    def test_worse_ranking_scores_lower(self):
        assert ndcg_at_k([9, 9, 1], {1}, 10) < ndcg_at_k([9, 1, 9], {1}, 10)

    def test_nothing_relevant(self):
        assert ndcg_at_k([9, 9], {1}, 10) == 0.0


class TestPercentile:
    def test_median_of_odd_length(self):
        assert percentile([3, 1, 2], 50) == 2

    def test_p95_picks_the_top_of_a_small_sample(self):
        assert percentile(list(range(1, 101)), 95) == 95

    def test_empty(self):
        assert percentile([], 50) == 0.0
