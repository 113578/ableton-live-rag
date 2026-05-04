"""
Тесты модуля metrics: релевантность, метрики ранжирования и подсчёт эталонных страниц.
"""

import math

from experiments.metrics import (
    compute_relevances,
    count_total_relevant,
    hit_rate,
    is_page_relevant,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_is_page_relevant_requires_matching_source():
    ranges = [[10, 20]]

    assert is_page_relevant("live_12", 15, "live_12", ranges)
    assert not is_page_relevant("push_3", 15, "live_12", ranges)
    assert not is_page_relevant("live_12", 25, "live_12", ranges)


def test_compute_relevances_mixes_sources_and_ranges():
    retrieved = [
        ("live_12", 15),
        ("push_3", 15),
        ("live_12", 5),
        ("live_12", 20),
    ]

    rels = compute_relevances(
        retrieved=retrieved,
        ground_truth_source="live_12",
        ground_truth_ranges=[[10, 20]],
    )

    assert rels == [True, False, False, True]


def test_hit_rate_and_mrr():
    assert hit_rate([False, False, False]) == 0.0
    assert hit_rate([False, True]) == 1.0

    assert mrr([False, False, False]) == 0.0
    assert mrr([False, True, False]) == 0.5
    assert mrr([True, True]) == 1.0


def test_precision_and_recall_at_k():
    rels = [True, False, True, False]

    assert precision_at_k(rels) == 0.5
    assert precision_at_k(rels, k=2) == 0.5
    assert precision_at_k([], k=3) == 0.0

    assert recall_at_k(rels, total_relevant=5) == 0.4
    assert recall_at_k(rels, total_relevant=0) == 0.0


def test_ndcg_at_k_ranks_earlier_hits_higher():
    perfect = ndcg_at_k([True, False, False], total_relevant=1)
    later = ndcg_at_k([False, False, True], total_relevant=1)

    assert perfect == 1.0
    assert 0 < later < 1
    assert later == 1.0 / math.log2(4)

    assert ndcg_at_k([False, False], total_relevant=1) == 0.0


def test_count_total_relevant_deduplicates_overlapping_ranges():
    assert count_total_relevant([[1, 3], [3, 5]]) == 5
    assert count_total_relevant([[10, 10]]) == 1
    assert count_total_relevant([]) == 0
