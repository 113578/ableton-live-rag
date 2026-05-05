"""
Metrics for evaluating retrieval quality.
"""

import math


def hit_rate(relevances: list[bool]) -> float:
    """
    Compute Hit Rate.

    Parameters
    ----------
    relevances : list[bool]
        Binary relevance flags of the ranked list.

    Returns
    -------
    float
        ``1.0`` or ``0.0``.
    """

    return 1.0 if any(relevances) else 0.0


def mrr(relevances: list[bool]) -> float:
    """
    Compute Mean Reciprocal Rank.

    Parameters
    ----------
    relevances : list[bool]
        Binary relevance flags of the ranked list.

    Returns
    -------
    float
        MRR value. Returns ``0.0`` if no result is relevant.
    """

    for i, rel in enumerate(relevances):
        if rel:
            return 1.0 / (i + 1)

    return 0.0


def precision_at_k(relevances: list[bool], k: int | None = None) -> float:
    """
    Compute Precision@k.

    Parameters
    ----------
    relevances : list[bool]
        Binary relevance flags of the ranked list.
    k : int or None, optional
        First ``k`` results to consider.

    Returns
    -------
    float
        Fraction of relevant results.
    """

    rels = relevances[:k] if k else relevances
    if not rels:
        return 0.0

    return sum(rels) / len(rels)


def recall_at_k(
    relevances: list[bool],
    total_relevant: int,
    k: int | None = None,
) -> float:
    """
    Compute Recall@k.

    Parameters
    ----------
    relevances : list[bool]
        Binary relevance flags of the ranked list.
    total_relevant : int
        Total number of relevant documents in the collection.
    k : int or None, optional
        First ``k`` results to consider.

    Returns
    -------
    float
        Fraction of relevant documents that were retrieved.
    """

    if total_relevant == 0:
        return 0.0

    rels = relevances[:k] if k else relevances

    return sum(rels) / total_relevant


def ndcg_at_k(
    relevances: list[bool],
    total_relevant: int,
    k: int | None = None,
) -> float:
    """
    Compute NDCG@k.

    Parameters
    ----------
    relevances : list[bool]
        Binary relevance flags of the ranked list (one entry per unique
        page — pass ``dedup_relevances``).
    total_relevant : int
        Total number of relevant pages in the collection (from ground truth).
    k : int or None, optional
        First ``k`` results to consider.

    Returns
    -------
    float
        NDCG value in the range ``[0, 1]``.
    """

    rels = relevances[:k] if k else relevances

    if not rels or not any(rels):
        return 0.0

    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(rels))

    ideal_relevant = min(total_relevant, len(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def is_page_relevant(
    source: str,
    page: int,
    ground_truth_source: str,
    ground_truth_ranges: list[list[int]],
) -> bool:
    """
    Check whether a page is relevant: source matches and page falls in range.

    Parameters
    ----------
    source : str
        Source of the retrieved chunk (PDF name without extension).
    page : int
        Page number (1-indexed).
    ground_truth_source : str
        Source of the ground truth answer.
    ground_truth_ranges : list[list[int]]
        List of ``[start, end]`` pairs — ground-truth page ranges.

    Returns
    -------
    bool
        ``True`` if the source matches and the page falls within at least one range.
    """

    if source != ground_truth_source:
        return False

    return any(start <= page <= end for start, end in ground_truth_ranges)


def compute_relevances(
    retrieved: list[tuple[str, int]],
    ground_truth_source: str,
    ground_truth_ranges: list[list[int]],
) -> list[bool]:
    """
    Compute binary relevance for each search result.

    Parameters
    ----------
    retrieved : list[tuple[str, int]]
        List of ``(source, page_start)`` pairs from retrieved chunks' metadata.
    ground_truth_source : str
        Source of the ground-truth answer.
    ground_truth_ranges : list[list[int]]
        List of ``[start, end]`` pairs — ground-truth page ranges.

    Returns
    -------
    list[bool]
        List of booleans: ``True`` if the chunk is relevant.
    """

    return [
        is_page_relevant(s, p, ground_truth_source, ground_truth_ranges)
        for s, p in retrieved
    ]


def count_total_relevant(ground_truth_ranges: list[list[int]]) -> int:
    """
    Count the number of unique relevant pages in the ground truth.

    Parameters
    ----------
    ground_truth_ranges : list[list[int]]
        List of ``[start, end]`` pairs — ground-truth page ranges.

    Returns
    -------
    int
        Number of unique relevant pages.
    """

    pages: set[int] = set()

    for start, end in ground_truth_ranges:
        pages.update(range(start, end + 1))

    return len(pages)
