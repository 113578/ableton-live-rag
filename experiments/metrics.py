"""
Метрики для оценки качества поиска.
"""

import math


def hit_rate(relevances: list[bool]) -> float:
    """
    Расчёт Hit Rate.

    Parameters
    ----------
    relevances : list[bool]
        Бинарные релевантности ранжированного списка.

    Returns
    -------
    float
        1.0 или 0.0.
    """

    return 1.0 if any(relevances) else 0.0


def mrr(relevances: list[bool]) -> float:
    """
    Расчёт Mean Reciprocal Rank.

    Parameters
    ----------
    relevances : list[bool]
        Бинарные релевантности ранжированного списка.

    Returns
    -------
    float
        Значение MRR. Возвращает 0.0 если ни один результат не релевантен.
    """

    for i, rel in enumerate(relevances):
        if rel:
            return 1.0 / (i + 1)

    return 0.0


def precision_at_k(relevances: list[bool], k: int | None = None) -> float:
    """
    Расчёт Precision@k.

    Parameters
    ----------
    relevances : list[bool]
        Бинарные релевантности ранжированного списка.
    k : int or None, optional
        Первые k результатов.

    Returns
    -------
    float
        Доля релевантных результатов.
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
    Расчёт Recall@k.

    Parameters
    ----------
    relevances : list[bool]
        Бинарные релевантности ранжированного списка.
    total_relevant : int
        Общее число релевантных документов в коллекции.
    k : int or None, optional
        Первые k результатов.

    Returns
    -------
    float
        Доля найденных релевантных документов.
    """

    if total_relevant == 0:
        return 0.0

    rels = relevances[:k] if k else relevances

    return sum(rels) / total_relevant


def ndcg_at_k(relevances: list[bool], k: int | None = None) -> float:
    """
    Расчёт NDCG@k.

    Parameters
    ----------
    relevances : list[bool]
        Бинарные релевантности ранжированного списка.
    k : int or None, optional
        Первые k результатов.

    Returns
    -------
    float
        Значение NDCG в диапазоне [0, 1].
    """

    rels = relevances[:k] if k else relevances

    if not rels or not any(rels):
        return 0.0

    dcg = sum((1.0 if rel else 0.0) / math.log2(i + 2) for i, rel in enumerate(rels))

    n_relevant = sum(rels)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    return dcg / idcg if idcg > 0 else 0.0


def is_page_relevant(
    page: int,
    ground_truth_ranges: list[list[int]],
) -> bool:
    """
    Проверка попадания страницы в один из диапазонов ground truth.

    Parameters
    ----------
    page : int
        Номер страницы (1-indexed).
    ground_truth_ranges : list[list[int]]
        Список пар ``[start, end]`` — эталонные диапазоны страниц.

    Returns
    -------
    bool
        ``True`` если страница попадает хотя бы в один диапазон.
    """

    return any(start <= page <= end for start, end in ground_truth_ranges)


def compute_relevances(
    retrieved_pages: list[int],
    ground_truth_ranges: list[list[int]],
) -> list[bool]:
    """
    Вычисление бинарной релевантности для каждого результата поиска.

    Parameters
    ----------
    retrieved_pages : list[int]
        Список значений ``page_start`` из метаданных найденных чанков.
    ground_truth_ranges : list[list[int]]
        Список пар ``[start, end]`` — эталонные диапазоны страниц.

    Returns
    -------
    list[bool]
        Список булевых значений: ``True`` если чанк релевантен.
    """

    return [is_page_relevant(p, ground_truth_ranges) for p in retrieved_pages]


def count_total_relevant(ground_truth_ranges: list[list[int]]) -> int:
    """
    Подсчёт числа уникальных релевантных страниц в ground truth.

    Parameters
    ----------
    ground_truth_ranges : list[list[int]]
        Список пар ``[start, end]`` — эталонные диапазоны страниц.

    Returns
    -------
    int
        Число уникальных релевантных страниц.
    """

    pages: set[int] = set()

    for start, end in ground_truth_ranges:
        pages.update(range(start, end + 1))

    return len(pages)
