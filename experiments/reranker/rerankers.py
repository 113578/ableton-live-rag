"""
Конфигурации ранжировщиков для экспериментов.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore

RERANKER_MODELS: dict[str, str] = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge": "BAAI/bge-reranker-base",
}


@dataclass
class RerankerConfig:
    """
    Обёртка над ранжировщиком с единым интерфейсом для оценки.

    Attributes
    ----------
    name : str
        Название ранжировщика.
    description : str
        Описание ранжировщика.
    """

    name: str
    description: str
    _rerank_fn: Callable[[str, list[NodeWithScore], int], list[NodeWithScore]] = field(
        repr=False
    )

    def rerank(
        self, query: str, nodes: list[NodeWithScore], top_k: int
    ) -> list[NodeWithScore]:
        """
        Выполнение ранжирования.

        Parameters
        ----------
        query : str
            Поисковый запрос.
        nodes : list[NodeWithScore]
            Кандидаты от базового ретривера.
        top_k : int
            Количество результатов после ранжирования.

        Returns
        -------
        list[NodeWithScore]
            Переранжированные узлы с оценками.
        """

        return self._rerank_fn(query, nodes, top_k)


def _make_cross_encoder(model_id: str, name: str) -> RerankerConfig:
    """
    Создание RerankerConfig на основе cross-encoder модели.

    Parameters
    ----------
    model_id : str
        Идентификатор модели на HuggingFace.
    name : str
        Короткое имя для отображения в таблице.

    Returns
    -------
    RerankerConfig
        Конфигурация ранжировщика.
    """

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_id)

    def _rerank(
        query: str, nodes: list[NodeWithScore], top_k: int
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        texts = [n.node.get_content() for n in nodes]
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(nodes, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [NodeWithScore(node=n.node, score=float(s)) for n, s in ranked[:top_k]]

    return RerankerConfig(
        name=name,
        description=f"Cross-encoder reranking ({model_id})",
        _rerank_fn=_rerank,
    )


def build_all_rerankers(
    selected: list[str] | None = None,
) -> list[RerankerConfig]:
    """
    Создание всех ранжировщиков для эксперимента.

    Parameters
    ----------
    selected : list[str] or None, optional
        Список ключей из ``RERANKER_MODELS`` для загрузки.
        ``None`` означает все доступные модели.

    Returns
    -------
    list[RerankerConfig]
        Список конфигураций ранжировщиков.
    """

    keys = selected if selected is not None else list(RERANKER_MODELS)

    return [_make_cross_encoder(model_id=RERANKER_MODELS[k], name=k) for k in keys]
