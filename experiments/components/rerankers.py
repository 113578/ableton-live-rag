"""
Reranker configurations for experiments.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

RERANKER_MODELS: dict[str, str] = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge": "BAAI/bge-reranker-base",
}


@dataclass
class RerankerConfig:
    """
    Wrapper around a reranker exposing a unified interface for evaluation.

    Attributes
    ----------
    name : str
        Reranker name.
    description : str
        Reranker description.
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
        Run the reranking.

        Parameters
        ----------
        query : str
            Search query.
        nodes : list[NodeWithScore]
            Candidates from the base retriever.
        top_k : int
            Number of results after reranking.

        Returns
        -------
        list[NodeWithScore]
            Reranked nodes with scores.
        """

        return self._rerank_fn(query, nodes, top_k)


def _make_cross_encoder(model_id: str, name: str) -> RerankerConfig:
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


def build_rerankers(
    selected: list[str] | None = None,
) -> list[RerankerConfig]:
    """
    Create reranker configurations for an experiment.

    Parameters
    ----------
    selected : list[str] or None, optional
        Keys from ``RERANKER_MODELS``. ``None`` loads all available models.

    Returns
    -------
    list[RerankerConfig]
        List of reranker configurations.
    """

    keys = selected if selected is not None else list(RERANKER_MODELS)

    return [_make_cross_encoder(model_id=RERANKER_MODELS[k], name=k) for k in keys]
