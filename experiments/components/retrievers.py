"""
Конфигурации ретриверов для экспериментов.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ableton_live_rag.config import EmbeddingModelConfig


@dataclass
class RetrieverConfig:
    """
    Обёртка над ретривером с единым интерфейсом для оценки.

    Attributes
    ----------
    name : str
        Название ретривера.
    description : str
        Описание ретривера.
    category : str
        Категория ретривера: ``sparse``, ``dense``, ``hybrid``.
    """

    name: str
    description: str
    category: str
    _retrieve_fn: Callable[[str, int], list[NodeWithScore]] = field(repr=False)

    def retrieve(self, query: str, top_k: int = 5) -> list[NodeWithScore]:
        """
        Запуск поиска.

        Parameters
        ----------
        query : str
            Поисковый запрос.
        top_k : int
            Количество результатов.

        Returns
        -------
        list[NodeWithScore]
            Ранжированные узлы с оценками.
        """

        return self._retrieve_fn(query, top_k)


def make_embed_model(emb: EmbeddingModelConfig) -> HuggingFaceEmbedding:
    """
    Создание HuggingFaceEmbedding из конфигурации.

    Parameters
    ----------
    emb : EmbeddingModelConfig
        Конфигурация модели эмбеддингов.

    Returns
    -------
    HuggingFaceEmbedding
        Модель эмбеддингов.
    """

    return HuggingFaceEmbedding(
        model_name=emb.model_id,
        query_instruction=emb.query_instruction or None,
        text_instruction=emb.text_instruction or None,
    )


def _make_bm25(bm25: BM25Retriever) -> RetrieverConfig:
    def _retrieve(query: str, top_k: int) -> list[NodeWithScore]:
        bm25.similarity_top_k = top_k
        return bm25.retrieve(query)

    return RetrieverConfig(
        name="bm25",
        description="BM25 keyword search",
        category="sparse",
        _retrieve_fn=_retrieve,
    )


def _make_tfidf(nodes: list[BaseNode]) -> RetrieverConfig:
    texts = [n.get_content() for n in nodes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(raw_documents=texts)

    def _retrieve(query: str, top_k: int) -> list[NodeWithScore]:
        query_vec = vectorizer.transform(raw_documents=[query])
        scores = cosine_similarity(X=query_vec, Y=tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            NodeWithScore(node=nodes[i], score=float(scores[i])) for i in top_indices
        ]

    return RetrieverConfig(
        name="tfidf",
        description="TF-IDF and cosine similarity",
        category="sparse",
        _retrieve_fn=_retrieve,
    )


def _make_vector(
    index: VectorStoreIndex,
    model_name: str,
    embed_model: HuggingFaceEmbedding,
) -> RetrieverConfig:
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    def _retrieve(query: str, top_k: int) -> list[NodeWithScore]:
        LlamaSettings.embed_model = embed_model
        retriever.similarity_top_k = top_k
        return retriever.retrieve(query)

    return RetrieverConfig(
        name=f"vector/{model_name}",
        description=f"Cosine similarity ({model_name})",
        category="dense",
        _retrieve_fn=_retrieve,
    )


def _reciprocal_rank_fusion(
    results_list: list[list[NodeWithScore]],
    top_k: int,
    rrf_k: int = 60,
) -> list[NodeWithScore]:
    scores: dict[str, float] = {}
    nodes_map: dict[str, NodeWithScore] = {}

    for results in results_list:
        for rank, node_with_score in enumerate(results):
            node_id = node_with_score.node.node_id

            if node_id not in nodes_map:
                nodes_map[node_id] = node_with_score

            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (rrf_k + rank + 1)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]

    return [
        NodeWithScore(node=nodes_map[nid].node, score=scores[nid]) for nid in sorted_ids
    ]


def _make_hybrid(
    index: VectorStoreIndex,
    bm25: BM25Retriever,
    model_name: str,
    embed_model: HuggingFaceEmbedding,
) -> RetrieverConfig:
    vec_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    def _retrieve(query: str, top_k: int) -> list[NodeWithScore]:
        LlamaSettings.embed_model = embed_model
        vec_retriever.similarity_top_k = top_k
        bm25.similarity_top_k = top_k

        vec_results = vec_retriever.retrieve(query)
        bm25_results = bm25.retrieve(query)

        return _reciprocal_rank_fusion([vec_results, bm25_results], top_k)

    return RetrieverConfig(
        name=f"hybrid_rrf/{model_name}",
        description=f"BM25 + Vector RRF ({model_name})",
        category="hybrid",
        _retrieve_fn=_retrieve,
    )


def build_retrievers(
    indexes: dict[str, VectorStoreIndex],
    nodes: list[BaseNode],
    embedding_configs: dict[str, EmbeddingModelConfig],
) -> list[RetrieverConfig]:
    """
    Создание конфигураций ретриверов для эксперимента.

    Parameters
    ----------
    indexes : dict[str, VectorStoreIndex]
        Индексы по имени модели эмбеддингов.
    nodes : list[BaseNode]
        Узлы корпуса (для sparse-ретриверов).
    embedding_configs : dict[str, EmbeddingModelConfig]
        Конфигурации моделей эмбеддингов.

    Returns
    -------
    list[RetrieverConfig]
        Список конфигураций ретриверов.
    """

    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)

    configs: list[RetrieverConfig] = [
        _make_bm25(bm25),
        _make_tfidf(nodes),
    ]

    for model_name, index in indexes.items():
        embed_model = make_embed_model(embedding_configs[model_name])
        configs.append(_make_vector(index, model_name, embed_model))
        configs.append(_make_hybrid(index, bm25, model_name, embed_model))

    return configs
