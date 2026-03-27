"""
Общие утилиты для экспериментов.
"""

import json
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console

from ableton_live_rag.config import EmbeddingModelConfig
from ableton_live_rag.index import load_index

_EXPERIMENTS_DIR = Path(__file__).resolve().parent
DATASET_PATH = _EXPERIMENTS_DIR / "eval_dataset.json"

console = Console()


def load_dataset() -> list[dict]:
    """
    Загрузка валидационного набора данных.

    Returns
    -------
    list[dict]
        Список вопросов с полями ``id``, ``question``,
        ``ground_truth_pages``, ``category``.
    """

    with open(DATASET_PATH) as f:
        return json.load(f)


def load_indexes(
    models: dict[str, EmbeddingModelConfig],
) -> dict[str, VectorStoreIndex]:
    """
    Загрузка Qdrant-индексов для выбранных моделей эмбеддингов.

    Parameters
    ----------
    models : dict[str, EmbeddingModelConfig]
        Модели, для которых нужно загрузить индексы.

    Returns
    -------
    dict[str, VectorStoreIndex]
        Индексы по имени модели.
    """

    indexes: dict[str, VectorStoreIndex] = {}

    for key, emb in models.items():
        console.print(f"[dim]  Загрузка индекса {emb.collection_name}...[/dim]")

        LlamaSettings.embed_model = HuggingFaceEmbedding(
            model_name=emb.model_id,
            query_instruction=emb.query_instruction or None,
            text_instruction=emb.text_instruction or None,
        )

        indexes[key] = load_index(collection_name=emb.collection_name)

    return indexes


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
