"""
Общие утилиты для экспериментов.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console

from ableton_live_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_live_rag.index import load_index, parse_nodes
from ableton_live_rag.ingest import load_documents
from experiments.metrics import (
    compute_relevances,
    count_total_relevant,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

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


def load_indexes(
    models: dict[str, EmbeddingModelConfig] | None = None,
) -> dict[str, VectorStoreIndex]:
    """
    Загрузка Qdrant-индексов для выбранных моделей эмбеддингов.

    Parameters
    ----------
    models : dict[str, EmbeddingModelConfig] or None, optional
        Модели, для которых нужно загрузить индексы. По умолчанию все.

    Returns
    -------
    dict[str, VectorStoreIndex]
        Индексы по имени модели.
    """

    models = models or EMBEDDING_MODELS
    indexes: dict[str, VectorStoreIndex] = {}

    for key, emb in models.items():
        console.print(f"[dim]  Загрузка индекса {emb.collection_name}...[/dim]")
        LlamaSettings.embed_model = make_embed_model(emb)
        indexes[key] = load_index(collection_name=emb.collection_name)

    return indexes


def prepare_experiment() -> tuple[
    dict[str, VectorStoreIndex], list[BaseNode], list[dict]
]:
    """
    Подготовка окружения для эксперимента.

    Настраивает LlamaSettings, загружает индексы, парсит узлы
    и загружает валидационный набор данных.

    Returns
    -------
    tuple[dict[str, VectorStoreIndex], list[BaseNode], list[dict]]
        Индексы, узлы и набор данных.
    """

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    console.print("[dim]Загрузка индексов из Qdrant...[/dim]")
    indexes = load_indexes()

    console.print("[dim]Парсинг документов в узлы...[/dim]")
    documents = load_documents()
    nodes = parse_nodes(documents=documents)
    console.print(f"[green]Получено {len(nodes)} узлов[/green]")

    dataset = load_dataset()
    console.print(f"[green]Загружено {len(dataset)} вопросов из eval-датасета[/green]")

    return indexes, nodes, dataset


def evaluate_dataset(
    retrieve_fn: Callable[[str], list[NodeWithScore]],
    dataset: list[dict],
) -> tuple[list[dict], float]:
    """
    Оценка функции поиска на наборе данных.

    Parameters
    ----------
    retrieve_fn : Callable[[str], list[NodeWithScore]]
        Функция поиска: принимает запрос, возвращает узлы.
    dataset : list[dict]
        Валидационный набор данных.

    Returns
    -------
    tuple[list[dict], float]
        Результаты по каждому вопросу и суммарное время.
    """

    per_question: list[dict] = []
    total_time = 0.0

    for item in dataset:
        t0 = time.perf_counter()

        try:
            nodes = retrieve_fn(item["question"])
        except Exception as e:
            console.print(f"[red]  Ошибка на '{item['id']}': {e}[/red]")
            per_question.append({"id": item["id"], "error": str(e), "relevances": []})
            continue

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        retrieved_pages = [n.metadata.get("page_start", 0) for n in nodes]
        gt = item["ground_truth_pages"]
        rels = compute_relevances(
            retrieved_pages=retrieved_pages, ground_truth_ranges=gt
        )

        per_question.append(
            {
                "id": item["id"],
                "relevances": rels,
                "retrieved_pages": retrieved_pages,
                "total_relevant": count_total_relevant(ground_truth_ranges=gt),
                "latency_s": round(elapsed, 3),
            }
        )

    return per_question, total_time


def aggregate_metrics(per_question: list[dict], total_time: float) -> dict:
    """
    Агрегация метрик по результатам оценки.

    Parameters
    ----------
    per_question : list[dict]
        Результаты по каждому вопросу от ``evaluate_dataset()``.
    total_time : float
        Суммарное время выполнения.

    Returns
    -------
    dict
        Словарь с метриками и ``details``.
    """

    valid = [q for q in per_question if "error" not in q]
    n = len(valid) or 1

    return {
        "hit_rate": round(sum(hit_rate(q["relevances"]) for q in valid) / n, 3),
        "mrr": round(sum(mrr(q["relevances"]) for q in valid) / n, 3),
        "precision": round(sum(precision_at_k(q["relevances"]) for q in valid) / n, 3),
        "recall": round(
            sum(recall_at_k(q["relevances"], q["total_relevant"]) for q in valid) / n,
            3,
        ),
        "ndcg": round(sum(ndcg_at_k(q["relevances"]) for q in valid) / n, 3),
        "avg_latency_s": round(total_time / n, 3),
        "errors": sum(1 for q in per_question if "error" in q),
        "details": per_question,
    }


def format_result_summary(result: dict) -> str:
    """
    Форматирование однострочной сводки результатов.

    Parameters
    ----------
    result : dict
        Результат от ``aggregate_metrics()``.

    Returns
    -------
    str
        Строка вида ``Hit Rate=0.xxx  MRR=0.xxx  NDCG=0.xxx  (0.xxxs/query)``.
    """

    return (
        f"  Hit Rate={result['hit_rate']:.3f}  "
        f"MRR={result['mrr']:.3f}  "
        f"NDCG={result['ndcg']:.3f}  "
        f"({result['avg_latency_s']:.3f}s/query)"
    )


def save_results(results: list[dict], results_dir: Path) -> Path:
    """
    Сохранение результатов в JSON-файл.

    Parameters
    ----------
    results : list[dict]
        Результаты экспериментов.
    results_dir : Path
        Директория для сохранения.

    Returns
    -------
    Path
        Путь к сохранённому файлу.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"eval_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Результаты сохранены: {out_path}[/green]")

    return out_path
