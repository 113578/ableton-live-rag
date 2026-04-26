"""
Общие утилиты для экспериментов.
"""

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import typer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode, NodeWithScore
from rich.console import Console

from ableton_live_rag.config import EMBEDDING_MODELS, settings
from experiments.components.generators import (
    GeneratorConfig,
    load_judge_spec,
    make_llm,
)
from experiments.components.rerankers import RerankerConfig
from experiments.components.retrievers import RetrieverConfig, make_embed_model
from experiments.metrics import (
    LlamaIndexJudge,
    ameasure,
    build_metrics,
    compute_relevances,
    count_total_relevant,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ableton_live_rag.index import load_index, parse_nodes
from ableton_live_rag.ingest import load_documents

_PACKAGE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = _PACKAGE_DIR / "datasets"
DATASET_PATH = DATASETS_DIR / "eval.json"
RESULTS_DIR = _PACKAGE_DIR / "eval_results"

GENERATOR_META_KEYS: set[str] = {
    "generator",
    "description",
    "pipeline",
    "avg_latency_s",
    "errors",
    "details",
}

console = Console()


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """
    Загрузка валидационного набора данных.

    Parameters
    ----------
    path : Path
        Путь к JSON-файлу с вопросами.

    Returns
    -------
    list[dict]
        Список вопросов с полями ``id``, ``question``, ``source``,
        ``ground_truth_pages``, ``category``.
    """

    with open(path) as f:
        return json.load(f)


def load_indexes(
    models: dict | None = None,
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


def find_by_name(configs: list, name: str, kind: str):
    """
    Поиск конфигурации по имени.

    Parameters
    ----------
    configs : list
        Список конфигураций (ретриверов, ранжировщиков, генераторов).
    name : str
        Искомое имя.
    kind : str
        Тип конфигурации для сообщения об ошибке.

    Returns
    -------
    Конфигурация с совпадающим именем.

    Raises
    ------
    typer.Exit
        Если конфигурация не найдена.
    """

    found = next((c for c in configs if c.name == name), None)

    if found is None:
        available = [c.name for c in configs]
        console.print(f"[red]{kind} {name!r} не найден. Доступные: {available}[/red]")
        raise typer.Exit(1)

    return found


def build_retrieval_pipeline(
    retriever: RetrieverConfig,
    reranker: RerankerConfig | None,
    top_k: int,
    candidate_k: int,
) -> Callable[[str], list[NodeWithScore]]:
    """
    Сборка retrieval-пайплайна: ретривер [+ ранжировщик] → top_k узлов.

    Parameters
    ----------
    retriever : RetrieverConfig
        Базовый ретривер.
    reranker : RerankerConfig or None
        Ранжировщик. ``None`` — без переранжирования.
    top_k : int
        Количество финальных результатов.
    candidate_k : int
        Размер пула кандидатов для ранжировщика.

    Returns
    -------
    Callable[[str], list[NodeWithScore]]
        Функция поиска.
    """

    pool = candidate_k if reranker is not None else top_k

    def _retrieve(query: str) -> list[NodeWithScore]:
        candidates = retriever.retrieve(query=query, top_k=pool)

        if reranker is not None:
            return reranker.rerank(query=query, nodes=candidates, top_k=top_k)

        return candidates[:top_k]

    return _retrieve


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

        retrieved = [
            (n.metadata.get("source", ""), n.metadata.get("page_start", 0))
            for n in nodes
        ]

        gt = item["ground_truth_pages"]
        gt_source = item["source"]

        rels = compute_relevances(
            retrieved=retrieved,
            ground_truth_source=gt_source,
            ground_truth_ranges=gt,
        )

        per_question.append(
            {
                "id": item["id"],
                "relevances": rels,
                "retrieved": [f"{s}:p{p}" for s, p in retrieved],
                "total_relevant": count_total_relevant(ground_truth_ranges=gt),
                "latency_s": round(elapsed, 3),
            }
        )

    return per_question, total_time


def aggregate_retrieval_metrics(per_question: list[dict], total_time: float) -> dict:
    """
    Агрегация метрик по результатам оценки поиска.

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


def format_retrieval_summary(result: dict) -> str:
    """
    Форматирование однострочной сводки результатов поиска.

    Parameters
    ----------
    result : dict
        Результат от ``aggregate_retrieval_metrics()``.

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


def col_header(key: str) -> str:
    """
    Преобразование ключа метрики в заголовок колонки таблицы.

    Parameters
    ----------
    key : str
        Ключ метрики.

    Returns
    -------
    str
        Заголовок.
    """

    return key.replace("_", " ").title()


def format_generator_summary(result: dict) -> str:
    """
    Форматирование однострочной сводки результатов генерации.

    Parameters
    ----------
    result : dict
        Результат с метриками генерации.

    Returns
    -------
    str
        Строка вида ``Metric1=0.xxx  Metric2=0.xxx  (0.xxxs/query)``.
    """

    score_keys = [k for k in result if k not in GENERATOR_META_KEYS]
    parts = [f"{col_header(k)}={result[k]:.3f}" for k in score_keys]

    return "  " + "  ".join(parts) + f"  ({result['avg_latency_s']:.3f}s/query)"


async def evaluate_generator(
    generator: GeneratorConfig,
    retrieve_fn: Callable[[str], list[NodeWithScore]],
    metrics: dict[str, BaseMetric],
    dataset: list[dict],
    concurrency: int = 16,
) -> dict:
    """
    Сквозная оценка генератора на наборе данных.

    Parameters
    ----------
    generator : GeneratorConfig
        Генератор ответов.
    retrieve_fn : Callable[[str], list[NodeWithScore]]
        Функция поиска контекста.
    metrics : dict[str, BaseMetric]
        Метрики DeepEval.
    dataset : list[dict]
        Валидационный набор данных.
    concurrency : int
        Число параллельных запросов.

    Returns
    -------
    dict
        Агрегированные метрики и ``details`` по вопросам.
    """

    semaphore = asyncio.Semaphore(concurrency)

    async def _process(item: dict) -> dict:
        async with semaphore:
            question = item["question"]

            try:
                t0 = time.perf_counter()
                nodes = retrieve_fn(question)
                contexts = [n.node.get_content() for n in nodes]
                answer = await generator.agenerate(question=question, contexts=contexts)
                latency = time.perf_counter() - t0

                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=contexts,
                )
                scores = await ameasure(test_case=test_case, metrics=metrics)

                return {"id": item["id"], "latency_s": round(latency, 3), **scores}

            except Exception as e:
                console.print(f"[red]  Ошибка на '{item['id']}': {e}[/red]")

                return {"id": item["id"], "error": str(e)}

    per_question: list[dict] = list(
        await asyncio.gather(*[_process(i) for i in dataset])
    )

    valid = [q for q in per_question if "error" not in q]
    n = len(valid) or 1
    errors = len(per_question) - len(valid)
    score_keys = [
        k for k in (valid[0] if valid else {}) if k not in ("id", "latency_s")
    ]

    return {
        "generator": generator.name,
        "description": generator.description,
        **{
            key: round(sum(q.get(key, 0.0) for q in valid) / n, 3) for key in score_keys
        },
        "avg_latency_s": round(sum(q.get("latency_s", 0.0) for q in valid) / n, 3),
        "errors": errors,
        "details": per_question,
    }


def build_judge_metrics() -> dict[str, BaseMetric]:
    """
    Инициализация судьи DeepEval и набора метрик генерации.

    Returns
    -------
    dict[str, BaseMetric]
        Метрики DeepEval, готовые к использованию.
    """

    judge_spec = load_judge_spec()
    console.print(
        f"[dim]Инициализация судьи DeepEval "
        f"({judge_spec.backend}/{judge_spec.model_id})...[/dim]"
    )
    judge = LlamaIndexJudge(llm=make_llm(judge_spec), name=judge_spec.name)

    return build_metrics(judge=judge)


def save_results(results: list[dict], results_dir: Path) -> Path:
    """
    Сохранение результатов в JSON.

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

    json_path = results_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Результаты сохранены: {json_path}[/green]")

    return json_path
