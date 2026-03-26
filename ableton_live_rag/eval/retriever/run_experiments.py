"""
Запуск экспериментов по оценке компонент поиска.

Перед запуском постройте индексы: ``uv run scripts/build_eval_indexes.py``.
"""

import json
import time
from pathlib import Path

import typer
from llama_index.core import Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_live_rag.eval.retrieval_metrics import (
    compute_relevances,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ableton_live_rag.eval.retrievers import (
    RetrieverConfig,
    apply_rerank,
    build_all_retrievers,
)
from ableton_live_rag.index import load_index, parse_nodes
from ableton_live_rag.ingest import load_documents

app = typer.Typer(no_args_is_help=False)
console = Console()

DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"


def load_dataset() -> list[dict]:
    """
    Загрузить eval-датасет из JSON.

    Returns
    -------
    list[dict]
        Список вопросов с полями ``id``, ``question``,
        ``ground_truth_pages``, ``category``.
    """

    with open(DATASET_PATH) as f:
        return json.load(f)


def _load_indexes(
    models: dict[str, EmbeddingModelConfig],
) -> dict[str, VectorStoreIndex]:
    """
    Загрузить Qdrant-индексы для выбранных моделей эмбеддингов.

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


def _count_total_relevant(ground_truth_ranges: list[list[int]]) -> int:
    """Подсчитать общее число релевантных страниц в ground truth."""
    pages: set[int] = set()
    for start, end in ground_truth_ranges:
        pages.update(range(start, end + 1))
    return len(pages)


def evaluate_retriever(
    config: RetrieverConfig,
    dataset: list[dict],
    top_k: int,
) -> dict:
    """
    Оценить один ретривер на всём датасете.

    Parameters
    ----------
    config : RetrieverConfig
        Конфигурация оцениваемого ретривера.
    dataset : list[dict]
        Eval-датасет (из ``load_dataset()``).
    top_k : int
        Количество результатов поиска.

    Returns
    -------
    dict
        Словарь с агрегированными метриками и ``details``
        по каждому вопросу.
    """
    per_question: list[dict] = []
    total_time = 0.0

    for item in dataset:
        t0 = time.perf_counter()
        try:
            nodes = config.retrieve(item["question"], top_k=top_k)
        except Exception as e:
            console.print(f"[red]  Ошибка на '{item['id']}': {e}[/red]")
            per_question.append(
                {
                    "id": item["id"],
                    "error": str(e),
                    "relevances": [],
                }
            )
            continue
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        retrieved_pages = [n.metadata.get("page_start", 0) for n in nodes]
        gt = item["ground_truth_pages"]
        rels = compute_relevances(retrieved_pages, gt)
        total_relevant = _count_total_relevant(gt)

        per_question.append(
            {
                "id": item["id"],
                "relevances": rels,
                "retrieved_pages": retrieved_pages,
                "total_relevant": total_relevant,
                "latency_s": round(elapsed, 3),
            }
        )

    # Агрегация
    valid = [q for q in per_question if "error" not in q]
    n = len(valid) or 1

    return {
        "retriever": config.name,
        "description": config.description,
        "category": config.category,
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


def print_results(results: list[dict], top_k: int) -> None:
    """
    Вывести сводную таблицу результатов, сгруппированную по категориям.

    Parameters
    ----------
    results : list[dict]
        Результаты оценки от ``evaluate_retriever()``.
    top_k : int
        Значение top_k (для заголовка таблицы).
    """
    table = Table(
        title=f"Результаты эксперимента (top_k={top_k})",
        show_lines=True,
    )
    table.add_column("Retriever", style="cyan", min_width=20)
    table.add_column("Hit Rate", style="green", justify="right")
    table.add_column("MRR", style="green", justify="right")
    table.add_column("P@k", style="green", justify="right")
    table.add_column("R@k", style="green", justify="right")
    table.add_column("NDCG@k", style="green", justify="right")
    table.add_column("Latency (s)", style="yellow", justify="right")
    table.add_column("Errors", style="red", justify="right")

    category_labels = {
        "sparse": "SPARSE",
        "dense": "DENSE",
        "hybrid": "HYBRID",
        "rerank": "BEST + RERANK",
    }

    current_category = None
    for r in results:
        if r["category"] != current_category:
            current_category = r["category"]
            label = category_labels.get(current_category, current_category.upper())
            table.add_row(f"[bold]{label}[/bold]", *[""] * 7)

        table.add_row(
            f"  {r['retriever']}",
            f"{r['hit_rate']:.3f}",
            f"{r['mrr']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['ndcg']:.3f}",
            f"{r['avg_latency_s']:.3f}",
            str(r["errors"]),
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def main(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Количество результатов"),
    models: str = typer.Option(
        "",
        "--models",
        "-m",
        help="Модели эмбеддингов через запятую (например minilm,e5). По умолчанию все.",
    ),
    rerank: bool = typer.Option(
        False, "--rerank", help="Применить ре-ранкинг к лучшему ретриверу"
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Запустить оценку всех ретриверов на eval-датасете.

    Parameters
    ----------
    top_k : int
        Количество результатов поиска.
    models : str
        Модели эмбеддингов через запятую.
    rerank : bool
        Применить ре-ранкинг к лучшему ретриверу.
    save : bool
        Сохранить детальные результаты в JSON.
    """
    # Выбор моделей
    if models:
        requested = [m.strip() for m in models.split(",")]
        unknown = set(requested) - set(EMBEDDING_MODELS)
        if unknown:
            console.print(
                f"[red]Неизвестные модели: {unknown}. "
                f"Доступные: {list(EMBEDDING_MODELS)}[/red]"
            )
            raise typer.Exit(1)
        selected = {k: EMBEDDING_MODELS[k] for k in requested}
    else:
        selected = EMBEDDING_MODELS

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    # Загрузка индексов
    console.print("[dim]Загрузка индексов из Qdrant...[/dim]")
    indexes = _load_indexes(selected)

    # Парсинг узлов (для sparse ретриверов)
    console.print("[dim]Парсинг документов в узлы (для BM25/TF-IDF)...[/dim]")
    documents = load_documents()
    nodes = parse_nodes(documents)
    console.print(f"[green]Получено {len(nodes)} узлов[/green]")

    dataset = load_dataset()
    console.print(f"[green]Загружено {len(dataset)} вопросов из eval-датасета[/green]")

    # Построение ретриверов
    configs = build_all_retrievers(indexes, nodes, selected)
    console.print(f"[green]Подготовлено {len(configs)} ретриверов[/green]\n")

    # Фаза 1: оценка всех ретриверов
    results: list[dict] = []
    for config in configs:
        console.print(f"[bold]▶ {config.name}[/bold] — {config.description}")
        result = evaluate_retriever(config, dataset, top_k)
        results.append(result)
        console.print(
            f"  Hit Rate={result['hit_rate']:.3f}  "
            f"MRR={result['mrr']:.3f}  "
            f"NDCG={result['ndcg']:.3f}  "
            f"({result['avg_latency_s']:.3f}s/query)\n"
        )

    # Фаза 2: ре-ранкинг лучшего (по NDCG)
    if rerank:
        best_result = max(results, key=lambda r: r["ndcg"])
        best_config = next(c for c in configs if c.name == best_result["retriever"])

        console.print(
            f"[bold yellow]▶ Rerank лучшего: {best_config.name}[/bold yellow]"
        )
        reranked = apply_rerank(best_config)
        rerank_result = evaluate_retriever(reranked, dataset, top_k)
        results.append(rerank_result)
        console.print(
            f"  Hit Rate={rerank_result['hit_rate']:.3f}  "
            f"MRR={rerank_result['mrr']:.3f}  "
            f"NDCG={rerank_result['ndcg']:.3f}  "
            f"({rerank_result['avg_latency_s']:.3f}s/query)\n"
        )

    # Сортировка по категории для таблицы
    category_order = {"sparse": 0, "dense": 1, "hybrid": 2, "rerank": 3}
    results.sort(key=lambda r: category_order.get(r["category"], 99))

    print_results(results, top_k)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"eval_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Результаты сохранены: {out_path}[/green]")


if __name__ == "__main__":
    app()
