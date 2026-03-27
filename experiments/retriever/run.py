"""
Запуск экспериментов по оценке компонент поиска.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.
"""

import json
import time
from pathlib import Path

import typer
from llama_index.core import Settings as LlamaSettings
from rich.console import Console
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS, settings
from ableton_live_rag.index import parse_nodes
from ableton_live_rag.ingest import load_documents
from experiments.metrics import (
    compute_relevances,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from experiments.retriever.retrievers import (
    RetrieverConfig,
    build_all_retrievers,
)
from experiments.utils import count_total_relevant, load_dataset, load_indexes

app = typer.Typer(no_args_is_help=False)
console = Console()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "retriever"


def evaluate_retriever(
    config: RetrieverConfig,
    dataset: list[dict],
    top_k: int,
) -> dict:
    """
    Оценка компонента поиска на всём наборе данных.

    Parameters
    ----------
    config : RetrieverConfig
        Конфигурация оцениваемого компонента поиска.
    dataset : list[dict]
        Валидационный набор данных.
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
            nodes = config.retrieve(query=item["question"], top_k=top_k)
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

        rels = compute_relevances(
            retrieved_pages=retrieved_pages, ground_truth_ranges=gt
        )
        total_relevant = count_total_relevant(ground_truth_ranges=gt)

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
    Вывод сводной таблицы результатов, сгруппированной по категориям.

    Parameters
    ----------
    results : list[dict]
        Результаты оценки от ``evaluate_retriever()``.
    top_k : int
        Значение top_k.
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
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Запуск экспериментов по оценке компонент поиска на валидационном наборе данных.

    Parameters
    ----------
    top_k : int
        Количество результатов поиска.
    save : bool
        Сохранить детальные результаты в JSON.
    """

    selected = EMBEDDING_MODELS

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    # Загрузка индексов
    console.print("[dim]Загрузка индексов из Qdrant...[/dim]")
    indexes = load_indexes(models=selected)

    # Парсинг узлов (для sparse ретриверов)
    console.print("[dim]Парсинг документов в узлы (для BM25/TF-IDF)...[/dim]")
    documents = load_documents()
    nodes = parse_nodes(documents=documents)
    console.print(f"[green]Получено {len(nodes)} узлов[/green]")

    dataset = load_dataset()
    console.print(f"[green]Загружено {len(dataset)} вопросов из eval-датасета[/green]")

    # Построение ретриверов
    configs = build_all_retrievers(
        indexes=indexes, nodes=nodes, embedding_configs=selected
    )
    console.print(f"[green]Подготовлено {len(configs)} ретриверов[/green]\n")

    # Оценка всех компонент поиска
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

    # Сортировка по категории для таблицы
    category_order = {"sparse": 0, "dense": 1, "hybrid": 2}
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
