"""
Запуск экспериментов по оценке компонент поиска.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.
"""

from pathlib import Path

import typer
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS
from experiments.retriever.retrievers import build_all_retrievers
from experiments.utils import (
    aggregate_metrics,
    console,
    evaluate_dataset,
    format_result_summary,
    prepare_experiment,
    save_results,
)

app = typer.Typer(no_args_is_help=False)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "retriever"


def print_results(results: list[dict], top_k: int) -> None:
    """
    Вывод сводной таблицы результатов, сгруппированной по категориям.

    Parameters
    ----------
    results : list[dict]
        Результаты оценки.
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

    category_labels = {"sparse": "SPARSE", "dense": "DENSE", "hybrid": "HYBRID"}
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

    indexes, nodes, dataset = prepare_experiment()

    configs = build_all_retrievers(
        indexes=indexes, nodes=nodes, embedding_configs=EMBEDDING_MODELS
    )
    console.print(f"[green]Подготовлено {len(configs)} ретриверов[/green]\n")

    results: list[dict] = []

    for config in configs:
        console.print(f"[bold]▶ {config.name}[/bold] — {config.description}")

        per_question, total_time = evaluate_dataset(
            retrieve_fn=lambda q, c=config: c.retrieve(query=q, top_k=top_k),
            dataset=dataset,
        )
        result = {
            "retriever": config.name,
            "description": config.description,
            "category": config.category,
            **aggregate_metrics(per_question, total_time),
        }
        results.append(result)
        console.print(format_result_summary(result) + "\n")

    category_order = {"sparse": 0, "dense": 1, "hybrid": 2}
    results.sort(key=lambda r: category_order.get(r["category"], 99))

    print_results(results, top_k)

    if save:
        save_results(results, RESULTS_DIR)


if __name__ == "__main__":
    app()
