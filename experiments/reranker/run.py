"""
Запуск экспериментов по оценке компонент ранжирования.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.
"""

from pathlib import Path

import typer
from llama_index.core.schema import NodeWithScore
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS
from experiments.reranker.rerankers import (
    RerankerConfig,
    build_all_rerankers,
)
from experiments.retriever.retrievers import RetrieverConfig, build_all_retrievers
from experiments.utils import (
    aggregate_metrics,
    console,
    evaluate_dataset,
    format_result_summary,
    prepare_experiment,
    save_results,
)

app = typer.Typer(no_args_is_help=False)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "reranker"


def evaluate_reranker(
    base_config: RetrieverConfig,
    reranker: RerankerConfig | None,
    dataset: list[dict],
    top_k: int,
    candidate_k: int,
) -> dict:
    """
    Оценка ранжировщика поверх базового ретривера.

    Parameters
    ----------
    base_config : RetrieverConfig
        Конфигурация базового ретривера.
    reranker : RerankerConfig or None
        Конфигурация ранжировщика. ``None`` — без переранжирования (baseline).
    dataset : list[dict]
        Валидационный набор данных.
    top_k : int
        Итоговое количество результатов.
    candidate_k : int
        Количество кандидатов, извлекаемых базовым ретривером.

    Returns
    -------
    dict
        Словарь с агрегированными метриками и ``details``.
    """

    def _retrieve(query: str) -> list[NodeWithScore]:
        candidates = base_config.retrieve(query=query, top_k=candidate_k)
        if reranker is not None:
            return reranker.rerank(query=query, nodes=candidates, top_k=top_k)
        return candidates[:top_k]

    per_question, total_time = evaluate_dataset(_retrieve, dataset)
    metrics = aggregate_metrics(per_question, total_time)

    reranker_name = reranker.name if reranker is not None else "baseline"
    reranker_desc = reranker.description if reranker is not None else "No reranking"

    return {
        "reranker": reranker_name,
        "description": reranker_desc,
        "base_retriever": base_config.name,
        "candidate_k": candidate_k,
        "multiplier": candidate_k // top_k,
        **metrics,
    }


def print_results(results: list[dict], top_k: int) -> None:
    """
    Вывод сводной таблицы результатов, сгруппированной по ранжировщикам.

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

    table.add_column("Reranker", style="cyan", min_width=18)
    table.add_column("Pool", style="magenta", justify="right")
    table.add_column("Hit Rate", style="green", justify="right")
    table.add_column("MRR", style="green", justify="right")
    table.add_column("P@k", style="green", justify="right")
    table.add_column("R@k", style="green", justify="right")
    table.add_column("NDCG@k", style="green", justify="right")
    table.add_column("Latency (s)", style="yellow", justify="right")
    table.add_column("Errors", style="red", justify="right")

    current_reranker = None

    for r in results:
        if r["reranker"] != current_reranker:
            current_reranker = r["reranker"]
            table.add_row(f"[bold]{current_reranker.upper()}[/bold]", *[""] * 8)

        table.add_row(
            f"  \u00d7{r['multiplier']}",
            str(r["candidate_k"]),
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


MULTIPLIERS = [2, 3, 5]


@app.command()
def main(
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество финальных результатов"
    ),
    retriever_name: str = typer.Option(
        "hybrid_rrf/e5",
        "--retriever",
        "-r",
        help="Базовый ретривер (например hybrid_rrf/e5, vector/bge, bm25)",
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Запуск экспериментов по оценке ранжировщиков.

    Parameters
    ----------
    top_k : int
        Итоговое количество результатов после ранжирования.
    retriever_name : str
        Имя базового ретривера.
    save : bool
        Сохранить детальные результаты в JSON.
    """

    indexes, nodes, dataset = prepare_experiment()

    all_retriever_configs = build_all_retrievers(
        indexes=indexes, nodes=nodes, embedding_configs=EMBEDDING_MODELS
    )
    base_configs = [c for c in all_retriever_configs if c.name == retriever_name]

    if not base_configs:
        available = [c.name for c in all_retriever_configs]
        console.print(
            f"[red]Ретривер {retriever_name!r} не найден. Доступные: {available}[/red]"
        )
        raise typer.Exit(1)

    base_config = base_configs[0]
    console.print(f"[green]Базовый ретривер: {base_config.name}[/green]")

    console.print("[dim]Загрузка моделей ранжировщиков...[/dim]")
    reranker_configs = build_all_rerankers()
    console.print(f"[green]Подготовлено {len(reranker_configs)} ранжировщиков[/green]")

    results: list[dict] = []

    console.print("\n[bold]▶ baseline[/bold] — no reranking (pool=top_k)")
    result = evaluate_reranker(
        base_config=base_config,
        reranker=None,
        dataset=dataset,
        top_k=top_k,
        candidate_k=top_k,
    )
    results.append(result)
    console.print(format_result_summary(result) + "\n")

    for reranker in reranker_configs:
        for mult in MULTIPLIERS:
            candidate_k = top_k * mult
            label = f"{reranker.name} (pool={candidate_k})"
            console.print(f"[bold]▶ {label}[/bold] — {reranker.description}")

            result = evaluate_reranker(
                base_config=base_config,
                reranker=reranker,
                dataset=dataset,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            results.append(result)
            console.print(format_result_summary(result) + "\n")

    print_results(results, top_k)

    if save:
        save_results(results, RESULTS_DIR)


if __name__ == "__main__":
    app()
