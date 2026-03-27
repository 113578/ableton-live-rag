"""
Запуск экспериментов по оценке компонент ранжирования.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.

Примеры::

    uv run experiments/reranker/run.py --retriever hybrid_rrf/e5
    uv run experiments/reranker/run.py --retriever vector/bge --multipliers 2,3,5 --save
    uv run experiments/reranker/run.py --rerankers minilm-l6,bge
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
from experiments.retriever.retrievers import RetrieverConfig, build_all_retrievers
from experiments.reranker.rerankers import (
    RERANKER_MODELS,
    RerankerConfig,
    build_all_rerankers,
)
from experiments.utils import count_total_relevant, load_dataset, load_indexes

app = typer.Typer(no_args_is_help=False)
console = Console()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "reranker"


def evaluate_reranker(
    base_config: RetrieverConfig,
    reranker: RerankerConfig | None,
    dataset: list[dict],
    top_k: int,
    candidate_k: int,
) -> dict:
    """
    Оценка ранжировщика поверх базового ретривера на всём наборе данных.

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
        Словарь с агрегированными метриками и ``details`` по каждому вопросу.
    """

    per_question: list[dict] = []
    total_time = 0.0

    for item in dataset:
        t0 = time.perf_counter()

        try:
            candidates = base_config.retrieve(query=item["question"], top_k=candidate_k)

            if reranker is not None:
                nodes = reranker.rerank(
                    query=item["question"], nodes=candidates, top_k=top_k
                )
            else:
                nodes = candidates[:top_k]
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

    reranker_name = reranker.name if reranker is not None else "baseline"
    reranker_desc = reranker.description if reranker is not None else "No reranking"

    return {
        "reranker": reranker_name,
        "description": reranker_desc,
        "base_retriever": base_config.name,
        "candidate_k": candidate_k,
        "multiplier": candidate_k // top_k,
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
    Вывод сводной таблицы результатов, сгруппированной по ранжировщикам.

    Parameters
    ----------
    results : list[dict]
        Результаты оценки от ``evaluate_reranker()``.
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
    rerankers_str: str = typer.Option(
        "",
        "--rerankers",
        help=(
            f"Ранжировщики через запятую. По умолчанию все. "
            f"Доступные: {list(RERANKER_MODELS)}"
        ),
    ),
    multipliers_str: str = typer.Option(
        "2,3,5",
        "--multipliers",
        help="Множители пула кандидатов через запятую (например 2,3,5)",
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
    rerankers_str : str
        Ранжировщики через запятую.
    multipliers_str : str
        Множители пула кандидатов через запятую.
    save : bool
        Сохранить детальные результаты в JSON.
    """

    # Разбор параметров
    try:
        multipliers = [int(m.strip()) for m in multipliers_str.split(",")]
    except ValueError:
        console.print(f"[red]Неверный формат множителей: {multipliers_str!r}[/red]")
        raise typer.Exit(1)

    selected_models = EMBEDDING_MODELS

    # Выбор ранжировщиков
    if rerankers_str:
        requested_rerankers = [r.strip() for r in rerankers_str.split(",")]
        unknown_rerankers = set(requested_rerankers) - set(RERANKER_MODELS)

        if unknown_rerankers:
            console.print(
                f"[red]Неизвестные ранжировщики: {unknown_rerankers}. "
                f"Доступные: {list(RERANKER_MODELS)}[/red]"
            )
            raise typer.Exit(1)

        selected_rerankers: list[str] | None = requested_rerankers
    else:
        selected_rerankers = None  # все

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    # Загрузка индексов
    console.print("[dim]Загрузка индексов из Qdrant...[/dim]")
    indexes = load_indexes(models=selected_models)

    # Парсинг узлов (для sparse/hybrid ретриверов)
    console.print("[dim]Парсинг документов в узлы (для BM25/TF-IDF)...[/dim]")
    documents = load_documents()
    nodes = parse_nodes(documents=documents)
    console.print(f"[green]Получено {len(nodes)} узлов[/green]")

    dataset = load_dataset()
    console.print(f"[green]Загружено {len(dataset)} вопросов из eval-датасета[/green]")

    # Поиск базового ретривера
    all_retriever_configs = build_all_retrievers(
        indexes=indexes, nodes=nodes, embedding_configs=selected_models
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

    # Загрузка ранжировщиков
    console.print("[dim]Загрузка моделей ранжировщиков...[/dim]")
    reranker_configs = build_all_rerankers(selected=selected_rerankers)
    console.print(f"[green]Подготовлено {len(reranker_configs)} ранжировщиков[/green]")

    results: list[dict] = []

    # Baseline: базовый ретривер без переранжирования
    console.print("\n[bold]▶ baseline[/bold] — no reranking (pool=top_k)")
    result = evaluate_reranker(
        base_config=base_config,
        reranker=None,
        dataset=dataset,
        top_k=top_k,
        candidate_k=top_k,
    )
    results.append(result)
    console.print(
        f"  Hit Rate={result['hit_rate']:.3f}  "
        f"MRR={result['mrr']:.3f}  "
        f"NDCG={result['ndcg']:.3f}  "
        f"({result['avg_latency_s']:.3f}s/query)\n"
    )

    # Ранжировщики с разными размерами пула кандидатов
    for reranker in reranker_configs:
        for mult in multipliers:
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

            console.print(
                f"  Hit Rate={result['hit_rate']:.3f}  "
                f"MRR={result['mrr']:.3f}  "
                f"NDCG={result['ndcg']:.3f}  "
                f"({result['avg_latency_s']:.3f}s/query)\n"
            )

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
