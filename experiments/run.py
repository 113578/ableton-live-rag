"""
Единая точка входа для экспериментов.

Поддерживает оценку ретриверов, ранжировщиков, генераторов и сквозного RAG-пайплайна,
а также поиск оптимальных параметров чанкинга.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.
Для команды ``chunking`` с vector/hybrid-ретривером: ``uv run scripts/build_chunking_indexes.py``.

Примеры:
    uv run experiments/run.py retriever --top-k 5
    uv run experiments/run.py chunking --chunk-size 512 --overlap 64 --retriever vector/bm25
    uv run experiments/run.py reranker --top-k 5 --retriever hybrid_rrf/e5
    uv run experiments/run.py generator --top-k 5 --retriever hybrid_rrf/e5 --reranker minilm-l6
    uv run experiments/run.py end2end --top-k 5 --retriever hybrid_rrf/e5 --reranker minilm-l6
"""

import asyncio

import typer
from llama_index.core import Settings as LlamaSettings
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS
from ableton_live_rag.ingest import load_documents
from ableton_live_rag.config import settings
from experiments.components import (
    RerankerConfig,
    build_generators,
    build_rerankers,
    build_retrievers,
)
from experiments.utils import (
    GENERATOR_META_KEYS,
    RESULTS_DIR,
    TEST_DATASET_PATH,
    aggregate_retrieval_metrics,
    build_judge_metrics,
    build_retrieval_pipeline,
    col_header,
    console,
    evaluate_dataset,
    evaluate_generator,
    find_by_name,
    format_generator_summary,
    format_retrieval_summary,
    load_dataset,
    load_indexes,
    load_indexes_for_chunking,
    parse_nodes_with_config,
    prepare_experiment,
    save_results,
)

app = typer.Typer(no_args_is_help=True)

_RERANKER_MULTIPLIERS = [2, 3, 5]


def _load_active(dataset_path=None) -> tuple[list, list, list]:
    active_cfg = EMBEDDING_MODELS[settings.active_embedding_model]
    active_embedding_configs = {settings.active_embedding_model: active_cfg}

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    console.print("[dim]Загрузка документов...[/dim]")
    documents = load_documents()
    nodes = parse_nodes_with_config(
        documents, settings.chunk_size, settings.chunk_overlap
    )

    console.print("[dim]Загрузка индекса из Qdrant...[/dim]")
    indexes = load_indexes(active_embedding_configs)

    dataset = load_dataset(dataset_path) if dataset_path else load_dataset()
    console.print(f"[green]Загружено {len(dataset)} вопросов[/green]")

    retriever_configs = build_retrievers(
        indexes=indexes, nodes=nodes, embedding_configs=active_embedding_configs
    )

    return retriever_configs, nodes, dataset


def _print_retriever_table(results: list[dict], top_k: int) -> None:
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


def _print_reranker_table(results: list[dict], top_k: int) -> None:
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
            f"  ×{r['multiplier']}",
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


def _print_generator_table(results: list[dict], title: str) -> None:
    score_keys = (
        [k for k in results[0] if k not in GENERATOR_META_KEYS] if results else []
    )

    table = Table(title=title, show_lines=True)
    table.add_column("Generator", style="cyan", min_width=18)

    for key in score_keys:
        table.add_column(col_header(key), style="green", justify="right")

    table.add_column("Latency (s)", style="yellow", justify="right")
    table.add_column("Errors", style="red", justify="right")

    for r in results:
        table.add_row(
            r["generator"],
            *[f"{r.get(k, 0.0):.3f}" for k in score_keys],
            f"{r['avg_latency_s']:.3f}",
            str(r["errors"]),
        )

    console.print()
    console.print(table)
    console.print()


@app.command()
def retriever(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Количество результатов"),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Оценка ретриверов на валидационном наборе данных.
    """

    indexes, nodes, dataset = prepare_experiment()

    configs = build_retrievers(
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
            **aggregate_retrieval_metrics(per_question, total_time),
        }
        results.append(result)
        console.print(format_retrieval_summary(result) + "\n")

    category_order = {"sparse": 0, "dense": 1, "hybrid": 2}
    results.sort(key=lambda r: category_order.get(r["category"], 99))

    _print_retriever_table(results, top_k)

    if save:
        save_results(results, RESULTS_DIR / "retriever")


def _evaluate_reranker(
    base_retriever,
    reranker: RerankerConfig | None,
    dataset: list[dict],
    top_k: int,
    candidate_k: int,
) -> dict:
    retrieve_fn = build_retrieval_pipeline(
        retriever=base_retriever,
        reranker=reranker,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    per_question, total_time = evaluate_dataset(retrieve_fn, dataset)
    metrics = aggregate_retrieval_metrics(per_question, total_time)

    name = reranker.name if reranker is not None else "baseline"
    description = reranker.description if reranker is not None else "No reranking"

    return {
        "reranker": name,
        "description": description,
        "base_retriever": base_retriever.name,
        "candidate_k": candidate_k,
        "multiplier": candidate_k // top_k,
        **metrics,
    }


@app.command()
def reranker(
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество финальных результатов"
    ),
    retriever_name: str = typer.Option(
        "vector/bge",
        "--retriever",
        "-r",
        help="Ретривер",
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Оценка ранжировщиков поверх базового ретривера.
    """

    retriever_configs, _, dataset = _load_active()
    base_retriever = find_by_name(retriever_configs, retriever_name, "Ретривер")
    console.print(f"[green]Базовый ретривер: {base_retriever.name}[/green]")

    console.print("[dim]Загрузка моделей ранжировщиков...[/dim]")
    reranker_configs = build_rerankers()
    console.print(f"[green]Подготовлено {len(reranker_configs)} ранжировщиков[/green]")

    results: list[dict] = []

    console.print("\n[bold]▶ baseline[/bold] — no reranking (pool=top_k)")
    results.append(
        _evaluate_reranker(base_retriever, None, dataset, top_k, candidate_k=top_k)
    )
    console.print(format_retrieval_summary(results[-1]) + "\n")

    for rerank_cfg in reranker_configs:
        for mult in _RERANKER_MULTIPLIERS:
            candidate_k = top_k * mult
            label = f"{rerank_cfg.name} (pool={candidate_k})"
            console.print(f"[bold]▶ {label}[/bold] — {rerank_cfg.description}")

            result = _evaluate_reranker(
                base_retriever, rerank_cfg, dataset, top_k, candidate_k
            )
            results.append(result)
            console.print(format_retrieval_summary(result) + "\n")

    _print_reranker_table(results, top_k)

    if save:
        save_results(results, RESULTS_DIR / "reranker")


@app.command()
def generator(
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество фрагментов контекста"
    ),
    retriever_name: str = typer.Option(
        "vector/bge",
        "--retriever",
        "-r",
        help="Базовый ретривер (например vector/bge, hybrid_rrf/bge, bm25)",
    ),
    reranker_name: str | None = typer.Option(
        None, "--reranker", help="Имя ранжировщика (опционально)"
    ),
    candidate_k: int = typer.Option(
        15, "--candidate-k", help="Размер пула кандидатов для ранжировщика"
    ),
    concurrency: int = typer.Option(
        16, "--concurrency", "-c", help="Число параллельных запросов"
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Оценка генераторов поверх фиксированного ретривера.
    """

    retriever_configs, _, dataset = _load_active()
    base_retriever = find_by_name(retriever_configs, retriever_name, "Ретривер")
    console.print(f"[green]Базовый ретривер: {base_retriever.name}[/green]")

    selected_reranker: RerankerConfig | None = None

    if reranker_name is not None:
        reranker_configs = build_rerankers()
        selected_reranker = find_by_name(reranker_configs, reranker_name, "Ранжировщик")

    console.print("[dim]Загрузка моделей генераторов...[/dim]")
    generator_configs = build_generators()
    console.print(f"[green]Подготовлено {len(generator_configs)} генераторов[/green]")

    metrics = build_judge_metrics()

    retrieve_fn = build_retrieval_pipeline(
        retriever=base_retriever,
        reranker=selected_reranker,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    async def _run() -> list[dict]:
        results: list[dict] = []

        for gen in generator_configs:
            console.print(f"\n[bold]▶ {gen.name}[/bold] — {gen.description}")
            result = await evaluate_generator(
                generator=gen,
                retrieve_fn=retrieve_fn,
                metrics=metrics,
                dataset=dataset,
                concurrency=concurrency,
            )
            results.append(result)
            console.print(format_generator_summary(result))

        return results

    results = asyncio.run(_run())
    _print_generator_table(results, title="Результаты эксперимента")

    if save:
        save_results(results, RESULTS_DIR / "generator")


@app.command()
def end2end(
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество фрагментов контекста"
    ),
    retriever_name: str = typer.Option(
        "vector/bge", "--retriever", "-r", help="Ретривер"
    ),
    reranker_name: str | None = typer.Option(None, "--reranker", help="Ранжировщик"),
    candidate_k: int = typer.Option(
        15, "--candidate-k", help="Размер пула кандидатов для ранжировщика"
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", "-c", help="Число параллельных запросов"
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Сквозная оценка всех компонентов.
    """

    retriever_configs, _, dataset = _load_active(TEST_DATASET_PATH)

    retriever_results: list[dict] = []

    for config in retriever_configs:
        console.print(f"[bold]▶ {config.name}[/bold] — {config.description}")
        per_q, total_time = evaluate_dataset(
            retrieve_fn=lambda q, c=config: c.retrieve(query=q, top_k=top_k),
            dataset=dataset,
        )
        result = {
            "retriever": config.name,
            "description": config.description,
            "category": config.category,
            **aggregate_retrieval_metrics(per_q, total_time),
        }
        retriever_results.append(result)
        console.print(format_retrieval_summary(result) + "\n")

    category_order = {"sparse": 0, "dense": 1, "hybrid": 2}
    retriever_results.sort(key=lambda r: category_order.get(r["category"], 99))
    _print_retriever_table(retriever_results, top_k)

    base_retriever = find_by_name(retriever_configs, retriever_name, "Ретривер")
    console.print(
        f"[green]Базовый ретривер для фаз 2–3: {base_retriever.name}[/green]\n"
    )

    reranker_cfgs = build_rerankers()
    reranker_results: list[dict] = []

    console.print("[bold]▶ baseline[/bold] — no reranking (pool=top_k)")
    reranker_results.append(
        _evaluate_reranker(base_retriever, None, dataset, top_k, top_k)
    )
    console.print(format_retrieval_summary(reranker_results[-1]) + "\n")

    for rerank_cfg in reranker_cfgs:
        for mult in _RERANKER_MULTIPLIERS:
            ck = top_k * mult
            console.print(
                f"[bold]▶ {rerank_cfg.name} (pool={ck})[/bold] — {rerank_cfg.description}"
            )
            reranker_results.append(
                _evaluate_reranker(base_retriever, rerank_cfg, dataset, top_k, ck)
            )
            console.print(format_retrieval_summary(reranker_results[-1]) + "\n")

    _print_reranker_table(reranker_results, top_k)

    selected_reranker: RerankerConfig | None = None
    if reranker_name is not None:
        selected_reranker = find_by_name(reranker_cfgs, reranker_name, "Ранжировщик")

    pipeline_label = (
        f"{base_retriever.name} → {selected_reranker.name}"
        if selected_reranker is not None
        else base_retriever.name
    )
    console.print(f"[green]Пайплайн поиска: {pipeline_label}[/green]")

    console.print("[dim]Загрузка моделей генераторов...[/dim]")
    generator_configs = build_generators()
    console.print(f"[green]Подготовлено {len(generator_configs)} генераторов[/green]")

    metrics = build_judge_metrics()

    retrieve_fn = build_retrieval_pipeline(
        retriever=base_retriever,
        reranker=selected_reranker,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    async def _run() -> list[dict]:
        results: list[dict] = []

        for gen in generator_configs:
            console.print(f"\n[bold]▶ {gen.name}[/bold] — {gen.description}")
            result = await evaluate_generator(
                generator=gen,
                retrieve_fn=retrieve_fn,
                metrics=metrics,
                dataset=dataset,
                concurrency=concurrency,
            )
            result["pipeline"] = pipeline_label
            results.append(result)
            console.print(format_generator_summary(result))

        return results

    generator_results = asyncio.run(_run())
    _print_generator_table(generator_results, title=f"End-to-end: {pipeline_label}")

    if save:
        save_results(
            {
                "retriever": retriever_results,
                "reranker": reranker_results,
                "generator": generator_results,
            },
            RESULTS_DIR / "end2end",
        )


def _print_chunking_table(results: list[dict], top_k: int, retriever: str) -> None:
    table = Table(
        title=f"Чанкинг: {retriever} (top_k={top_k})",
        show_lines=True,
    )

    table.add_column("chunk_size", style="cyan", justify="right")
    table.add_column("overlap", style="magenta", justify="right")
    table.add_column("nodes", style="dim", justify="right")
    table.add_column("Hit Rate", style="green", justify="right")
    table.add_column("MRR", style="green", justify="right")
    table.add_column("P@k", style="green", justify="right")
    table.add_column("R@k", style="green", justify="right")
    table.add_column("NDCG@k", style="green", justify="right")
    table.add_column("Latency (s)", style="yellow", justify="right")
    table.add_column("Errors", style="red", justify="right")

    for r in results:
        table.add_row(
            str(r["chunk_size"]),
            str(r["overlap"]),
            str(r["nodes_count"]),
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
def chunking(
    chunk_sizes: list[int] = typer.Option(
        [128, 256, 512, 1024],
        "--chunk-size",
        "-c",
        help="Размеры чанков в токенах",
    ),
    overlaps: list[int] = typer.Option(
        [16, 32, 64, 128],
        "--overlap",
        "-o",
        help="Перекрытия чанков в токенах",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Количество результатов"),
    retriever_name: str = typer.Option(
        "bm25",
        "--retriever",
        "-r",
        help=(
            "Ретривер для оценки (bm25, tfidf — без pre-built индексов; "
            "vector/bge, hybrid_rrf/e5 и т.д. — запустите build_chunking_indexes.py)"
        ),
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Поиск оптимальных параметров чанкинга (chunk_size × overlap).
    """

    is_sparse = retriever_name in ("bm25", "tfidf")

    console.print("[dim]Загрузка документов...[/dim]")
    documents = load_documents()
    dataset = load_dataset()

    active_cfg = EMBEDDING_MODELS[settings.active_embedding_model]
    active_embedding_configs = {settings.active_embedding_model: active_cfg}

    pairs = [(cs, ov) for cs in chunk_sizes for ov in overlaps]
    console.print(
        f"[bold]Проверка {len(pairs)} конфигураций чанкинга "
        f"(ретривер: {retriever_name})[/bold]\n"
    )

    results: list[dict] = []

    for chunk_size, overlap in pairs:
        label = f"cs={chunk_size}, co={overlap}"
        console.print(f"[bold cyan]▶ {label}[/bold cyan]")

        nodes = parse_nodes_with_config(documents, chunk_size, overlap)
        console.print(f"  Узлов: {len(nodes)}")

        if is_sparse:
            indexes: dict = {}
        else:
            try:
                indexes = load_indexes_for_chunking(chunk_size, overlap, active_cfg)
            except RuntimeError as e:
                console.print(f"[red]  Индекс не найден: {e}[/red]")
                console.print(
                    "[yellow]  Запустите: uv run scripts/build_chunking_indexes.py[/yellow]"
                )
                continue

        LlamaSettings.chunk_size = chunk_size
        LlamaSettings.chunk_overlap = overlap

        retriever_configs = build_retrievers(
            indexes=indexes, nodes=nodes, embedding_configs=active_embedding_configs
        )
        config = find_by_name(retriever_configs, retriever_name, "Ретривер")

        per_question, total_time = evaluate_dataset(
            retrieve_fn=lambda q, c=config: c.retrieve(query=q, top_k=top_k),
            dataset=dataset,
        )
        result = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "nodes_count": len(nodes),
            "retriever": retriever_name,
            **aggregate_retrieval_metrics(per_question, total_time),
        }
        results.append(result)
        console.print(format_retrieval_summary(result) + "\n")

    if results:
        _print_chunking_table(results, top_k, retriever_name)

    if save and results:
        save_results(results, RESULTS_DIR / "chunking")


if __name__ == "__main__":
    app()
