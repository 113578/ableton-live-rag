"""
Запуск экспериментов по оценке генераторов.

Перед запуском создайте индексы: ``uv run scripts/build_eval_indexes.py``.
"""

import asyncio
import time
from pathlib import Path

import typer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from rich.table import Table

from ableton_live_rag.config import EMBEDDING_MODELS
from experiments.generator.generators import (
    GeneratorConfig,
    JUDGE_SPEC,
    build_all_generators,
    make_llm,
)
from experiments.generator.helpers import (
    LlamaIndexJudge,
    ameasure,
    build_metrics,
)
from experiments.retriever.retrievers import build_all_retrievers
from experiments.utils import console, prepare_experiment, save_results

app = typer.Typer(no_args_is_help=False)

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "generator"

_META = {"generator", "description", "avg_latency_s", "errors", "details"}


async def evaluate_generator(
    generator: GeneratorConfig,
    retrieve_fn,  # noqa: ANN001
    metrics: dict[str, BaseMetric],
    dataset: list[dict],
    top_k: int,
    concurrency: int = 16,
) -> dict:
    """
    Оценка генератора на всём наборе данных.

    Parameters
    ----------
    generator : GeneratorConfig
        Конфигурация генератора.
    retrieve_fn : Callable[[str, int], list[NodeWithScore]]
        Функция поиска контекста.
    metrics : dict[str, BaseMetric]
        Метрики.
    dataset : list[dict]
        Валидационный набор данных.
    top_k : int
        Количество фрагментов контекста.
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
                nodes = retrieve_fn(question, top_k)
                contexts = [n.node.get_content() for n in nodes]

                t0 = time.perf_counter()
                answer = await generator.agenerate(question=question, contexts=contexts)
                gen_time = time.perf_counter() - t0

                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=contexts,
                )
                scores = await ameasure(test_case=test_case, metrics=metrics)

                return {"id": item["id"], "latency_s": round(gen_time, 3), **scores}

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


def _col_header(key: str) -> str:
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


def format_result_summary(result: dict) -> str:
    """
    Однострочная сводка результатов по генератору.

    Parameters
    ----------
    result : dict
        Результат.

    Returns
    -------
    str
        Строка вида ``Metric1=0.xxx  Metric2=0.xxx  (0.xxxs/query)``.
    """

    score_keys = [k for k in result if k not in _META]
    parts = [f"{_col_header(k)}={result[k]:.3f}" for k in score_keys]

    return "  " + "  ".join(parts) + f"  ({result['avg_latency_s']:.3f}s/query)"


def print_results(results: list[dict]) -> None:
    """
    Вывод сводной таблицы результатов в терминал.

    Колонки метрик формируются динамически из ключей первого результата.

    Parameters
    ----------
    results : list[dict]
        Результаты.
    """

    score_keys = [k for k in results[0] if k not in _META] if results else []

    table = Table(title="Результаты эксперимента", show_lines=True)
    table.add_column("Generator", style="cyan", min_width=18)

    for key in score_keys:
        table.add_column(_col_header(key), style="green", justify="right")

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
def main(
    top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество фрагментов контекста"
    ),
    retriever_name: str = typer.Option(
        "hybrid_rrf/e5",
        "--retriever",
        "-r",
        help="Базовый ретривер (например hybrid_rrf/e5, vector/bge, bm25)",
    ),
    concurrency: int = typer.Option(
        16, "--concurrency", "-c", help="Число параллельных запросов"
    ),
    save: bool = typer.Option(
        False, "--save", help="Сохранить детальные результаты в JSON"
    ),
) -> None:
    """
    Запуск экспериментов по оценке генераторов.

    Parameters
    ----------
    top_k : int
        Количество фрагментов контекста, передаваемых генератору.
    retriever_name : str
        Имя базового компонента поиска из доступных конфигураций.
    concurrency : int
        Число вопросов, обрабатываемых параллельно.
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

    console.print("[dim]Загрузка моделей генераторов...[/dim]")
    generator_configs = build_all_generators()
    console.print(f"[green]Подготовлено {len(generator_configs)} генераторов[/green]")

    console.print(
        f"[dim]Инициализация судьи DeepEval ({JUDGE_SPEC.backend}/{JUDGE_SPEC.model_id})...[/dim]"
    )
    judge = LlamaIndexJudge(llm=make_llm(JUDGE_SPEC), name=JUDGE_SPEC.name)
    metrics = build_metrics(judge=judge)

    async def _run() -> list[dict]:
        results: list[dict] = []

        for generator in generator_configs:
            console.print(
                f"\n[bold]▶ {generator.name}[/bold] — {generator.description}"
            )
            result = await evaluate_generator(
                generator=generator,
                retrieve_fn=base_config.retrieve,
                metrics=metrics,
                dataset=dataset,
                top_k=top_k,
                concurrency=concurrency,
            )
            results.append(result)
            console.print(format_result_summary(result))

        return results

    results = asyncio.run(_run())
    print_results(results)

    if save:
        save_results(results, _RESULTS_DIR)


if __name__ == "__main__":
    app()
