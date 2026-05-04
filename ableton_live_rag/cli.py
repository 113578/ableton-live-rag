"""
Интерфейс командной строки RAG-системы.
"""

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ableton_live_rag.config import settings

app = typer.Typer(
    name="rag",
    help="RAG-система для документации экосистемы Ableton",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest() -> None:
    """
    Загрузка PDF, разбитие на чанки, векторизация и сохранение в Qdrant.
    """

    from ableton_live_rag import llm
    from ableton_live_rag.config import EMBEDDING_MODELS
    from ableton_live_rag.index import build_index
    from ableton_live_rag.ingest import load_documents

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Извлечение документов из PDF...", total=None)
        documents = load_documents(pdf_path=settings.corpus_path)

    console.print(f"[green]Извлечено {len(documents)} разделов из TOC[/green]")

    for cfg in EMBEDDING_MODELS.values():
        console.print(
            f"[dim]Индексирование → {cfg.collection_name} ({cfg.model_id})...[/dim]"
        )

        llm.setup_embedding(cfg)
        build_index(documents, collection_name=cfg.collection_name)

    console.print("[bold green]✓ Индексирование завершено![/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Запрос пользователя"),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", help="Количество фрагментов для контекста"
    ),
) -> None:
    """
    Задать вопрос и получить ответ от LLM с источниками из документации.

    Parameters
    ----------
    question : str
        Вопрос об Ableton Live на любом языке.
    top_k : int or None, optional
        Количество фрагментов для контекста. Если не указан,
        используется ``settings.similarity_top_k``.
    """

    from ableton_live_rag import llm
    from ableton_live_rag.config import settings
    from ableton_live_rag.query import ask as query_ask

    console.print("[dim]Инициализация...[/dim]")
    llm.setup()

    k = top_k or settings.similarity_top_k

    console.print()
    console.print(Panel(question, title="[blue]Вопрос[/blue]", border_style="blue"))
    console.print()

    async def _run() -> None:
        with console.status("[dim]Поиск в документации...[/dim]"):
            answer = await query_ask(question, top_k=k)

        console.print(Panel.fit("[green]Ответ[/green]", border_style="green"))

        async for token in answer.response_gen:
            console.print(token, end="")

        console.print("\n")

        if answer.source_nodes:
            table = Table(title="Источники", show_lines=True)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Глава / Раздел", style="white")
            table.add_column("Стр.", style="yellow", width=6)
            table.add_column("Score", style="green", width=7)

            for i, node in enumerate(answer.source_nodes, 1):
                chapter_section = node.chapter

                if node.section:
                    chapter_section += f" › {node.section}"

                table.add_row(
                    str(i),
                    chapter_section,
                    str(node.page_start),
                    f"{node.score:.3f}",
                )

            console.print(table)

    asyncio.run(_run())


@app.command()
def search(
    query: str = typer.Argument(..., help="Поисковый запрос"),
    similarity_top_k: int = typer.Option(
        5, "--top-k", "-k", help="Количество результатов"
    ),
) -> None:
    """
    Векторный поиск без генерации ответа.

    Parameters
    ----------
    query : str
        Поисковый запрос.
    similarity_top_k : int
        Количество результатов.
    """

    from ableton_live_rag import llm
    from ableton_live_rag.query import retrieve

    console.print("[dim]Инициализация эмбеддингов...[/dim]")
    llm.setup()

    with console.status("[dim]Поиск...[/dim]"):
        results = asyncio.run(retrieve(query, similarity_top_k=similarity_top_k))

    for i, r in enumerate(results, 1):
        meta = f"Глава: {r.chapter or '?'}"

        if r.section:
            meta += f" | Раздел: {r.section}"

        meta += f" | Стр. {r.page_start}"

        console.print(
            Panel(
                f"[dim]{meta}[/dim]\n\n{r.text[:600]}...",
                title=f"Результат {i}  (score: {r.score:.4f})",
                border_style="cyan",
            )
        )


@app.command()
def stats() -> None:
    """
    Показать статистику коллекции Qdrant.

    Выводит таблицу с количеством точек, векторов и статусом коллекции.
    """

    from ableton_live_rag.index import get_stats

    info = get_stats()

    table = Table(title="Qdrant — статистика коллекции")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="white")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def bot() -> None:
    """
    Запуск Telegram-бота.

    Бот подключается к запущенному FastAPI-серверу (``rag serve``)
    и отвечает на вопросы пользователей из Telegram.
    """

    from ableton_live_rag.bot.bot import run
    from ableton_live_rag.config import settings

    if not settings.telegram_bot_token:
        console.print("[red]Токен бота не задан.[/red]")
        raise typer.Exit(1)

    console.print("[bold green]Запуск Telegram-бота...[/bold green]")

    run(token=settings.telegram_bot_token, api_base_url=settings.api_base_url)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Адрес для привязки"),
    port: int = typer.Option(8000, "--port", "-p", help="Порт"),
) -> None:
    """
    Запуск FastAPI приложения.

    Parameters
    ----------
    host : str
        Адрес сервера.
    port : int
        Порт.
    """

    import uvicorn

    uvicorn.run("ableton_live_rag.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
