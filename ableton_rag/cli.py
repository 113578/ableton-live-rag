"""
Command-line interface for the RAG system.
"""

import asyncio

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ableton_rag.config import settings

app = typer.Typer(
    name="rag",
    help="RAG system for the Ableton ecosystem documentation",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest() -> None:
    """
    Load PDFs, split into chunks, vectorize and persist to Qdrant.
    """

    from ableton_rag import llm
    from ableton_rag.config import EMBEDDING_MODELS
    from ableton_rag.index import build_index
    from ableton_rag.ingest import load_documents

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Extracting documents from PDF...", total=None)
        documents = load_documents(pdf_path=settings.corpus_path)

    console.print(f"[green]Extracted {len(documents)} TOC sections[/green]")

    for cfg in EMBEDDING_MODELS.values():
        console.print(
            f"[dim]Indexing → {cfg.collection_name} ({cfg.model_id})...[/dim]"
        )

        llm.setup_embedding(cfg)
        build_index(documents, collection_name=cfg.collection_name)

    console.print("[bold green]✓ Indexing complete![/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="User query"),
    top_k: int | None = typer.Option(
        None, "--top-k", "-k", help="Number of context fragments"
    ),
) -> None:
    """
    Ask a question and get an answer from the LLM with sources from the documentation.

    Parameters
    ----------
    question : str
        Question about Ableton Live in any language.
    top_k : int or None, optional
        Number of context fragments. If not provided,
        ``settings.similarity_top_k`` is used.
    """

    from ableton_rag import llm
    from ableton_rag.query import ask as query_ask

    console.print("[dim]Initializing...[/dim]")
    llm.setup()

    k = top_k or settings.similarity_top_k

    console.print()
    console.print(Panel(question, title="[blue]Question[/blue]", border_style="blue"))
    console.print()

    async def _run() -> None:
        with console.status("[dim]Searching documentation...[/dim]"):
            answer = await query_ask(question, top_k=k)

        console.print(Panel.fit("[green]Answer[/green]", border_style="green"))

        async for token in answer.response_gen:
            console.print(token, end="")

        console.print("\n")

        if answer.source_nodes:
            table = Table(title="Sources", show_lines=True)
            table.add_column("#", style="cyan", width=4)
            table.add_column("Chapter / Section", style="white")
            table.add_column("Page", style="yellow", width=6)
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
    query: str = typer.Argument(..., help="Search query"),
    similarity_top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
) -> None:
    """
    Vector search without answer generation.

    Parameters
    ----------
    query : str
        Search query.
    similarity_top_k : int
        Number of results.
    """

    from ableton_rag import llm
    from ableton_rag.query import retrieve

    console.print("[dim]Initializing embeddings...[/dim]")
    llm.setup()

    with console.status("[dim]Searching...[/dim]"):
        results = asyncio.run(retrieve(query, similarity_top_k=similarity_top_k))

    for i, r in enumerate(results, 1):
        meta = f"Chapter: {r.chapter or '?'}"

        if r.section:
            meta += f" | Section: {r.section}"

        meta += f" | Page {r.page_start}"

        console.print(
            Panel(
                f"[dim]{meta}[/dim]\n\n{r.text[:600]}...",
                title=f"Result {i}  (score: {r.score:.4f})",
                border_style="cyan",
            )
        )


@app.command()
def stats() -> None:
    """
    Show statistics for the Qdrant collection.

    Prints a table with the point count, vector count and collection status.
    """

    from ableton_rag.index import get_stats

    info = get_stats()

    table = Table(title="Qdrant — collection statistics")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def bot() -> None:
    """
    Start the Telegram bot.

    The bot connects to a running FastAPI server (``rag serve``)
    and answers user questions from Telegram.
    """

    from ableton_rag.bot.bot import run

    if not settings.telegram_bot_token:
        console.print("[red]Bot token is not set.[/red]")
        raise typer.Exit(1)

    console.print("[bold green]Starting Telegram bot...[/bold green]")

    run(token=settings.telegram_bot_token, api_base_url=settings.api_base_url)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Port"),
) -> None:
    """
    Start the FastAPI application.

    Parameters
    ----------
    host : str
        Server address.
    port : int
        Port.
    """

    import uvicorn

    uvicorn.run("ableton_rag.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    app()
