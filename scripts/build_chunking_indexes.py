"""
Построение Qdrant-коллекций для разных параметров чанкинга.

Создаёт индексы с именами вида ``{base}_{model}_cs{chunk_size}_co{overlap}``
для каждой комбинации chunk_size × overlap.

Примеры:
    uv run scripts/build_chunking_indexes.py --chunk-size 256 --overlap 32
"""

import typer
from llama_index.core import Settings as LlamaSettings
from rich.console import Console

from ableton_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_rag.index import build_index
from ableton_rag.ingest import load_documents
from experiments.components.retrievers import make_embed_model
from experiments.utils import chunking_collection_name

console = Console()


def _build_for_model(
    emb: EmbeddingModelConfig,
    documents: list,
    chunk_size: int,
    overlap: int,
) -> None:
    cname = chunking_collection_name(emb, chunk_size, overlap)
    console.print(
        f"  [dim]{emb.name}[/dim] (dim={emb.dim}) — коллекция: [green]{cname}[/green]"
    )
    LlamaSettings.embed_model = make_embed_model(emb)
    build_index(documents, collection_name=cname)
    console.print("  [green]✓ Готово[/green]")


def main(
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
) -> None:
    """
    Построение Qdrant-коллекций для всех комбинаций chunk_size × overlap.
    """

    embedding_model = EMBEDDING_MODELS[settings.active_embedding_model]

    console.print("[dim]Загрузка документов...[/dim]")
    documents = load_documents()
    console.print(f"  Документов: {len(documents)}")

    pairs = [(cs, ov) for cs in chunk_sizes for ov in overlaps]
    total = len(pairs)
    console.print(f"\n[bold]Построение {total} индексов.")

    for chunk_size, overlap in pairs:
        console.print(
            f"\n[bold cyan]▶ chunk_size={chunk_size}, overlap={overlap}[/bold cyan]"
        )
        LlamaSettings.chunk_size = chunk_size
        LlamaSettings.chunk_overlap = overlap

        _build_for_model(
            emb=embedding_model,
            documents=documents,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    console.print("\n[bold green]Индексы построены.[/bold green]")


if __name__ == "__main__":
    typer.run(main)
