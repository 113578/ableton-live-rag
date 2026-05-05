"""
Build Qdrant collections for different chunking parameters.

Creates indexes with names of the form ``{base}_{model}_cs{chunk_size}_co{overlap}``
for each combination of chunk_size × overlap.

Examples:
    uv run scripts/build_chunking_indexes.py --chunk-size 256 --overlap 32
"""

import typer
from llama_index.core import Document, Settings as LlamaSettings
from rich.console import Console

from ableton_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_rag.index import build_index
from ableton_rag.ingest import load_documents
from experiments.components.retrievers import make_embed_model
from experiments.utils import chunking_collection_name

console = Console()


def _build_for_model(
    emb: EmbeddingModelConfig,
    documents: list[Document],
    chunk_size: int,
    overlap: int,
) -> None:
    cname = chunking_collection_name(emb, chunk_size, overlap)
    console.print(
        f"  [dim]{emb.name}[/dim] (dim={emb.dim}) — collection: [green]{cname}[/green]"
    )
    LlamaSettings.embed_model = make_embed_model(emb)
    build_index(documents, collection_name=cname)
    console.print("  [green]✓ Done[/green]")


def main(
    chunk_sizes: list[int] = typer.Option(
        [128, 256, 512, 1024],
        "--chunk-size",
        "-c",
        help="Chunk sizes in tokens",
    ),
    overlaps: list[int] = typer.Option(
        [16, 32, 64, 128],
        "--overlap",
        "-o",
        help="Chunk overlaps in tokens",
    ),
) -> None:
    """
    Build Qdrant collections for every chunk_size × overlap combination.
    """

    embedding_model = EMBEDDING_MODELS[settings.active_embedding_model]

    console.print("[dim]Loading documents...[/dim]")
    documents = load_documents()
    console.print(f"  Documents: {len(documents)}")

    pairs = [(cs, ov) for cs in chunk_sizes for ov in overlaps]
    total = len(pairs)
    console.print(f"\n[bold]Building {total} indexes.")

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

    console.print("\n[bold green]Indexes built.[/bold green]")


if __name__ == "__main__":
    typer.run(main)
