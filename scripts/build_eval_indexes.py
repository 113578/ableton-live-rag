"""
Build Qdrant collections for each embedding model.
"""

from llama_index.core import Document, Settings as LlamaSettings
from rich.console import Console

from ableton_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_rag.index import build_index
from ableton_rag.ingest import load_documents
from experiments.components.retrievers import make_embed_model

console = Console()


def build_for_model(emb: EmbeddingModelConfig, documents: list[Document]) -> None:
    """
    Build the index for a specific embedding model.

    Parameters
    ----------
    emb : EmbeddingModelConfig
        Embedding-model configuration.
    documents : list[Document]
        Pre-loaded documents.
    """

    console.print(
        f"\n[bold cyan]▶ {emb.name}[/bold cyan] — {emb.model_id} (dim={emb.dim})"
    )

    LlamaSettings.embed_model = make_embed_model(emb)

    console.print(f"  Collection: [green]{emb.collection_name}[/green]")
    build_index(documents, collection_name=emb.collection_name)
    console.print("  [green]✓ Done[/green]")


def main() -> None:
    """Entry point."""

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    console.print("[dim]Loading documents...[/dim]")
    documents = load_documents()
    console.print(f"  Documents: {len(documents)}")

    console.print(f"[bold]Building indexes for {len(EMBEDDING_MODELS)} models[/bold]")

    for emb in EMBEDDING_MODELS.values():
        build_for_model(emb=emb, documents=documents)

    console.print("\n[bold green]All indexes built.[/bold green]")


if __name__ == "__main__":
    main()
