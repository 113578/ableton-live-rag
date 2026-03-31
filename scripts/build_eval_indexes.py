"""
Построение Qdrant-коллекций для каждой модели эмбеддингов.
"""

from llama_index.core import Document, Settings as LlamaSettings
from rich.console import Console

from ableton_live_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_live_rag.index import build_index
from ableton_live_rag.ingest import load_documents
from experiments.utils import make_embed_model

console = Console()


def build_for_model(emb: EmbeddingModelConfig, documents: list[Document]) -> None:
    """
    Создание индекса для конкретной модели эмбеддингов.

    Parameters
    ----------
    emb : EmbeddingModelConfig
        Конфигурация модели эмбеддингов.
    documents : list[Document]
        Предзагруженные документы.
    """

    console.print(
        f"\n[bold cyan]▶ {emb.name}[/bold cyan] — {emb.model_id} (dim={emb.dim})"
    )

    LlamaSettings.embed_model = make_embed_model(emb)

    console.print(f"  Коллекция: [green]{emb.collection_name}[/green]")
    build_index(documents, collection_name=emb.collection_name)
    console.print("  [green]✓ Готово[/green]")


def main() -> None:
    """Точка входа."""

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    console.print("[dim]Загрузка документов...[/dim]")
    documents = load_documents()
    console.print(f"  Документов: {len(documents)}")

    console.print(
        f"[bold]Построение индексов для {len(EMBEDDING_MODELS)} моделей[/bold]"
    )

    for emb in EMBEDDING_MODELS.values():
        build_for_model(emb=emb, documents=documents)

    console.print("\n[bold green]Все индексы построены.[/bold green]")


if __name__ == "__main__":
    main()
