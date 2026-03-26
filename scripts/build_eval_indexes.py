"""
Построение Qdrant-коллекций для каждой модели эмбеддингов.
"""

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console

from ableton_live_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from ableton_live_rag.index import build_index
from ableton_live_rag.ingest import load_documents

console = Console()


def build_for_model(emb: EmbeddingModelConfig) -> None:
    """
    Создание индекса для конкретной модели эмбеддингов.

    Parameters
    ----------
    emb : EmbeddingModelConfig
        Конфигурация модели эмбеддингов для экспериментов.
    """

    console.print(
        f"\n[bold cyan]▶ {emb.name}[/bold cyan] — {emb.model_id} (dim={emb.dim})"
    )

    LlamaSettings.embed_model = HuggingFaceEmbedding(
        model_name=emb.model_id,
        query_instruction=emb.query_instruction or None,
        text_instruction=emb.text_instruction or None,
    )

    console.print("[dim]  Загрузка документов...[/dim]")
    documents = load_documents()
    console.print(f"  Документов: {len(documents)}")

    console.print(f"  Коллекция: [green]{emb.collection_name}[/green]")
    build_index(documents, collection_name=emb.collection_name)
    console.print("  [green]✓ Готово[/green]")


def main() -> None:
    """Точка входа."""

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    models = EMBEDDING_MODELS

    console.print(f"[bold]Построение индексов для {len(models)} моделей[/bold]")

    for emb in models.values():
        build_for_model(emb=emb)

    console.print("\n[bold green]Все индексы построены.[/bold green]")


if __name__ == "__main__":
    main()
