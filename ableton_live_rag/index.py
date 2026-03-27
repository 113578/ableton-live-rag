"""
Управление векторным индексом: создание, загрузка, статистика.
"""

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ableton_live_rag.config import settings

_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    """
    Получение экземпляра QdrantClient.

    Returns
    -------
    QdrantClient
        Клиент Qdrant.
    """
    global _qdrant_client

    if _qdrant_client is None:
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        _qdrant_client = QdrantClient(path=str(settings.qdrant_path))

    return _qdrant_client


def _get_vector_store(
    client: QdrantClient,
    collection_name: str | None = None,
) -> QdrantVectorStore:
    """
    Создание QdrantVectorStore.

    Parameters
    ----------
    client : QdrantClient
        Клиент Qdrant.
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    QdrantVectorStore
        Хранилище векторов для LlamaIndex.
    """

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name or settings.collection_name,
    )


def build_index(
    documents: list[Document],
    collection_name: str | None = None,
) -> VectorStoreIndex:
    """
    Построение VectorStoreIndex из документов с сохранением в Qdrant.

    Parameters
    ----------
    documents : list[Document]
        Список документов из ``ingest.load_documents()``.
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    VectorStoreIndex
        Сохранённый индекс.
    """

    name = collection_name or settings.collection_name
    client = _get_qdrant_client()

    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass

    vector_store = _get_vector_store(client=client, collection_name=name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )


def load_index(collection_name: str | None = None) -> VectorStoreIndex:
    """
    Загрузка созданного VectorStoreIndex из Qdrant.

    Parameters
    ----------
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    VectorStoreIndex
        Загруженный индекс.

    Raises
    ------
    RuntimeError
        Если коллекция не найдена (инжест не был выполнен).
    """

    client = _get_qdrant_client()
    vector_store = _get_vector_store(client=client, collection_name=collection_name)

    try:
        return VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        name = collection_name or settings.collection_name
        raise RuntimeError(
            f"Индекс '{name}' не найден в Qdrant. "
            f"Сначала выполните 'rag ingest'.\nОшибка: {e}"
        ) from e


def parse_nodes(documents: list[Document]) -> list[BaseNode]:
    """
    Разбивка документов на чанки с помощью SentenceSplitter.

    Parameters
    ----------
    documents : list[Document]
        Список документов из ``ingest.load_documents()``.

    Returns
    -------
    list[BaseNode]
        Список чанков.
    """

    parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return parser.get_nodes_from_documents(documents=documents)


def get_stats(collection_name: str | None = None) -> dict:
    """
    Получить статистику коллекции Qdrant.

    Parameters
    ----------
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    dict
        Словарь с ключами ``collection``, ``points_count``,
        ``indexed_vectors_count``, ``status``.
    """

    name = collection_name or settings.collection_name
    client = _get_qdrant_client()

    try:
        info = client.get_collection(collection_name=name)
        return {
            "collection": name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value,
        }
    except Exception:
        return {
            "collection": name,
            "points_count": 0,
            "status": "not_found (запустите 'rag ingest')",
        }
