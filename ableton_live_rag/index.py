"""
Управление векторным индексом: создание, загрузка, статистика.
"""

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ableton_live_rag.config import settings


def _get_qdrant_client() -> QdrantClient:
    """
    Создание локального клиента Qdrant.

    Returns
    -------
    QdrantClient
        Клиент Qdrant.
    """

    settings.qdrant_path.mkdir(parents=True, exist_ok=True)

    return QdrantClient(path=str(settings.qdrant_path))


def _get_vector_store(client: QdrantClient) -> QdrantVectorStore:
    """
    Создание QdrantVectorStore.

    Parameters
    ----------
    client : QdrantClient
        Клиент Qdrant.

    Returns
    -------
    QdrantVectorStore
        Хранилище векторов для LlamaIndex.
    """

    return QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
    )


def build_index(documents: list[Document]) -> VectorStoreIndex:
    """
    Построение VectorStoreIndex из документов с сохранением в Qdrant.

    Parameters
    ----------
    documents : list[Document]
        Список документов из ``ingest.load_documents()``.

    Returns
    -------
    VectorStoreIndex
        Сохранённый индекс.
    """
    client = _get_qdrant_client()

    # Удаляем старую коллекцию, чтобы избежать дублирования векторов при повторном запуске
    try:
        client.delete_collection(collection_name=settings.collection_name)
    except Exception:
        pass

    vector_store = _get_vector_store(client=client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )


def load_index() -> VectorStoreIndex:
    """
    Загрузка созданного VectorStoreIndex из Qdrant.

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
    vector_store = _get_vector_store(client=client)

    try:
        return VectorStoreIndex.from_vector_store(vector_store)
    except Exception as e:
        raise RuntimeError(
            f"Индекс не найден в Qdrant. Сначала выполните 'rag ingest'.\nОшибка: {e}"
        ) from e


def parse_nodes(documents: list[Document]) -> list[TextNode]:
    """
    Разбивка документов на чанки с помощью SentenceSplitter.

    Parameters
    ----------
    documents : list[Document]
        Список документов из ``ingest.load_documents()``.

    Returns
    -------
    list[TextNode]
        Список чанков.
    """

    parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return parser.get_nodes_from_documents(documents=documents)


def get_stats() -> dict:
    """
    Получить статистику коллекции Qdrant.

    Returns
    -------
    dict
        Словарь с ключами ``collection``, ``points_count``, ``vectors_count``, ``status``.
    """

    client = _get_qdrant_client()

    try:
        info = client.get_collection(collection_name=settings.collection_name)
        return {
            "collection": settings.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value,
        }
    except Exception:
        return {
            "collection": settings.collection_name,
            "points_count": 0,
            "status": "not_found (запустите 'rag ingest')",
        }
