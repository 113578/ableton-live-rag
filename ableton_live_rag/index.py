"""
Управление векторным индексом: создание, загрузка, статистика.
"""

import json

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from ableton_live_rag.config import EMBEDDING_MODELS, get_logger, settings

logger = get_logger(__name__)

_qdrant_client: QdrantClient | None = None
_async_qdrant_client: AsyncQdrantClient | None = None


def _active_collection() -> str:
    return EMBEDDING_MODELS[settings.active_embedding_model].collection_name


def _get_qdrant_client() -> QdrantClient:
    """
    Получение экземпляра QdrantClient (синглтон).

    Returns
    -------
    QdrantClient
        Клиент Qdrant.
    """

    global _qdrant_client

    if _qdrant_client is None:
        if settings.qdrant_url:
            logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
            _qdrant_client = QdrantClient(url=settings.qdrant_url)
        else:
            logger.info("Using local Qdrant at %s", settings.qdrant_path)
            settings.qdrant_path.mkdir(parents=True, exist_ok=True)
            _qdrant_client = QdrantClient(path=str(settings.qdrant_path))

    return _qdrant_client


def _get_async_qdrant_client() -> AsyncQdrantClient | None:
    """
    Получение экземпляра AsyncQdrantClient (синглтон).

    Returns
    -------
    AsyncQdrantClient or None
        Асинхронный клиент Qdrant, либо ``None`` для локального хранилища.
    """

    global _async_qdrant_client

    if not settings.qdrant_url:
        return None

    if _async_qdrant_client is None:
        _async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    return _async_qdrant_client


def _get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """
    Создание QdrantVectorStore.

    Parameters
    ----------
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    QdrantVectorStore
        Хранилище векторов для LlamaIndex.
    """

    aclient = _get_async_qdrant_client()

    return QdrantVectorStore(
        client=_get_qdrant_client(),
        aclient=aclient,
        collection_name=collection_name or _active_collection(),
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

    name = collection_name or _active_collection()
    client = _get_qdrant_client()

    logger.info("Building index '%s' from %d documents...", name, len(documents))

    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass

    vector_store = _get_vector_store(collection_name=name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("Index '%s' built successfully.", name)

    return index


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

    name = collection_name or _active_collection()
    client = _get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        raise RuntimeError(
            f"Index '{name}' not found in Qdrant. Run 'rag ingest' first."
        )

    logger.info("Loading index from collection '%s'.", name)

    vector_store = _get_vector_store(collection_name=name)

    return VectorStoreIndex.from_vector_store(vector_store)


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

    name = collection_name or _active_collection()
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


def get_all_nodes(collection_name: str | None = None) -> list[BaseNode]:
    """
    Загрузка всех узлов из Qdrant (для BM25-индекса).

    Parameters
    ----------
    collection_name : str or None, optional
        Имя коллекции.

    Returns
    -------
    list[BaseNode]
        Все узлы коллекции.
    """

    name = collection_name or _active_collection()
    client = _get_qdrant_client()
    nodes: list[BaseNode] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=name,
            with_payload=True,
            with_vectors=False,
            limit=1000,
            offset=offset,
        )

        for point in results:
            raw = point.payload.get("_node_content") if point.payload else None

            if raw:
                nodes.append(TextNode.model_validate(json.loads(raw)))

        if next_offset is None:
            break

        offset = next_offset

    logger.info("Loaded %d nodes from collection '%s'.", len(nodes), name)

    return nodes
