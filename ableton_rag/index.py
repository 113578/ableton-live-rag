"""
Vector index management: creation, loading, statistics.
"""

import json

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from ableton_rag.config import EMBEDDING_MODELS, get_logger, settings

logger = get_logger(__name__)

_qdrant_client: QdrantClient | None = None
_async_qdrant_client: AsyncQdrantClient | None = None


def _active_collection() -> str:
    return EMBEDDING_MODELS[settings.active_embedding_model].collection_name


def _get_qdrant_client() -> QdrantClient:
    """
    Return the singleton ``QdrantClient`` instance.

    Returns
    -------
    QdrantClient
        Qdrant client.
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
    Return the singleton ``AsyncQdrantClient`` instance.

    Returns
    -------
    AsyncQdrantClient or None
        Asynchronous Qdrant client, or ``None`` for local storage.
    """

    global _async_qdrant_client

    if not settings.qdrant_url:
        return None

    if _async_qdrant_client is None:
        _async_qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)

    return _async_qdrant_client


def _get_vector_store(collection_name: str | None = None) -> QdrantVectorStore:
    """
    Create a ``QdrantVectorStore``.

    Parameters
    ----------
    collection_name : str or None, optional
        Collection name.

    Returns
    -------
    QdrantVectorStore
        Vector store for LlamaIndex.
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
    Build a ``VectorStoreIndex`` from documents and persist it to Qdrant.

    Parameters
    ----------
    documents : list[Document]
        List of documents from ``ingest.load_documents()``.
    collection_name : str or None, optional
        Collection name.

    Returns
    -------
    VectorStoreIndex
        Persisted index.
    """

    name = collection_name or _active_collection()
    client = _get_qdrant_client()

    logger.info("Building index '%s' from %d documents...", name, len(documents))

    if client.collection_exists(collection_name=name):
        client.delete_collection(collection_name=name)

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
    Load an existing ``VectorStoreIndex`` from Qdrant.

    Parameters
    ----------
    collection_name : str or None, optional
        Collection name.

    Returns
    -------
    VectorStoreIndex
        Loaded index.

    Raises
    ------
    RuntimeError
        If the collection is not found (ingest has not been run).
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
    Split documents into chunks using ``SentenceSplitter``.

    Parameters
    ----------
    documents : list[Document]
        List of documents from ``ingest.load_documents()``.

    Returns
    -------
    list[BaseNode]
        List of chunks.
    """

    parser = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return parser.get_nodes_from_documents(documents=documents)


def get_stats(collection_name: str | None = None) -> dict:
    """
    Return statistics for a Qdrant collection.

    Parameters
    ----------
    collection_name : str or None, optional
        Collection name.

    Returns
    -------
    dict
        Dictionary with the keys ``collection``, ``points_count``,
        ``indexed_vectors_count`` and ``status``.
    """

    name = collection_name or _active_collection()
    client = _get_qdrant_client()

    if not client.collection_exists(collection_name=name):
        return {
            "collection": name,
            "points_count": 0,
            "status": "not_found (run 'rag ingest')",
        }

    try:
        info = client.get_collection(collection_name=name)
    except (UnexpectedResponse, ValueError) as exc:
        logger.warning("Failed to read collection '%s' stats: %s", name, exc)
        return {
            "collection": name,
            "points_count": 0,
            "status": "error",
        }

    return {
        "collection": name,
        "points_count": info.points_count,
        "indexed_vectors_count": info.indexed_vectors_count,
        "status": info.status.value,
    }


def get_all_nodes(collection_name: str | None = None) -> list[BaseNode]:
    """
    Load all nodes from Qdrant (e.g. for a BM25 index).

    Parameters
    ----------
    collection_name : str or None, optional
        Collection name.

    Returns
    -------
    list[BaseNode]
        All nodes in the collection.
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
