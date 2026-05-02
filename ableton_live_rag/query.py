"""
Пайплайн запросов: поиск по индексу и генерация ответов.
"""

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import AsyncGenerator, cast

from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core import Settings as LlamaIndexSettings
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.chat_store.redis import RedisChatStore
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

from ableton_live_rag import index as idx
from ableton_live_rag.config import EMBEDDING_MODELS, get_logger, settings

logger = get_logger(__name__)

_BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"
_EMBEDDING_CONFIG = EMBEDDING_MODELS[settings.active_embedding_model]


_CHAT_SYSTEM_PROMPT = """\
You are an expert assistant for the Ableton ecosystem. Answer questions based \
exclusively on the provided documentation excerpts.

Rules:
- Answer only from the context. If it lacks enough information, say so.
- Be concise and practical — users are musicians and producers.
- Structure your answer with line breaks between thoughts; never return a wall of text.
- Use bullet lists for steps or multiple items; keep each bullet short.
- Bold key terms with **double asterisks**.
- Use a relevant emoji at the start of each bullet or key paragraph (🎛️ 🎚️ 🎹 ⌨️ 📁 💡 ▶️ 🔁 etc.) — one per point, not in every sentence.
- When describing UI actions, mention exact menu paths and keyboard shortcuts.\
"""

_TEXT_QA_TEMPLATE = RichPromptTemplate(
    """\
    You are an expert assistant for the Ableton ecosystem. Answer questions based \
    exclusively on the provided documentation excerpts.

    Rules:
    - Answer only from the context below. If it lacks enough information, say so.
    - Be concise and practical — users are musicians and producers.
    - Structure your answer with line breaks between thoughts; never return a wall of text.
    - Use bullet lists for steps or multiple items; keep each bullet short.
    - Bold key terms with **double asterisks**.
    - Use a relevant emoji at the start of each bullet or key paragraph (🎛️ 🎚️ 🎹 ⌨️ 📁 💡 ▶️ 🔁 etc.) — one per point, not in every sentence.
    - When describing UI actions, mention exact menu paths and keyboard shortcuts.

    Documentation context:
    ---------------------
    {{ context_str }}
    ---------------------

    Question: {{ query_str }}
    Answer: \
    """
)

if not settings.redis_url:
    raise RuntimeError("REDIS_URL is required but not set.")

_chat_store = RedisChatStore(redis_url=settings.redis_url, ttl=86400)
_llmcache = SemanticCache(
    name=settings.collection_name,
    redis_url=settings.redis_url,
    distance_threshold=0.1,
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1"),
    ttl=86400,
)


@dataclass
class SearchResult:
    """
    Результат поиска.

    Attributes
    ----------
    text : str
        Текст найденного фрагмента.
    score : float
        Оценка релевантности.
    chapter : str
        Название главы.
    section : str
        Название раздела.
    subsection : str
        Название подраздела.
    page_start : int
        Начальная страница (1-indexed).
    metadata : dict
        Полные метаданные узла.
    """

    text: str
    score: float
    chapter: str = ""
    section: str = ""
    subsection: str = ""
    page_start: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class StreamingAnswer:
    """
    Ответ с поддержкой стриминга и списком источников.

    Attributes
    ----------
    source_nodes : list[SearchResult]
        Найденные фрагменты документации.
    response_gen : AsyncGenerator[str, None]
        Асинхронный генератор токенов ответа LLM.
    """

    source_nodes: list[SearchResult]
    response_gen: AsyncGenerator[str, None]


def _load_bge_index() -> tuple[VectorStoreIndex, list]:
    LlamaIndexSettings.embed_model = HuggingFaceEmbedding(
        model_name=_EMBEDDING_CONFIG.model_id,
        query_instruction=_EMBEDDING_CONFIG.query_instruction,
        text_instruction=_EMBEDDING_CONFIG.text_instruction,
    )

    collection = _EMBEDDING_CONFIG.collection_name
    index = idx.load_index(collection_name=collection)
    nodes = idx.get_all_nodes(collection_name=collection)

    return index, nodes


def _build_hybrid_retriever(
    index: VectorStoreIndex, nodes: list, similarity_top_k: int
) -> QueryFusionRetriever:
    fetch_k = similarity_top_k * 2
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=fetch_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=fetch_k)

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=False,
        verbose=False,
    )


def _build_query_engine(
    similarity_top_k: int, rerank: bool = False
) -> RetrieverQueryEngine:
    """
    Создание RetrieverQueryEngine с гибридным поиском (RRF) и опциональным BGE reranker.

    Parameters
    ----------
    similarity_top_k : int
        Количество фрагментов для контекста.
    rerank : bool
        Использовать ли BGE reranker в качестве постпроцессора.

    Returns
    -------
    RetrieverQueryEngine
        Настроенный движок запросов со стримингом.
    """

    index, nodes = _load_bge_index()
    retriever = _build_hybrid_retriever(index, nodes, similarity_top_k)
    postprocessors: list[BaseNodePostprocessor] = (
        [SentenceTransformerRerank(model=_BGE_RERANKER_MODEL, top_n=similarity_top_k)]
        if rerank
        else []
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        streaming=True,
        text_qa_template=_TEXT_QA_TEMPLATE,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=postprocessors,
    )


async def ask(question: str, top_k: int = settings.similarity_top_k) -> StreamingAnswer:
    """
    Постановка вопроса и получение стримингового ответа с источниками.

    Parameters
    ----------
    question : str
        Вопрос пользователя.
    top_k : int
        Количество фрагментов документации для контекста.

    Returns
    -------
    StreamingAnswer
        Объект с ``source_nodes`` и асинхронным ``response_gen``.
    """

    cached = await asyncio.to_thread(_llmcache.check, question)

    if cached:
        logger.info("Cache HIT for: %r", question)

        response_text: str = cached[0]["response"]
        sources_data: list[dict] = json.loads(
            cached[0].get("metadata", {}).get("sources", "[]")
        )
        source_nodes = [SearchResult(**s) for s in sources_data]

        async def _cached_gen() -> AsyncGenerator[str, None]:
            yield response_text

        return StreamingAnswer(source_nodes=source_nodes, response_gen=_cached_gen())

    logger.info("Cache MISS for: %r", question)
    engine = _build_query_engine(similarity_top_k=top_k)
    response = cast(
        StreamingResponse,
        await asyncio.to_thread(engine.query, question),
    )
    source_nodes = [_to_search_result(node) for node in response.source_nodes]

    async def _streaming_gen() -> AsyncGenerator[str, None]:
        tokens: list[str] = []

        for token in response.response_gen:
            tokens.append(token)
            yield token

        full_text = "".join(tokens)
        sources_json = json.dumps([asdict(n) for n in source_nodes])
        await asyncio.to_thread(
            _llmcache.store, question, full_text, None, {"sources": sources_json}
        )
        logger.info("Cache stored response for: %r", question)

    return StreamingAnswer(source_nodes=source_nodes, response_gen=_streaming_gen())


async def retrieve(
    query: str, similarity_top_k: int = settings.similarity_top_k
) -> list[SearchResult]:
    """
    Выполнение векторного поиска без генерации ответа.

    Parameters
    ----------
    query : str
        Поисковый запрос.
    similarity_top_k : int
        Количество результатов.

    Returns
    -------
    list[SearchResult]
        Список результатов, отсортированный по убыванию релевантности.
    """

    index, corpus_nodes = _load_bge_index()
    retriever = _build_hybrid_retriever(index, corpus_nodes, similarity_top_k)
    nodes = await asyncio.to_thread(retriever.retrieve, query)

    return [_to_search_result(node) for node in nodes]


def create_chat_engine(
    session_id: str, top_k: int = settings.similarity_top_k
) -> ContextChatEngine:
    """
    Создание движка диалогового чата с памятью и поиском по документации.

    Parameters
    ----------
    session_id : str
        Идентификатор сессии.
    top_k : int
        Количество фрагментов документации для контекста на каждый ход.

    Returns
    -------
    ContextChatEngine
        Движок с ``ChatMemoryBuffer`` и векторным ретривером.
    """

    index, nodes = _load_bge_index()
    retriever = _build_hybrid_retriever(index, nodes, top_k)

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=settings.context_window // 2,
        chat_store=_chat_store,
        chat_store_key=session_id,
    )

    return ContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=_CHAT_SYSTEM_PROMPT,
    )


def _chat_to_search_results(response: StreamingAgentChatResponse) -> list[SearchResult]:
    return [_to_search_result(node) for node in (response.source_nodes or [])]


def _to_search_result(node) -> SearchResult:
    return SearchResult(
        text=node.text,
        score=node.score or 0.0,
        chapter=node.metadata.get("chapter", ""),
        section=node.metadata.get("section", ""),
        subsection=node.metadata.get("subsection", ""),
        page_start=node.metadata.get("page_start", 0),
        metadata=node.metadata,
    )
