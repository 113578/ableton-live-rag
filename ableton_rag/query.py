"""
Query pipeline: search the index and generate answers.
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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.storage.chat_store.redis import RedisChatStore
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

from ableton_rag import guardrails, index as idx
from ableton_rag.config import EMBEDDING_MODELS, get_logger, settings

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
    Search result.

    Attributes
    ----------
    text : str
        Text of the retrieved fragment.
    score : float
        Relevance score.
    chapter : str
        Chapter title.
    section : str
        Section title.
    subsection : str
        Subsection title.
    page_start : int
        Starting page (1-indexed).
    metadata : dict
        Full node metadata.
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
    Streaming answer with a list of sources.

    Attributes
    ----------
    source_nodes : list[SearchResult]
        Retrieved documentation fragments.
    response_gen : AsyncGenerator[str, None]
        Asynchronous generator yielding LLM response tokens.
    """

    source_nodes: list[SearchResult]
    response_gen: AsyncGenerator[str, None]


def _load_active_index() -> VectorStoreIndex:
    LlamaIndexSettings.embed_model = HuggingFaceEmbedding(
        model_name=_EMBEDDING_CONFIG.model_id,
        query_instruction=_EMBEDDING_CONFIG.query_instruction,
        text_instruction=_EMBEDDING_CONFIG.text_instruction,
    )

    return idx.load_index(collection_name=_EMBEDDING_CONFIG.collection_name)


def _build_vector_retriever(
    index: VectorStoreIndex, similarity_top_k: int
) -> VectorIndexRetriever:
    return VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)


def _build_query_engine(
    similarity_top_k: int, rerank: bool = False
) -> RetrieverQueryEngine:
    """
    Create a ``RetrieverQueryEngine`` with hybrid search (RRF) and an optional BGE reranker.

    Parameters
    ----------
    similarity_top_k : int
        Number of fragments retrieved as context.
    rerank : bool
        Whether to apply the BGE reranker as a post-processor.

    Returns
    -------
    RetrieverQueryEngine
        Configured query engine with streaming.
    """

    index = _load_active_index()
    retriever = _build_vector_retriever(index, similarity_top_k)
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
    Submit a question and get a streaming answer with sources.

    Parameters
    ----------
    question : str
        User question.
    top_k : int
        Number of documentation fragments used as context.

    Returns
    -------
    StreamingAnswer
        Object with ``source_nodes`` and an async ``response_gen``.
    """

    guard_result = await guardrails.guard(question)

    if not guard_result.safe:
        msg = guardrails.rejection_message(guard_result.category)

        async def _rejected_gen() -> AsyncGenerator[str, None]:
            yield msg

        return StreamingAnswer(source_nodes=[], response_gen=_rejected_gen())

    query = await guardrails.rewrite(question)

    cached = await asyncio.to_thread(_llmcache.check, query)

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

    logger.info("Cache MISS for: %r", query)

    engine = _build_query_engine(similarity_top_k=top_k)
    response = cast(
        StreamingResponse,
        await asyncio.to_thread(engine.query, query),
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
            _llmcache.store, query, full_text, None, {"sources": sources_json}
        )

        logger.info("Cache stored response for: %r", query)

    return StreamingAnswer(source_nodes=source_nodes, response_gen=_streaming_gen())


async def retrieve(
    query: str, similarity_top_k: int = settings.similarity_top_k
) -> list[SearchResult]:
    """
    Run vector search without generating an answer.

    Parameters
    ----------
    query : str
        Search query.
    similarity_top_k : int
        Number of results.

    Returns
    -------
    list[SearchResult]
        Results sorted by descending relevance.
    """

    guard_result = await guardrails.guard(query)

    if not guard_result.safe:
        return []

    rewritten = await guardrails.rewrite(query)

    index = _load_active_index()
    retriever = _build_vector_retriever(index, similarity_top_k)
    nodes = await asyncio.to_thread(retriever.retrieve, rewritten)

    return [_to_search_result(node) for node in nodes]


def create_chat_engine(
    session_id: str, top_k: int = settings.similarity_top_k
) -> ContextChatEngine:
    """
    Create a conversational chat engine with memory and documentation search.

    Parameters
    ----------
    session_id : str
        Session identifier.
    top_k : int
        Number of documentation fragments used as context per turn.

    Returns
    -------
    ContextChatEngine
        Engine with ``ChatMemoryBuffer`` and a vector retriever.
    """

    index = _load_active_index()
    retriever = _build_vector_retriever(index, top_k)

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


def _to_search_result(node: NodeWithScore) -> SearchResult:
    return SearchResult(
        text=node.text,
        score=node.score or 0.0,
        chapter=node.metadata.get("chapter", ""),
        section=node.metadata.get("section", ""),
        subsection=node.metadata.get("subsection", ""),
        page_start=node.metadata.get("page_start", 0),
        metadata=node.metadata,
    )
