"""
FastAPI-приложение.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core.llms import ChatMessage

from ableton_live_rag import guardrails, llm
from ableton_live_rag import query as rag_query
from ableton_live_rag.config import EMBEDDING_MODELS, get_logger, settings
from ableton_live_rag.index import get_stats
from ableton_live_rag.api.schemas import (
    AskRequest,
    ChatRequest,
    SearchRequest,
    SearchResultOut,
)

logger = get_logger(__name__)


def _run_ingest() -> None:
    from ableton_live_rag.index import build_index
    from ableton_live_rag.ingest import load_documents

    documents = load_documents(pdf_path=settings.corpus_path)
    collection = EMBEDDING_MODELS[settings.active_embedding_model].collection_name

    build_index(documents, collection_name=collection)


async def _ensure_index() -> None:
    collection = EMBEDDING_MODELS[settings.active_embedding_model].collection_name
    stats = get_stats(collection_name=collection)

    if stats.get("points_count", 0) == 0:
        logger.info("Collection '%s' is empty — running ingest...", collection)

        await asyncio.to_thread(_run_ingest)

        logger.info("Ingest complete.")
    else:
        logger.info(
            "Collection '%s' already has %d points — skipping ingest.",
            collection,
            stats["points_count"],
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    llm.setup()
    await _ensure_index()
    yield


app = FastAPI(title="Ableton Live RAG", lifespan=_lifespan)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/stats")
async def stats() -> dict:
    """
    Статистика коллекции Qdrant.

    Returns
    -------
    dict
        Словарь с параметрами и значениями коллекции.
    """

    return get_stats(
        collection_name=EMBEDDING_MODELS[
            settings.active_embedding_model
        ].collection_name
    )


@app.post("/search", response_model=list[SearchResultOut])
async def search(req: SearchRequest) -> list[SearchResultOut]:
    """
    Векторный поиск по документации без генерации ответа.

    Parameters
    ----------
    req : SearchRequest
        Запрос с поисковой строкой и опциональным top_k.

    Returns
    -------
    list[SearchResultOut]
        Список результатов, отсортированных по убыванию релевантности.
    """

    k = req.top_k or settings.similarity_top_k
    results = await rag_query.retrieve(req.query, similarity_top_k=k)

    return [SearchResultOut(**asdict(r)) for r in results]


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """
    Задать вопрос и получить SSE-стрим токенов ответа и источников.

    Parameters
    ----------
    req : AskRequest
        Запрос с вопросом и опциональным top_k.

    Returns
    -------
    StreamingResponse
        Поток Server-Sent Events.
    """

    k = req.top_k or settings.similarity_top_k
    answer = await rag_query.ask(req.question, top_k=k)

    async def _generate():
        async for token in answer.response_gen:
            yield _sse({"type": "token", "content": token})

        sources = [
            SearchResultOut(**asdict(r)).model_dump() for r in answer.source_nodes
        ]

        yield _sse({"type": "sources", "content": sources})
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Диалоговый чат с сохранением истории в рамках сессии.

    Parameters
    ----------
    req : ChatRequest
        Сообщение, опциональный ``session_id`` и ``top_k``.

    Returns
    -------
    StreamingResponse
        Поток Server-Sent Events.
    """

    session_id = req.session_id or str(uuid.uuid4())
    k = req.top_k or settings.similarity_top_k

    async def _generate():
        yield _sse({"type": "session_id", "content": session_id})

        guard_result = await guardrails.guard(req.message)

        if not guard_result.safe:
            yield _sse(
                {
                    "type": "token",
                    "content": guardrails.rejection_message(guard_result.category),
                }
            )
            yield "data: [DONE]\n\n"
            return

        query = await guardrails.rewrite(req.message)
        engine = rag_query.create_chat_engine(session_id=session_id, top_k=k)
        cached = await asyncio.to_thread(rag_query._llmcache.check, query)

        if cached:
            logger.info("Cache HIT for session %s: %r", session_id, query)

            full_text: str = cached[0]["response"]
            sources: list[dict] = json.loads(
                cached[0].get("metadata", {}).get("sources", "[]")
            )
            engine.memory.put(ChatMessage(role="user", content=query))
            engine.memory.put(ChatMessage(role="assistant", content=full_text))

            yield _sse({"type": "token", "content": full_text})
        else:
            logger.info("Cache MISS for session %s: %r", session_id, query)

            response = await engine.astream_chat(query)

            tokens: list[str] = []
            async for token in response.async_response_gen():
                tokens.append(token)
                yield _sse({"type": "token", "content": token})

            sources = [
                SearchResultOut(**asdict(r)).model_dump()
                for r in rag_query._chat_to_search_results(response)
            ]
            full_text = "".join(tokens)
            sources_json = json.dumps(sources)

            await asyncio.to_thread(
                rag_query._llmcache.store,
                query,
                full_text,
                None,
                {"sources": sources_json},
            )

            logger.info("Cache stored response for session %s: %r", session_id, query)

        yield _sse({"type": "sources", "content": sources})
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
