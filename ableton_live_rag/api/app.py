"""
FastAPI-приложение.
"""

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_index.core.chat_engine import ContextChatEngine

from ableton_live_rag import llm
from ableton_live_rag import query as rag_query
from ableton_live_rag.config import EMBEDDING_MODELS, settings
from ableton_live_rag.index import get_stats
from ableton_live_rag.api.schemas import (
    AskRequest,
    ChatRequest,
    SearchRequest,
    SearchResultOut,
)

_sessions: dict[str, ContextChatEngine] = {}


@asynccontextmanager
async def _lifespan(app: FastAPI):
    llm.setup()
    yield


app = FastAPI(title="Ableton Live RAG", lifespan=_lifespan)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


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

    if session_id not in _sessions:
        if req.session_id is not None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        k = req.top_k or settings.similarity_top_k
        _sessions[session_id] = rag_query.create_chat_engine(top_k=k)

    engine = _sessions[session_id]

    async def _generate():
        yield _sse({"type": "session_id", "content": session_id})

        response = await engine.astream_chat(req.message)

        async for token in response.async_response_gen():
            yield _sse({"type": "token", "content": token})

        sources = [
            SearchResultOut(**asdict(r)).model_dump()
            for r in rag_query._chat_to_search_results(response)
        ]

        yield _sse({"type": "sources", "content": sources})
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.delete("/chat/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    """
    Удаление сессии и очистка истории диалога.

    Parameters
    ----------
    session_id : str
        Идентификатор сессии.
    """

    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    del _sessions[session_id]
