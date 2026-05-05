"""
FastAPI application.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core.llms import ChatMessage

from ableton_rag import guardrails, llm
from ableton_rag import query as rag_query
from ableton_rag.config import EMBEDDING_MODELS, get_logger, settings
from ableton_rag.index import get_stats
from ableton_rag.api.schemas import (
    AskRequest,
    ChatRequest,
    SearchRequest,
    SearchResultOut,
)

logger = get_logger(__name__)


def _run_ingest() -> None:
    from ableton_rag.index import build_index
    from ableton_rag.ingest import load_documents

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
    Return statistics for the Qdrant collection.

    Returns
    -------
    dict
        Dictionary of collection parameters and values.
    """

    return get_stats(
        collection_name=EMBEDDING_MODELS[
            settings.active_embedding_model
        ].collection_name
    )


@app.post("/search", response_model=list[SearchResultOut])
async def search(req: SearchRequest) -> list[SearchResultOut]:
    """
    Vector search over the documentation without answer generation.

    Parameters
    ----------
    req : SearchRequest
        Request with the search string and an optional ``top_k``.

    Returns
    -------
    list[SearchResultOut]
        Results sorted by descending relevance.
    """

    k = req.top_k or settings.similarity_top_k
    results = await rag_query.retrieve(req.query, similarity_top_k=k)

    return [SearchResultOut(**asdict(r)) for r in results]


@app.post("/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """
    Ask a question and stream answer tokens and sources via SSE.

    Parameters
    ----------
    req : AskRequest
        Request with a question and an optional ``top_k``.

    Returns
    -------
    StreamingResponse
        Server-Sent Events stream.
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
    Conversational chat with per-session history.

    Parameters
    ----------
    req : ChatRequest
        Message, optional ``session_id`` and ``top_k``.

    Returns
    -------
    StreamingResponse
        Server-Sent Events stream.
    """

    session_id = req.session_id or str(uuid.uuid4())
    k = req.top_k or settings.similarity_top_k

    async def _generate():
        yield _sse({"type": "session_id", "content": session_id})

        history_messages = await asyncio.to_thread(
            rag_query._chat_store.get_messages, session_id
        )
        history = [m.content for m in history_messages[-4:] if m.content]

        guard_result = await guardrails.guard(req.message, history=history)

        if not guard_result.safe:
            yield _sse(
                {
                    "type": "token",
                    "content": guardrails.rejection_message(guard_result.category),
                }
            )
            yield "data: [DONE]\n\n"
            return

        query = await guardrails.rewrite(req.message, history=history)
        engine = rag_query.create_chat_engine(session_id=session_id, top_k=k)
        cached = await asyncio.to_thread(rag_query._llmcache.check, query)

        if cached:
            logger.info("Cache HIT for session %s: %r", session_id, query)

            full_text: str = cached[0]["response"]
            sources: list[dict] = json.loads(
                cached[0].get("metadata", {}).get("sources", "[]")
            )
            engine._memory.put(ChatMessage(role="user", content=query))
            engine._memory.put(ChatMessage(role="assistant", content=full_text))

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
