"""
Пайплайн запросов: поиск по индексу и генерация ответов.
"""

from dataclasses import dataclass, field
from typing import Generator, cast

from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.retrievers import VectorIndexRetriever

from ableton_live_rag import index as idx
from ableton_live_rag.config import settings


_TEXT_QA_TEMPLATE = RichPromptTemplate(
    """\
    You are an expert assistant for Ableton Live 12. Answer questions based \
    exclusively on the provided documentation excerpts.

    Rules:
    - Answer only from the context below. If it lacks enough information, say so.
    - Cite sources as [Source 1], [Source 2], etc.
    - Be precise and practical — users are musicians and producers.
    - When describing UI actions, mention exact menu paths and keyboard shortcuts.

    Documentation context:
    ---------------------
    {{ context_str }}
    ---------------------

    Question: {{ query_str }}
    Answer: \
    """
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
    response_gen : Generator[str, None, None]
        Генератор токенов ответа LLM.
    """

    source_nodes: list[SearchResult]
    response_gen: Generator[str, None, None]


def _build_query_engine(similarity_top_k: int) -> RetrieverQueryEngine:
    """
    Создание RetrieverQueryEngine с кастомным промптом и поддержкой стриминга.

    Parameters
    ----------
    similarity_top_k : int
        Количество фрагментов для контекста.

    Returns
    -------
    RetrieverQueryEngine
        Настроенный движок запросов со стримингом.
    """

    index = idx.load_index()

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,
    )

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        streaming=True,
        text_qa_template=_TEXT_QA_TEMPLATE,
    )

    return RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )


def ask(question: str, top_k: int = settings.similarity_top_k) -> StreamingAnswer:
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
        Объект с ``source_nodes`` и ``response_gen``.
    """

    engine = _build_query_engine(similarity_top_k=top_k)
    response = cast(StreamingResponse, engine.query(str_or_query_bundle=question))

    source_nodes = [
        SearchResult(
            text=node.text,
            score=node.score or 0.0,
            chapter=node.metadata.get("chapter", ""),
            section=node.metadata.get("section", ""),
            subsection=node.metadata.get("subsection", ""),
            page_start=node.metadata.get("page_start", 0),
            metadata=node.metadata,
        )
        for node in response.source_nodes
    ]

    return StreamingAnswer(
        source_nodes=source_nodes,
        response_gen=response.response_gen,
    )


def retrieve(
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

    vector_index = idx.load_index()
    retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=similarity_top_k
    )
    nodes = retriever.retrieve(str_or_query_bundle=query)

    return [
        SearchResult(
            text=node.text,
            score=node.score or 0.0,
            chapter=node.metadata.get("chapter", ""),
            section=node.metadata.get("section", ""),
            subsection=node.metadata.get("subsection", ""),
            page_start=node.metadata.get("page_start", 0),
            metadata=node.metadata,
        )
        for node in nodes
    ]
