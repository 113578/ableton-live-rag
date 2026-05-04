"""
Pydantic-схемы запросов и ответов API.
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    Запрос к эндпоинту /ask.

    Attributes
    ----------
    question : str
        Вопрос пользователя.
    top_k : int or None
        Количество фрагментов для контекста.
    """

    question: str
    top_k: int | None = None


class ChatRequest(BaseModel):
    """
    Запрос к эндпоинту /chat.

    Attributes
    ----------
    message : str
        Сообщение пользователя.
    session_id : str or None
        Идентификатор сессии. Если не указан, создаётся новая сессия.
    top_k : int or None
        Количество фрагментов для контекста (применяется только при создании сессии).
    """

    message: str
    session_id: str | None = None
    top_k: int | None = None


class SearchRequest(BaseModel):
    """
    Запрос к эндпоинту /search.

    Attributes
    ----------
    query : str
        Поисковый запрос.
    top_k : int or None
        Количество результатов.
    """

    query: str
    top_k: int | None = None


class SearchResultOut(BaseModel):
    """
    Результат поиска для JSON-ответа.

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
    chapter: str
    section: str
    subsection: str
    page_start: int
    metadata: dict
