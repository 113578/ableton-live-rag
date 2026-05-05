"""
Pydantic request and response schemas for the API.
"""

from pydantic import BaseModel


class AskRequest(BaseModel):
    """
    Request body for the ``/ask`` endpoint.

    Attributes
    ----------
    question : str
        User question.
    top_k : int or None
        Number of context fragments.
    """

    question: str
    top_k: int | None = None


class ChatRequest(BaseModel):
    """
    Request body for the ``/chat`` endpoint.

    Attributes
    ----------
    message : str
        User message.
    session_id : str or None
        Session identifier. If not provided, a new session is created.
    top_k : int or None
        Number of context fragments (applied only when creating a session).
    """

    message: str
    session_id: str | None = None
    top_k: int | None = None


class SearchRequest(BaseModel):
    """
    Request body for the ``/search`` endpoint.

    Attributes
    ----------
    query : str
        Search query.
    top_k : int or None
        Number of results.
    """

    query: str
    top_k: int | None = None


class SearchResultOut(BaseModel):
    """
    Search result returned in JSON responses.

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
    chapter: str
    section: str
    subsection: str
    page_start: int
    metadata: dict
