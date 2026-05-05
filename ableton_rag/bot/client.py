"""
Asynchronous HTTP client for the RAG API.
"""

import json
from typing import AsyncIterator

import httpx


class RAGClient:
    """
    Client for talking to the FastAPI backend of the RAG system.

    Parameters
    ----------
    base_url : str
        Base URL of the API.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        timeout: float = 120,
    ) -> AsyncIterator[tuple[str, object]]:
        """
        Stream SSE events from the ``/chat`` endpoint.

        Parameters
        ----------
        message : str
            Text of the user's message.
        session_id : str or None, optional
            Session identifier used to persist dialogue history.
        timeout : float, optional
            HTTP connection timeout in seconds.

        Yields
        ------
        tuple[str, object]
            Pair ``(event_type, content)`` for each SSE event.
        """

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat",
                json={"message": message, "session_id": session_id},
            ) as response:
                response.raise_for_status()

                try:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data = line[6:]

                        if data == "[DONE]":
                            return

                        try:
                            event = json.loads(data)
                            yield event["type"], event["content"]

                        except (json.JSONDecodeError, KeyError):
                            continue

                except httpx.RemoteProtocolError:
                    return
