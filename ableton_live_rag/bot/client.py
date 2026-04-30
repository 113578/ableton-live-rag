"""
Асинхронный HTTP-клиент для RAG API.
"""

import json
from typing import AsyncIterator

import httpx


class RAGClient:
    """
    Клиент для взаимодействия с FastAPI-бэкендом RAG-системы.

    Parameters
    ----------
    base_url : str
        Базовый URL API.
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
        Отправка сообщения и получение потока SSE-событий.

        Yields
        ------
        tuple[str, object]
            Пара ``(event_type, content)`` для каждого события.
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

    async def delete_session(self, session_id: str) -> None:
        """Удаление сессии на сервере."""

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.delete(f"{self.base_url}/chat/{session_id}")
