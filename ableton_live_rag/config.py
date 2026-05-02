"""
Конфигурация проекта.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class LLMProvider(str, Enum):
    """
    Поддерживаемые провайдеры LLM.
    """

    ollama = "ollama"
    openai = "openai"
    vllm = "vllm"


class Settings(BaseSettings):
    """
    Конфигурация проекта.

    Все параметры можно переопределить через переменные окружения
    или файл ``.env`` в корне репозитория.

    Attributes
    ----------
    llm_provider : LLMProvider
        Провайдер LLM (``ollama``, ``openai`` или ``vllm``).
    request_timeout : int
        Таймаут запроса к Ollama в секундах.
    temperature : float
        Температура генерации.
    ollama_base_url : str
        Базовый URL сервера Ollama.
    ollama_model : str
        Имя модели Ollama.
    openai_api_key : str
        API-ключ OpenAI.
    openai_model : str
        Имя модели OpenAI.
    vllm_url_base : str
        Базовый URL vLLM-сервера (OpenAI-совместимый API).
    vllm_api_key : str
        API-ключ для vLLM.
    vllm_model : str
        Имя модели vLLM.
    guard_model : str
        Имя модели для guardrails. Если пустое, используется основная LLM.
    embedding_model : str
        Идентификатор модели эмбеддингов (для основного пайплайна).
    embedding_dim : int
        Размерность эмбеддинга основной модели.
    corpus_path : Path
        Путь к директории с PDF-файлами корпуса.
    qdrant_path : Path
        Путь к директории с хранилищем Qdrant (используется только
        при пустом ``qdrant_url``).
    qdrant_url : str
        URL для клиента Qdrant.
    redis_url : str
        URL для клиента Redis.
    collection_name : str
        Базовое имя коллекции Qdrant.
    chunk_size : int
        Максимальный размер чанка в токенах.
    chunk_overlap : int
        Перекрытие между соседними чанками в токенах.
    context_window : int
        Размер контекстного окна LLM в токенах.
    num_output : int
        Максимальное число токенов в ответе LLM.
    similarity_top_k : int
        Количество фрагментов, возвращаемых компонентом поиска по умолчанию.
    telegram_bot_token : str
        Токен Telegram-бота.
    api_base_url : str
        URL FastAPI-приложения.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.vllm

    context_window: int = 32768
    num_output: int = 4096
    request_timeout: int = 120
    temperature: float = 0.7

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5"

    vllm_url_base: str = "http://localhost:9999"
    vllm_api_key: str = ""
    vllm_model: str = "gemma-4"

    openai_api_key: str = ""
    openai_model: str = "gpt-5.4"

    guard_model: str = ""

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    corpus_path: Path = PROJECT_ROOT / "corpus"

    qdrant_path: Path = PROJECT_ROOT / "qdrant_data"
    qdrant_url: str = "http://localhost:6333"

    redis_url: str = "redis://localhost:6379"

    active_embedding_model: str = "bge"
    collection_name: str = "ableton_live"

    chunk_size: int = 512
    chunk_overlap: int = 64

    similarity_top_k: int = 5

    telegram_bot_token: str = ""
    api_base_url: str = "http://localhost:8000"


settings = Settings()


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """
    Конфигурация модели эмбеддингов для экспериментов.

    Attributes
    ----------
    name : str
        Имя модели.
    model_id : str
        Идентификатор модели на HuggingFace.
    dim : int
        Размерность эмбеддинга.
    query_instruction : str
        Префикс для запросов (E5 требует ``"query: "``).
    text_instruction : str
        Префикс для документов (E5 требует ``"passage: "``).
    """

    name: str
    model_id: str
    dim: int
    query_instruction: str = ""
    text_instruction: str = ""

    @property
    def collection_name(self) -> str:
        """Имя коллекции Qdrant."""

        return f"{settings.collection_name}_{self.name}"


EMBEDDING_MODELS: dict[str, EmbeddingModelConfig] = {
    "minilm": EmbeddingModelConfig(
        name="minilm",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
    ),
    "e5": EmbeddingModelConfig(
        name="e5",
        model_id="intfloat/multilingual-e5-base",
        dim=768,
        query_instruction="query: ",
        text_instruction="passage: ",
    ),
    "bge": EmbeddingModelConfig(
        name="bge",
        model_id="BAAI/bge-base-en-v1.5",
        dim=768,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
}
