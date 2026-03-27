"""
Конфигурации проекта.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LLMProvider(str, Enum):
    """
    Поддерживаемые провайдеры LLM.
    """

    ollama = "ollama"
    vllm = "vllm"


class Settings(BaseSettings):
    """
    Конфигурация проекта.

    Все параметры можно переопределить через переменные окружения
    или файл ``.env`` в корне репозитория.

    Attributes
    ----------
    llm_provider : LLMProvider
        Провайдер LLM (``ollama`` или ``vllm``).
    ollama_base_url : str
        Базовый URL сервера Ollama.
    ollama_model : str
        Имя модели Ollama.
    ollama_request_timeout : int
        Таймаут запроса к Ollama в секундах.
    vllm_url_base : str
        Базовый URL vLLM-сервера (OpenAI-совместимый API).
    vllm_api_key : str
        API-ключ для vLLM.
    vllm_model : str
        Имя модели vLLM.
    embedding_model : str
        Идентификатор модели эмбеддингов (для основного пайплайна).
    embedding_dim : int
        Размерность эмбеддинга основной модели.
    corpus_path : Path
        Путь к PDF-файлу корпуса.
    qdrant_path : Path
        Путь к директории с хранилищем Qdrant.
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
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = LLMProvider.ollama

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5"
    ollama_request_timeout: int = 120

    vllm_url_base: str = "http://localhost:9999"
    vllm_api_key: str = "vllm-api-key"
    vllm_model: str = "gpt-oss"

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    corpus_path: Path = PROJECT_ROOT / "corpus.pdf"
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"

    collection_name: str = "ableton_live_docs"

    chunk_size: int = 512
    chunk_overlap: int = 64

    context_window: int = 32768
    num_output: int = 4096

    similarity_top_k: int = 5


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
