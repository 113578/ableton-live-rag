"""
Конфигурации проекта.
"""

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

    similarity_top_k: int = 5


settings = Settings()
