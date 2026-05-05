"""
Project configuration.
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
    Supported LLM providers.
    """

    ollama = "ollama"
    openai = "openai"
    vllm = "vllm"


class Settings(BaseSettings):
    """
    Project configuration.

    All parameters can be overridden via environment variables
    or a ``.env`` file at the repository root.

    Attributes
    ----------
    llm_provider : LLMProvider
        LLM provider (``ollama``, ``openai`` or ``vllm``).
    request_timeout : int
        Ollama request timeout in seconds.
    temperature : float
        Generation temperature.
    ollama_base_url : str
        Base URL of the Ollama server.
    ollama_model : str
        Ollama model name.
    openai_api_key : str
        OpenAI API key.
    openai_model : str
        OpenAI model name.
    vllm_url_base : str
        Base URL of the vLLM server (OpenAI-compatible API).
    vllm_api_key : str
        API key for vLLM.
    vllm_model : str
        vLLM model name.
    guard_model : str
        Model name for guardrails. If empty, the main LLM is used.
    embedding_model : str
        Embedding model identifier (for the main pipeline).
    embedding_dim : int
        Embedding dimension of the main model.
    corpus_path : Path
        Path to the directory containing PDF files of the corpus.
    qdrant_path : Path
        Path to the local Qdrant storage directory (used only when
        ``qdrant_url`` is empty).
    qdrant_url : str
        URL for the Qdrant client.
    redis_url : str
        URL for the Redis client.
    collection_name : str
        Base name of the Qdrant collection.
    chunk_size : int
        Maximum chunk size in tokens.
    chunk_overlap : int
        Overlap between adjacent chunks in tokens.
    context_window : int
        Size of the LLM context window in tokens.
    num_output : int
        Maximum number of tokens in the LLM response.
    similarity_top_k : int
        Number of fragments returned by the default search component.
    telegram_bot_token : str
        Telegram bot token.
    api_base_url : str
        URL of the FastAPI application.
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

    active_embedding_model: str = "e5"
    collection_name: str = "ableton"

    chunk_size: int = 512
    chunk_overlap: int = 64

    similarity_top_k: int = 5

    telegram_bot_token: str = ""
    api_base_url: str = "http://localhost:8000"


settings = Settings()


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """
    Embedding-model configuration for experiments.

    Attributes
    ----------
    name : str
        Model name.
    model_id : str
        HuggingFace model identifier.
    dim : int
        Embedding dimension.
    query_instruction : str
        Prefix for queries (E5 requires ``"query: "``).
    text_instruction : str
        Prefix for documents (E5 requires ``"passage: "``).
    """

    name: str
    model_id: str
    dim: int
    query_instruction: str = ""
    text_instruction: str = ""

    @property
    def collection_name(self) -> str:
        """Qdrant collection name."""

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
