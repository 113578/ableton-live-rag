"""
Конфигурация LlamaIndex для работы с LLM.
"""

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ableton_live_rag.config import LLMProvider, settings


def setup() -> None:
    """
    Настройка параметров LlamaIndex.
    """

    LlamaIndexSettings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
    )

    LlamaIndexSettings.chunk_size = settings.chunk_size
    LlamaIndexSettings.chunk_overlap = settings.chunk_overlap

    if settings.llm_provider == LLMProvider.ollama:
        _setup_ollama()
    elif settings.llm_provider == LLMProvider.vllm:
        _setup_vllm()


def _setup_ollama() -> None:
    """Подключение локальной модели через Ollama."""

    from llama_index.llms.ollama import Ollama

    LlamaIndexSettings.llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        request_timeout=settings.ollama_request_timeout,
    )


def _setup_vllm() -> None:
    """Подключение модели через vLLM (OpenAI-совместимый API)."""

    from llama_index.llms.openai import OpenAI

    LlamaIndexSettings.llm = OpenAI(
        model=settings.vllm_model,
        api_key=settings.vllm_api_key,
        api_base=settings.vllm_url_base,
    )
