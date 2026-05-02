"""
Конфигурация LlamaIndex для работы с LLM.
"""

from llama_index.core import Settings as LlamaIndexSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ableton_live_rag.config import LLMProvider, get_logger, settings

logger = get_logger(__name__)


def setup() -> None:
    """
    Настройка параметров LlamaIndex.
    """

    logger.info("Setting up LLM provider: %s", settings.llm_provider.value)

    LlamaIndexSettings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embedding_model,
    )

    LlamaIndexSettings.chunk_size = settings.chunk_size
    LlamaIndexSettings.chunk_overlap = settings.chunk_overlap
    LlamaIndexSettings.context_window = settings.context_window
    LlamaIndexSettings.num_output = settings.num_output

    if settings.llm_provider == LLMProvider.ollama:
        _setup_ollama()
    elif settings.llm_provider == LLMProvider.openai:
        _setup_openai()
    elif settings.llm_provider == LLMProvider.vllm:
        _setup_vllm()


def _setup_ollama() -> None:
    from llama_index.llms.ollama import Ollama

    logger.info(
        "Ollama: model=%s url=%s", settings.ollama_model, settings.ollama_base_url
    )

    LlamaIndexSettings.llm = Ollama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        context_window=settings.context_window,
        request_timeout=settings.request_timeout,
    )


def _setup_openai() -> None:
    from llama_index.llms.openai import OpenAI

    logger.info("OpenAI: model=%s", settings.openai_model)

    LlamaIndexSettings.llm = OpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
        max_tokens=settings.num_output,
        timeout=settings.request_timeout,
    )


def _setup_vllm() -> None:
    from llama_index.llms.openai_like import OpenAILike

    logger.info("vLLM: model=%s url=%s", settings.vllm_model, settings.vllm_url_base)

    LlamaIndexSettings.llm = OpenAILike(
        model=settings.vllm_model,
        temperature=settings.temperature,
        api_key=settings.vllm_api_key,
        api_base=settings.vllm_url_base,
        context_window=settings.context_window,
        max_tokens=settings.num_output,
        is_chat_model=True,
        timeout=settings.request_timeout,
    )
