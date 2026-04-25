"""
Конфигурации больших языковых моделей для экспериментов.
"""

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate

from ableton_live_rag.config import settings

_ROOT_DOTENV = Path(__file__).resolve().parent.parent.parent / ".env"
_LOCAL_DOTENV = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=_ROOT_DOTENV)
load_dotenv(dotenv_path=_LOCAL_DOTENV, override=True)

_OPENAI_LIKE_PREFIXES: list[tuple[str, str]] = [
    ("GPT_OSS", "gpt-oss"),
    ("GLM", "glm-4.7"),
    ("GEMMA", "gemma-4"),
]

_PROMPT = RichPromptTemplate(
    """\
    You are an expert assistant for Ableton ecosystem. Answer the question based \
    strictly on the provided documentation excerpts.

    Rules:
    - Use only the context below. If it is insufficient, say so.
    - Be concise, precise, and practical.

    Context:
    ---------------------
    {{ context_str }}
    ---------------------

    Question: {{ query_str }}
    Answer: \
    """
)


@dataclass(frozen=True)
class GeneratorSpec:
    """
    Спецификация генератора.

    Attributes
    ----------
    name : str
        Имя.
    backend : str
        Бэкенд: ``"ollama"``, ``"openai"`` или ``"openai_like"``.
    model_id : str
        Идентификатор модели.
    api_base : str
        URL API.
    api_key : str
        Ключ API.
    """

    name: str
    backend: str
    model_id: str
    api_base: str = ""
    api_key: str = ""


@dataclass
class GeneratorConfig:
    """
    Обёртка над генератором с единым интерфейсом для оценки.

    Attributes
    ----------
    name : str
        Название генератора.
    description : str
        Описание генератора.
    """

    name: str
    description: str
    _generate_fn: Callable[[str, list[str]], str] = field(repr=False)
    _agenerate_fn: Callable[[str, list[str]], Awaitable[str]] = field(repr=False)

    def generate(self, question: str, contexts: list[str]) -> str:
        return self._generate_fn(question, contexts)

    async def agenerate(self, question: str, contexts: list[str]) -> str:
        return await self._agenerate_fn(question, contexts)


def _load_generator_models() -> list[GeneratorSpec]:
    """
    Построение списка спецификаций из переменных окружения.

    Returns
    -------
    list[GeneratorSpec]
        Список спецификаций генераторов.
    """

    specs: list[GeneratorSpec] = []

    if os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODELS"):
        for model_id in os.environ["OPENAI_MODELS"].split(","):
            model_id = model_id.strip()

            if model_id:
                specs.append(
                    GeneratorSpec(name=model_id, backend="openai", model_id=model_id)
                )

    for prefix, name in _OPENAI_LIKE_PREFIXES:
        base_url = os.environ.get(f"{prefix}_BASE_URL")

        if base_url:
            specs.append(
                GeneratorSpec(
                    name=name,
                    backend="openai_like",
                    model_id=os.environ.get(f"{prefix}_MODEL", name),
                    api_base=base_url,
                    api_key=os.environ.get(f"{prefix}_API_KEY", ""),
                )
            )

    return specs


GENERATOR_MODELS: list[GeneratorSpec] = _load_generator_models()


def _load_judge_spec() -> GeneratorSpec:
    requested = os.environ.get("JUDGE_MODEL", "").strip()

    if not requested:
        raise RuntimeError("JUDGE_MODEL не задан в .env.")

    spec = next((s for s in GENERATOR_MODELS if s.name == requested), None)

    if spec is None:
        available = [s.name for s in GENERATOR_MODELS]
        raise RuntimeError(
            f"JUDGE_MODEL={requested!r} не найден в GENERATOR_MODELS. "
            f"Доступные: {available}"
        )

    return GeneratorSpec(
        name="judge",
        backend=spec.backend,
        model_id=spec.model_id,
        api_base=spec.api_base,
        api_key=spec.api_key,
    )


JUDGE_SPEC: GeneratorSpec = _load_judge_spec()


def make_llm(spec: GeneratorSpec) -> LLM:
    """
    Создание LlamaIndex LLM по спецификации.

    Parameters
    ----------
    spec : GeneratorSpec
        Спецификация модели.

    Returns
    -------
    LLM
        LlamaIndex LLM одного из трёх бэкендов.
    """

    if spec.backend == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=spec.model_id,
            temperature=0.7,
            base_url=spec.api_base or settings.ollama_base_url,
            request_timeout=settings.ollama_request_timeout,
        )

    if spec.backend == "openai":
        from llama_index.llms.openai import OpenAI

        return OpenAI(
            model=spec.model_id,
            temperature=0.7,
            api_key=spec.api_key or None,
            timeout=120.0,
        )

    if spec.backend == "openai_like":
        from llama_index.llms.openai_like import OpenAILike

        return OpenAILike(
            model=spec.model_id,
            temperature=0.7,
            api_base=spec.api_base,
            api_key=spec.api_key,
            is_chat_model=True,
            timeout=120.0,
        )

    raise ValueError(f"Неизвестный бэкенд: {spec.backend!r}")


def _make_generator(spec: GeneratorSpec) -> GeneratorConfig:
    """
    Создание GeneratorConfig по спецификации.

    Parameters
    ----------
    spec : GeneratorSpec
        Спецификация модели.

    Returns
    -------
    GeneratorConfig
        Конфигурация генератора с привязанной LLM.
    """

    def _format_context(contexts: list[str]) -> str:
        """
        Форматирование списка фрагментов с нумерацией источников.

        Parameters
        ----------
        contexts : list[str]
            Фрагменты документации.

        Returns
        -------
        str
            Строка вида ``[Source 1]\n...\n\n[Source 2]\n...``.
        """

        return "\n\n".join(f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts))

    llm = make_llm(spec)

    def _generate(question: str, contexts: list[str]) -> str:
        prompt = _PROMPT.format(
            context_str=_format_context(contexts), query_str=question
        )
        return str(llm.complete(prompt))

    async def _agenerate(question: str, contexts: list[str]) -> str:
        prompt = _PROMPT.format(
            context_str=_format_context(contexts), query_str=question
        )
        return str(await llm.acomplete(prompt))

    return GeneratorConfig(
        name=spec.name,
        description=f"{spec.backend} / {spec.model_id}",
        _generate_fn=_generate,
        _agenerate_fn=_agenerate,
    )


def build_all_generators(
    selected: list[str] | None = None,
) -> list[GeneratorConfig]:
    """
    Создание всех генераторов для эксперимента.

    Parameters
    ----------
    selected : list[str] or None, optional
        Имена спецификаций для загрузки. ``None`` — все доступные.

    Returns
    -------
    list[GeneratorConfig]
        Список конфигураций генераторов.
    """

    specs = (
        GENERATOR_MODELS
        if selected is None
        else [s for s in GENERATOR_MODELS if s.name in selected]
    )

    if not specs:
        raise RuntimeError(
            "Нет доступных генераторов. Проверьте experiments/generator/.env."
        )

    return [_make_generator(spec=s) for s in specs]
