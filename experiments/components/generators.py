"""
Конфигурации генераторов для экспериментов.
"""

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from llama_index.core.llms import LLM
from llama_index.core.prompts import RichPromptTemplate

from ableton_live_rag.config import settings

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
        Название генератора.
    backend : str
        Бэкенд: ``ollama``, ``openai`` или ``openai_like``.
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
        """
        Синхронная генерация ответа.

        Parameters
        ----------
        question : str
            Вопрос пользователя.
        contexts : list[str]
            Фрагменты документации.

        Returns
        -------
        str
            Ответ генератора.
        """

        return self._generate_fn(question, contexts)

    async def agenerate(self, question: str, contexts: list[str]) -> str:
        """
        Асинхронная генерация ответа.

        Parameters
        ----------
        question : str
            Вопрос пользователя.
        contexts : list[str]
            Фрагменты документации.

        Returns
        -------
        str
            Ответ генератора.
        """

        return await self._agenerate_fn(question, contexts)


def _load_generator_specs() -> list[GeneratorSpec]:
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


GENERATOR_SPECS: list[GeneratorSpec] = _load_generator_specs()


def load_judge_spec() -> GeneratorSpec:
    """
    Загрузка спецификации судьи DeepEval из переменной ``JUDGE_MODEL``.

    Returns
    -------
    GeneratorSpec
        Спецификация судьи.

    Raises
    ------
    RuntimeError
        Если ``JUDGE_MODEL`` не задан или не найден среди генераторов.
    """

    requested = os.environ.get("JUDGE_MODEL", "").strip()

    if not requested:
        raise RuntimeError("JUDGE_MODEL не задан в .env.")

    spec = next((s for s in GENERATOR_SPECS if s.name == requested), None)

    if spec is None:
        available = [s.name for s in GENERATOR_SPECS]
        raise RuntimeError(
            f"JUDGE_MODEL={requested!r} не найден среди генераторов. "
            f"Доступные: {available}"
        )

    return GeneratorSpec(
        name="judge",
        backend=spec.backend,
        model_id=spec.model_id,
        api_base=spec.api_base,
        api_key=spec.api_key,
    )


def make_llm(spec: GeneratorSpec) -> LLM:
    """
    Создание LlamaIndex LLM по спецификации.

    Parameters
    ----------
    spec : GeneratorSpec
        Спецификация генератора.

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


def _format_context(contexts: list[str]) -> str:
    return "\n\n".join(f"[Source {i + 1}]\n{c}" for i, c in enumerate(contexts))


def _make_generator(spec: GeneratorSpec) -> GeneratorConfig:
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


def build_generators(
    selected: list[str] | None = None,
) -> list[GeneratorConfig]:
    """
    Создание конфигураций генераторов для эксперимента.

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
        GENERATOR_SPECS
        if selected is None
        else [s for s in GENERATOR_SPECS if s.name in selected]
    )

    if not specs:
        raise RuntimeError("Нет доступных генераторов. Проверьте experiments/.env.")

    return [_make_generator(spec=s) for s in specs]
