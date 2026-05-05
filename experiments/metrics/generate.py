"""
Judges and metrics for evaluating generation quality.
"""

import asyncio

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from llama_index.core.llms import LLM


class LlamaIndexJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by a LlamaIndex LLM."""

    def __init__(self, llm: LLM, name: str = "judge"):
        self._llm = llm
        self._name = name
        super().__init__(model=name)

    def load_model(self) -> DeepEvalBaseLLM:
        return self

    def generate(self, prompt: str) -> str:
        return str(self._llm.complete(prompt))

    async def a_generate(self, prompt: str) -> str:
        return str(await self._llm.acomplete(prompt))

    def get_model_name(self) -> str:
        return self._name


def build_metrics(judge: DeepEvalBaseLLM) -> dict[str, BaseMetric]:
    """
    Create the metric set used to evaluate generation.

    Parameters
    ----------
    judge : DeepEvalBaseLLM
        Judge model.

    Returns
    -------
    dict[str, BaseMetric]
        Mapping ``{metric_key: metric}``.
    """

    return {
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=0.5, model=judge, async_mode=True
        ),
        "faithfulness": FaithfulnessMetric(threshold=0.5, model=judge, async_mode=True),
        "contextual_relevancy": ContextualRelevancyMetric(
            threshold=0.5, model=judge, async_mode=True
        ),
        "completeness": GEval(
            name="Completeness",
            criteria=(
                "Does the actual output completely and accurately answer the question, "
                "covering all relevant points from the retrieval context?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
            threshold=0.5,
            model=judge,
            async_mode=True,
        ),
        "conciseness": GEval(
            name="Conciseness",
            criteria=(
                "Is the actual output concise, avoiding unnecessary repetition or padding, "
                "while still fully answering the question?"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=0.5,
            model=judge,
            async_mode=True,
        ),
    }


def measure(test_case: LLMTestCase, metrics: dict[str, BaseMetric]) -> dict[str, float]:
    """
    Evaluate a single test case across all metrics.

    Parameters
    ----------
    test_case : LLMTestCase
        Test case.
    metrics : dict[str, BaseMetric]
        Metrics with their keys.

    Returns
    -------
    dict[str, float]
        Mapping ``{metric_key: score}``.
    """

    scores: dict[str, float] = {}

    for key, metric in metrics.items():
        metric.measure(test_case)
        scores[key] = float(metric.score or 0.0)

    return scores


async def ameasure(
    test_case: LLMTestCase, metrics: dict[str, BaseMetric]
) -> dict[str, float]:
    """
    Asynchronously evaluate a single test case across all metrics in parallel.

    Parameters
    ----------
    test_case : LLMTestCase
        Test case.
    metrics : dict[str, BaseMetric]
        Metrics with their keys.

    Returns
    -------
    dict[str, float]
        Mapping ``{metric_key: score}``.
    """

    await asyncio.gather(*[m.a_measure(test_case) for m in metrics.values()])

    return {key: float(m.score or 0.0) for key, m in metrics.items()}
