"""
Shared utilities for experiments.
"""

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import typer
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from llama_index.core import Document, Settings as LlamaSettings
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, NodeWithScore
from rich.console import Console

from ableton_rag.config import EMBEDDING_MODELS, EmbeddingModelConfig, settings
from experiments.components.generators import (
    GeneratorConfig,
    load_judge_spec,
    make_llm,
)
from experiments.components.rerankers import RerankerConfig
from experiments.components.retrievers import RetrieverConfig, make_embed_model
from experiments.metrics import (
    LlamaIndexJudge,
    ameasure,
    build_metrics,
    compute_relevances,
    count_total_relevant,
    hit_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ableton_rag.index import load_index, parse_nodes
from ableton_rag.ingest import load_documents

_PACKAGE_DIR = Path(__file__).resolve().parent

DATASETS_DIR = _PACKAGE_DIR / "datasets"
DATASET_PATH = DATASETS_DIR / "eval.json"
TEST_DATASET_PATH = DATASETS_DIR / "test.json"
RESULTS_DIR = _PACKAGE_DIR / "eval_results"

GENERATOR_META_KEYS: set[str] = {
    "generator",
    "description",
    "pipeline",
    "avg_latency_s",
    "errors",
    "details",
}

console = Console()


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    """
    Load a validation dataset.

    Parameters
    ----------
    path : Path
        Path to the JSON file containing questions.

    Returns
    -------
    list[dict]
        List of questions with the fields ``id``, ``question``, ``source``,
        ``ground_truth_pages`` and ``category``.
    """

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_indexes(
    models: dict | None = None,
) -> dict[str, VectorStoreIndex]:
    """
    Load Qdrant indexes for the selected embedding models.

    Parameters
    ----------
    models : dict[str, EmbeddingModelConfig] or None, optional
        Models for which to load indexes. Defaults to all models.

    Returns
    -------
    dict[str, VectorStoreIndex]
        Indexes keyed by model name.
    """

    models = models or EMBEDDING_MODELS
    indexes: dict[str, VectorStoreIndex] = {}

    for key, emb in models.items():
        console.print(f"[dim]  Loading index {emb.collection_name}...[/dim]")
        LlamaSettings.embed_model = make_embed_model(emb)
        indexes[key] = load_index(collection_name=emb.collection_name)

    return indexes


def chunking_collection_name(
    emb: EmbeddingModelConfig,
    chunk_size: int,
    overlap: int,
) -> str:
    """
    Build the Qdrant collection name for a chunking parameter set.

    Parameters
    ----------
    emb : EmbeddingModelConfig
        Embedding-model configuration.
    chunk_size : int
        Chunk size in tokens.
    overlap : int
        Chunk overlap in tokens.

    Returns
    -------
    str
        Collection name of the form ``{base}_cs{chunk_size}_co{overlap}``.
    """

    return f"{emb.collection_name}_cs{chunk_size}_co{overlap}"


def load_indexes_for_chunking(
    chunk_size: int,
    overlap: int,
    embedding_cfg: EmbeddingModelConfig,
) -> dict[str, VectorStoreIndex]:
    """
    Load Qdrant indexes for a particular chunking configuration.

    Parameters
    ----------
    chunk_size : int
        Chunk size in tokens.
    overlap : int
        Chunk overlap in tokens.
    embedding_cfg : EmbeddingModelConfig
        Embedding-model configuration.

    Returns
    -------
    dict[str, VectorStoreIndex]
        Indexes keyed by model name.

    Raises
    ------
    RuntimeError
        If the collection is missing — run ``build_chunking_indexes.py`` first.
    """

    indexes: dict[str, VectorStoreIndex] = {}

    cname = chunking_collection_name(embedding_cfg, chunk_size, overlap)

    console.print(f"[dim]  Loading index {cname}...[/dim]")

    LlamaSettings.embed_model = make_embed_model(embedding_cfg)
    indexes[embedding_cfg.name] = load_index(collection_name=cname)

    return indexes


def parse_nodes_with_config(
    documents: list[Document],
    chunk_size: int,
    overlap: int,
) -> list[BaseNode]:
    """
    Split documents into chunks using the given parameters.

    Parameters
    ----------
    documents : list[Document]
        List of documents.
    chunk_size : int
        Maximum chunk size in tokens.
    overlap : int
        Chunk overlap in tokens.

    Returns
    -------
    list[BaseNode]
        List of chunks.
    """

    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    return parser.get_nodes_from_documents(documents=documents)


def prepare_experiment() -> tuple[
    dict[str, VectorStoreIndex], list[BaseNode], list[dict]
]:
    """
    Prepare the environment for an experiment.

    Configures ``LlamaSettings``, loads indexes, parses nodes
    and loads the validation dataset.

    Returns
    -------
    tuple[dict[str, VectorStoreIndex], list[BaseNode], list[dict]]
        Indexes, nodes and dataset.
    """

    LlamaSettings.chunk_size = settings.chunk_size
    LlamaSettings.chunk_overlap = settings.chunk_overlap

    console.print("[dim]Loading indexes from Qdrant...[/dim]")
    indexes = load_indexes()

    console.print("[dim]Parsing documents into nodes...[/dim]")
    documents = load_documents()
    nodes = parse_nodes(documents=documents)
    console.print(f"[green]Got {len(nodes)} nodes[/green]")

    dataset = load_dataset()
    console.print(
        f"[green]Loaded {len(dataset)} questions from the eval dataset[/green]"
    )

    return indexes, nodes, dataset


def find_by_name(configs: list, name: str, kind: str):
    """
    Look up a configuration by name.

    Parameters
    ----------
    configs : list
        List of configurations (retrievers, rerankers, generators).
    name : str
        Name to find.
    kind : str
        Configuration kind, used in the error message.

    Returns
    -------
    Configuration with the matching name.

    Raises
    ------
    typer.Exit
        If no matching configuration is found.
    """

    found = next((c for c in configs if c.name == name), None)

    if found is None:
        available = [c.name for c in configs]
        console.print(f"[red]{kind} {name!r} not found. Available: {available}[/red]")
        raise typer.Exit(1)

    return found


def build_retrieval_pipeline(
    retriever: RetrieverConfig,
    reranker: RerankerConfig | None,
    top_k: int,
    candidate_k: int,
) -> Callable[[str], list[NodeWithScore]]:
    """
    Build a retrieval pipeline: retriever [+ reranker] → top_k nodes.

    Parameters
    ----------
    retriever : RetrieverConfig
        Base retriever.
    reranker : RerankerConfig or None
        Reranker. ``None`` disables reranking.
    top_k : int
        Number of final results.
    candidate_k : int
        Candidate-pool size for the reranker.

    Returns
    -------
    Callable[[str], list[NodeWithScore]]
        Retrieval function.
    """

    pool = candidate_k if reranker is not None else top_k

    def _retrieve(query: str) -> list[NodeWithScore]:
        candidates = retriever.retrieve(query=query, top_k=pool)

        if reranker is not None:
            return reranker.rerank(query=query, nodes=candidates, top_k=top_k)

        return candidates[:top_k]

    return _retrieve


def evaluate_dataset(
    retrieve_fn: Callable[[str], list[NodeWithScore]],
    dataset: list[dict],
) -> tuple[list[dict], float]:
    """
    Evaluate a retrieval function on the dataset.

    Parameters
    ----------
    retrieve_fn : Callable[[str], list[NodeWithScore]]
        Retrieval function: takes a query, returns nodes.
    dataset : list[dict]
        Validation dataset.

    Returns
    -------
    tuple[list[dict], float]
        Per-question results and total elapsed time.
    """

    per_question: list[dict] = []
    total_time = 0.0

    for item in dataset:
        t0 = time.perf_counter()

        try:
            nodes = retrieve_fn(item["question"])
        except Exception as e:
            console.print(f"[red]  Error on '{item['id']}': {e}[/red]")
            per_question.append({"id": item["id"], "error": str(e), "relevances": []})
            continue

        elapsed = time.perf_counter() - t0
        total_time += elapsed

        retrieved = [
            (n.metadata.get("source", ""), n.metadata.get("page_start", 0))
            for n in nodes
        ]

        gt = item["ground_truth_pages"]
        gt_source = item["source"]

        rels = compute_relevances(
            retrieved=retrieved,
            ground_truth_source=gt_source,
            ground_truth_ranges=gt,
        )

        seen: set[tuple[str, int]] = set()
        dedup_rels: list[bool] = []

        for (s, p), rel in zip(retrieved, rels):
            if (s, p) not in seen:
                seen.add((s, p))
                dedup_rels.append(rel)

        per_question.append(
            {
                "id": item["id"],
                "relevances": rels,
                "dedup_relevances": dedup_rels,
                "retrieved": [f"{s}:p{p}" for s, p in retrieved],
                "total_relevant": count_total_relevant(ground_truth_ranges=gt),
                "latency_s": round(elapsed, 3),
            }
        )

    return per_question, total_time


def aggregate_retrieval_metrics(per_question: list[dict], total_time: float) -> dict:
    """
    Aggregate retrieval metrics from per-question results.

    Parameters
    ----------
    per_question : list[dict]
        Per-question results from ``evaluate_dataset()``.
    total_time : float
        Total elapsed time.

    Returns
    -------
    dict
        Dictionary with metrics and a ``details`` field.
    """

    valid = [q for q in per_question if "error" not in q]
    n = len(valid) or 1

    return {
        "hit_rate": round(sum(hit_rate(q["relevances"]) for q in valid) / n, 3),
        "mrr": round(sum(mrr(q["relevances"]) for q in valid) / n, 3),
        "precision": round(sum(precision_at_k(q["relevances"]) for q in valid) / n, 3),
        "recall": round(
            sum(recall_at_k(q["dedup_relevances"], q["total_relevant"]) for q in valid)
            / n,
            3,
        ),
        "ndcg": round(
            sum(ndcg_at_k(q["dedup_relevances"], q["total_relevant"]) for q in valid)
            / n,
            3,
        ),
        "avg_latency_s": round(total_time / n, 3),
        "errors": sum(1 for q in per_question if "error" in q),
        "details": per_question,
    }


def format_retrieval_summary(result: dict) -> str:
    """
    Format a one-line summary of retrieval results.

    Parameters
    ----------
    result : dict
        Result from ``aggregate_retrieval_metrics()``.

    Returns
    -------
    str
        String of the form ``Hit Rate=0.xxx  MRR=0.xxx  NDCG=0.xxx  (0.xxxs/query)``.
    """

    return (
        f"  Hit Rate={result['hit_rate']:.3f}  "
        f"MRR={result['mrr']:.3f}  "
        f"NDCG={result['ndcg']:.3f}  "
        f"({result['avg_latency_s']:.3f}s/query)"
    )


def col_header(key: str) -> str:
    """
    Convert a metric key into a table column header.

    Parameters
    ----------
    key : str
        Metric key.

    Returns
    -------
    str
        Header.
    """

    return key.replace("_", " ").title()


def format_generator_summary(result: dict) -> str:
    """
    Format a one-line summary of generator results.

    Parameters
    ----------
    result : dict
        Result with generator metrics.

    Returns
    -------
    str
        String of the form ``Metric1=0.xxx  Metric2=0.xxx  (0.xxxs/query)``.
    """

    score_keys = [k for k in result if k not in GENERATOR_META_KEYS]
    parts = [f"{col_header(k)}={result[k]:.3f}" for k in score_keys]

    return "  " + "  ".join(parts) + f"  ({result['avg_latency_s']:.3f}s/query)"


async def evaluate_generator(
    generator: GeneratorConfig,
    retrieve_fn: Callable[[str], list[NodeWithScore]],
    metrics: dict[str, BaseMetric],
    dataset: list[dict],
    concurrency: int = 16,
) -> dict:
    """
    End-to-end evaluation of a generator on the dataset.

    Parameters
    ----------
    generator : GeneratorConfig
        Answer generator.
    retrieve_fn : Callable[[str], list[NodeWithScore]]
        Function used to retrieve context.
    metrics : dict[str, BaseMetric]
        DeepEval metrics.
    dataset : list[dict]
        Validation dataset.
    concurrency : int
        Number of parallel requests.

    Returns
    -------
    dict
        Aggregated metrics with per-question ``details``.
    """

    semaphore = asyncio.Semaphore(concurrency)

    async def _process(item: dict) -> dict:
        async with semaphore:
            question = item["question"]

            try:
                t0 = time.perf_counter()
                nodes = retrieve_fn(question)
                contexts = [n.node.get_content() for n in nodes]
                answer = await generator.agenerate(question=question, contexts=contexts)
                latency = time.perf_counter() - t0

                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=contexts,
                )
                scores = await ameasure(test_case=test_case, metrics=metrics)

                return {"id": item["id"], "latency_s": round(latency, 3), **scores}

            except Exception as e:
                console.print(f"[red]  Error on '{item['id']}': {e}[/red]")

                return {"id": item["id"], "error": str(e)}

    per_question: list[dict] = list(
        await asyncio.gather(*[_process(i) for i in dataset])
    )

    valid = [q for q in per_question if "error" not in q]
    n = len(valid) or 1
    errors = len(per_question) - len(valid)
    score_keys = [
        k for k in (valid[0] if valid else {}) if k not in ("id", "latency_s")
    ]

    return {
        "generator": generator.name,
        "description": generator.description,
        **{
            key: round(sum(q.get(key, 0.0) for q in valid) / n, 3) for key in score_keys
        },
        "avg_latency_s": round(sum(q.get("latency_s", 0.0) for q in valid) / n, 3),
        "errors": errors,
        "details": per_question,
    }


def build_judge_metrics() -> dict[str, BaseMetric]:
    """
    Initialize the DeepEval judge and the set of generation metrics.

    Returns
    -------
    dict[str, BaseMetric]
        DeepEval metrics, ready to use.
    """

    judge_spec = load_judge_spec()
    console.print(
        f"[dim]Initializing DeepEval judge "
        f"({judge_spec.backend}/{judge_spec.model_id})...[/dim]"
    )
    judge = LlamaIndexJudge(llm=make_llm(judge_spec), name=judge_spec.name)

    return build_metrics(judge=judge)


def save_results(results: list[dict] | dict, results_dir: Path) -> Path:
    """
    Save results as JSON.

    Parameters
    ----------
    results : list[dict] or dict
        Experiment results.
    results_dir : Path
        Output directory.

    Returns
    -------
    Path
        Path to the saved file.
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    json_path = results_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"[green]Results saved: {json_path}[/green]")

    return json_path
