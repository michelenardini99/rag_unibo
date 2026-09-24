import concurrent.futures
import json
import statistics
import threading
import time
from pathlib import Path

import openai
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from generation.condense import condense_question
from generation.generator import generate_response
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.openai_like import OpenAILike
from qdrant_client import QdrantClient
from ragas import evaluate as ragas_evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity,
)
from ragas.run_config import RunConfig
from retrieval.retriever import retrieve

RETRIEVING_METRICS = [context_precision, context_recall]
GENERATIVE_METRICS = [faithfulness, answer_relevancy]
END_TO_END_METRICS = [answer_correctness, answer_similarity]


class BGEM3RagasEmbeddings(BaseRagasEmbeddings):
    """Adapts our BGE-M3 model to ragas' embeddings interface, reusing the
    instance already loaded for retrieval instead of loading a second embedder.
    """

    def __init__(self, model: BGEM3FlagModel):
        """
        Args:
            model (BGEM3FlagModel): Loaded BGE-M3 model to wrap.
        """
        super().__init__()
        self.embed_model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        output = self.embed_model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"].tolist()

    def embed_query(self, text: str) -> list[float]:
        output = self.embed_model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def _run_case(case: dict, llm: OpenAILike, embed: BGEM3FlagModel, client: QdrantClient,
              reranker: FlagReranker, docstore: SimpleDocumentStore, collection_name: str,
              retrieval_kwargs: dict, retrieval_lock: threading.Lock) -> tuple[dict, dict]:
    """Runs one test case through condense -> retrieve -> generate.

    Returns:
        tuple[dict, dict]: (ragas input row, timing dict for this case).
    """
    query = case["question"]
    history = [tuple(turn) for turn in case.get("history", [])]

    search_query = condense_question(llm, history, query)
    with retrieval_lock:
        # embed/reranker share HuggingFace fast tokenizers, which mutate
        # internal truncation/padding state and aren't safe to call
        # concurrently from multiple threads (raises "Already borrowed", or
        # segfaults outright depending on timing). Only condense/generate
        # (HTTP round trips to the batching vLLM server) actually benefit
        # from running in parallel, so retrieval stays serialized here.
        chunks = retrieve(search_query, client, collection_name, embed, reranker, docstore, **retrieval_kwargs)
    t0 = time.perf_counter()
    # search_query (condensed), not the raw query: it's already a
    # self-contained question consistent with what was actually
    # retrieved, so the model isn't left resolving "e per il 2026/27?"
    # against history on its own during generation.
    response = generate_response(llm, search_query, chunks, history or None)
    t1 = time.perf_counter()

    row = {
        "user_input": query,
        "retrieved_contexts": [c["text"] for c in chunks] or [""],
        "response": response,
        "reference": case["expected_answer"],
    }
    timing = {
        "id": case["id"],
        "generate_s": round(t1 - t0, 3),
    }
    return row, timing


def build_eval_dataset(dataset: list[dict], llm: OpenAILike, embed: BGEM3FlagModel, client: QdrantClient,
                        reranker: FlagReranker, docstore: SimpleDocumentStore, collection_name: str,
                        retrieval_kwargs: dict | None = None, max_workers: int = 8) -> tuple[list[dict], list[dict]]:
    """Runs each test case through condense -> retrieve -> generate and
    collects both the ragas input rows and per-case generation timing.

    Args:
        dataset (list[dict]): Test cases, each with "question", "expected_answer"
            and optionally "history".
        llm (OpenAILike): Model used for both condensation and generation.
        embed (BGEM3FlagModel): Embedding model for retrieval.
        client (QdrantClient): Qdrant client.
        reranker (FlagReranker): Cross-encoder reranker.
        docstore (SimpleDocumentStore): Docstore backing the retriever.
        collection_name (str): Qdrant collection to search.
        retrieval_kwargs (dict | None): Extra kwargs forwarded to `retrieve()`.
        max_workers (int): Test cases run concurrently, each doing its own
            condense/retrieve/generate round trip. The condense and generate
            calls hit the shared vLLM server, which continuous-batches
            concurrent requests, so running cases in parallel overlaps
            network/generation latency; the retrieval step (embed + rerank)
            stays serialized regardless (see `_run_case`), since those models'
            tokenizers aren't safe to call concurrently.

    Returns:
        tuple[list[dict], list[dict]]: (rows for ragas, per-case timings), in
        the same order as `dataset`.
        Timings cover only the generation call — not condensation/retrieval,
        and not the ragas judge (that's eval overhead, not user-facing latency).
    """
    retrieval_kwargs = retrieval_kwargs or {}
    retrieval_lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda case: _run_case(
                case, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs, retrieval_lock,
            ),
            dataset,
        ))
    rows, timings = (list(x) for x in zip(*results)) if results else ([], [])
    return rows, timings


def timing_summary(timings: list[dict]) -> dict:
    """Aggregates per-case timings into mean/median/p95.

    Args:
        timings (list[dict]): Per-case timing dicts from `build_eval_dataset`.

    Returns:
        dict: "generate_s" stats plus the raw "per_case" list.
    """
    def stats(key: str) -> dict:
        values = [t[key] for t in timings]
        return {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0], 3),
        }

    return {
        "generate_s": stats("generate_s"),
        "per_case": timings,
    }


def results_to_json(dataset: list[dict], df, timings: list[dict]) -> dict:
    """Assembles the final results JSON from ragas scores and timings.

    Args:
        dataset (list[dict]): Test cases, in the same order as `df`'s rows.
        df: Pandas DataFrame returned by `ragas_evaluate(...).to_pandas()`.
        timings (list[dict]): Per-case timings from `build_eval_dataset`.

    Returns:
        dict: "retrieving"/"generative"/"end_to_end" metric blocks plus "timing".
    """
    def category_block(metrics: list) -> dict:
        names = [m.name for m in metrics]
        return {
            "mean": {name: round(float(df[name].mean()), 4) for name in names},
            "per_case": [
                {
                    "id": case["id"],
                    **({"diagnostic_tag": case["diagnostic_tag"]} if "diagnostic_tag" in case else {}),
                    **{name: round(float(df.iloc[i][name]), 4) for name in names},
                }
                for i, case in enumerate(dataset)
            ],
        }

    return {
        "retrieving": category_block(RETRIEVING_METRICS),
        "generative": category_block(GENERATIVE_METRICS),
        "end_to_end": category_block(END_TO_END_METRICS),
        "timing": timing_summary(timings),
    }


def evaluate(
    dataset: list[dict],
    llm: OpenAILike,
    embed: BGEM3FlagModel,
    client: QdrantClient,
    reranker: FlagReranker,
    docstore: SimpleDocumentStore,
    collection_name: str = "ateneo_docs",
    retrieval_kwargs: dict | None = None,
    output_path: Path = Path("datasets/eval/results.json"),
    max_workers: int = 8,
    ragas_max_workers: int = 32,
):
    """Runs the full RAGAS evaluation for a dataset and writes results to disk.

    Args:
        dataset (list[dict]): Test cases to evaluate.
        llm (OpenAILike): Model used for condensation, generation, and as the
            ragas judge.
        embed (BGEM3FlagModel): Embedding model for retrieval and ragas similarity.
        client (QdrantClient): Qdrant client.
        reranker (FlagReranker): Cross-encoder reranker.
        docstore (SimpleDocumentStore): Docstore backing the retriever.
        collection_name (str): Qdrant collection to search.
        retrieval_kwargs (dict | None): Extra kwargs forwarded to `retrieve()`.
        output_path (Path): Where to write the results JSON.
        max_workers (int): Test cases run concurrently in the condense/
            retrieve/generate pipeline, see `build_eval_dataset`.
        ragas_max_workers (int): Concurrency for the ragas judge itself
            (`RunConfig.max_workers`, default 16 in ragas) — these are pure
            LLM/embedding calls against the same vLLM server, so they can
            usually afford more concurrency than `max_workers`.

    Returns:
        The ragas EvaluationResult object.
    """
    rows, timings = build_eval_dataset(
        dataset, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs, max_workers=max_workers,
    )

    openai_client = openai.OpenAI(base_url=llm.api_base, api_key="EMPTY")

    ragas_llm = llm_factory(llm.model, provider="openai", client=openai_client, max_tokens=4096, temperature=0-0)
    ragas_emb = BGEM3RagasEmbeddings(embed)

    final_dataset = Dataset.from_list(rows)

    metrics = RETRIEVING_METRICS + GENERATIVE_METRICS + END_TO_END_METRICS

    res = ragas_evaluate(
        dataset=final_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        show_progress=True,
        run_config=RunConfig(max_workers=ragas_max_workers),
    )
    df = res.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_to_json(dataset, df, timings), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Risultati salvati in {output_path}")

    for metric in metrics:
        print(f"{metric.name}: {df[metric.name].mean():.2f}")

    t = timing_summary(timings)
    print(f"  di cui generazione — media: {t['generate_s']['mean']}, mediana: {t['generate_s']['median']}")
    return res
