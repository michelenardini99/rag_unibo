import concurrent.futures
import json
import statistics
import threading
import time
from pathlib import Path

import openai
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from generation.condense import condense_question, decompose_question
from generation.generator import generate_response
from generation.hyde import hyde_query
from generation.prompt import question_for_answer
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
from retrieval.query_filters import course_tipo_laurea_map, extract_query_filters, known_values_for_fields
from retrieval.retriever import retrieve_decomposed

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
              retrieval_kwargs: dict, retrieval_lock: threading.Lock,
              known_values: dict[str, list[str]], tipo_laurea_by_corso: dict[str, str]) -> tuple[dict, dict]:
    """Runs one test case through condense -> retrieve -> generate.

    Returns:
        tuple[dict, dict]: (ragas input row, timing dict for this case).
    """
    query = case["question"]
    history = [tuple(turn) for turn in case.get("history", [])]

    search_query = condense_question(llm, history, query)
    sub_queries = decompose_question(llm, search_query)
    # As in app.py: the course named by the student, else the one the
    # conversation is about (the session's default_corso, here the case's
    # "corso"); search the university-wide pages plus that course's.
    student_text = " ".join([past for past, _ in history] + [query])
    priority_corso = (extract_query_filters(llm, search_query, known_values, tipo_laurea_by_corso, student_text).get("corso")
                      or case.get("corso"))
    retrieval_kwargs = {"priority_corso": priority_corso, "scope": [priority_corso] if priority_corso else [],
                        **retrieval_kwargs}

    def retrieve_locked(queries: list[str]) -> list[dict]:
        # embed/reranker share HuggingFace fast tokenizers, which mutate
        # internal truncation/padding state and aren't safe to call
        # concurrently from multiple threads (raises "Already borrowed", or
        # segfaults outright depending on timing). Only condense/generate
        # (HTTP round trips to the batching vLLM server) actually benefit
        # from running in parallel, so retrieval stays serialized here.
        with retrieval_lock:
            return retrieve_decomposed(queries, client, collection_name, embed, reranker, docstore, **retrieval_kwargs)

    chunks = retrieve_locked(sub_queries)
    if not chunks and priority_corso:
        # As in app.py: retry with a HyDE query, only with a course.
        chunks = retrieve_locked([hyde_query(llm, search_query)])
    t0 = time.perf_counter()
    # search_query (condensed), not the raw query: it's already a
    # self-contained question consistent with what was actually
    # retrieved, so the model isn't left resolving "e per il 2026/27?"
    # against history on its own during generation.
    response = generate_response(llm, question_for_answer(search_query, sub_queries), chunks, history or None,
                                 corso=priority_corso)
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
    known_values = known_values_for_fields(client, collection_name)
    tipo_laurea_by_corso = course_tipo_laurea_map(client, collection_name, known_values.get("corso") or [])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda case: _run_case(
                case, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs, retrieval_lock,
                known_values, tipo_laurea_by_corso,
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


# The judge's answers are short JSON (statements, verdicts): 2048 leaves the
# prompt (question + retrieved contexts + ragas instructions) 6144 of the
# 8192-token window; 4096 made the longest contexts fail ("maximum context
# length"), dropping exactly the multi-part and synthesis cases from the means.
JUDGE_MAX_TOKENS = 2048


def generate(
    dataset: list[dict],
    llm: OpenAILike,
    embed: BGEM3FlagModel,
    client: QdrantClient,
    reranker: FlagReranker,
    docstore: SimpleDocumentStore,
    output_path: Path,
    model_id: str,
    collection_name: str = "ateneo_docs",
    retrieval_kwargs: dict | None = None,
    max_workers: int = 8,
) -> None:
    """Runs every test case through condense -> retrieve -> generate and
    saves question, contexts, answer and timing, for `judge()` to score later.

    Kept apart from judging so all models can be scored by the same judge
    (vLLM serves one model at a time) and answers can be re-judged or read
    without generating them again.

    Args:
        dataset (list[dict]): Test cases to evaluate.
        llm (OpenAILike): Model under evaluation (condensation and generation).
        embed (BGEM3FlagModel): Embedding model for retrieval.
        client (QdrantClient): Qdrant client.
        reranker (FlagReranker): Cross-encoder reranker.
        docstore (SimpleDocumentStore): Docstore backing the retriever.
        output_path (Path): Where to write the generations JSON.
        model_id (str): Model served by vLLM, recorded in the output.
        collection_name (str): Qdrant collection to search.
        retrieval_kwargs (dict | None): Extra kwargs forwarded to `retrieve()`.
        max_workers (int): Test cases run concurrently, see `build_eval_dataset`.
    """
    rows, timings = build_eval_dataset(
        dataset, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs, max_workers=max_workers,
    )
    cases = [
        {"id": case["id"], **({"diagnostic_tag": case["diagnostic_tag"]} if "diagnostic_tag" in case else {}), **row}
        for case, row in zip(dataset, rows)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"model": model_id, "cases": cases, "timing": timing_summary(timings)},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
    t = timing_summary(timings)
    print(f"Risposte salvate in {output_path} — generazione media: {t['generate_s']['mean']}, "
          f"mediana: {t['generate_s']['median']}")


def judge(
    generations_path: Path,
    judge_base_url: str,
    judge_model: str,
    judge_model_id: str,
    embed: BGEM3FlagModel,
    output_path: Path,
    ragas_max_workers: int = 32,
):
    """Scores the answers saved by `generate()` with ragas and writes the results.

    Args:
        generations_path (Path): Output of `generate()`.
        judge_base_url (str): OpenAI-compatible endpoint serving the judge.
        judge_model (str): Model name to request from that endpoint.
        judge_model_id (str): The judge's actual model, recorded in the results.
        embed (BGEM3FlagModel): Embedding model for ragas' similarity metrics.
        output_path (Path): Where to write the results JSON.
        ragas_max_workers (int): Concurrency for the ragas judge
            (`RunConfig.max_workers`, default 16 in ragas): pure LLM/embedding
            calls against the vLLM server, which batches them.

    Returns:
        The ragas EvaluationResult object.
    """
    generations = json.loads(generations_path.read_text(encoding="utf-8"))
    cases = generations["cases"]
    rows = [{k: c[k] for k in ("user_input", "retrieved_contexts", "response", "reference")} for c in cases]

    openai_client = openai.OpenAI(base_url=judge_base_url, api_key="EMPTY")
    ragas_llm = llm_factory(judge_model, provider="openai", client=openai_client, max_tokens=JUDGE_MAX_TOKENS,
                            temperature=0.0)
    ragas_emb = BGEM3RagasEmbeddings(embed)

    metrics = RETRIEVING_METRICS + GENERATIVE_METRICS + END_TO_END_METRICS
    res = ragas_evaluate(
        dataset=Dataset.from_list(rows),
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        show_progress=True,
        run_config=RunConfig(max_workers=ragas_max_workers),
    )
    df = res.to_pandas()

    results = {"model": generations["model"], "judge": judge_model_id,
               **results_to_json(cases, df, generations["timing"]["per_case"])}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Risultati salvati in {output_path} (modello {generations['model']}, giudice {judge_model_id})")
    for metric in metrics:
        print(f"{metric.name}: {df[metric.name].mean():.2f}")
    return res
