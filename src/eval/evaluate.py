import json
import statistics
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
from retrieval.retriever import retrieve

RETRIEVING_METRICS = [context_precision, context_recall]
GENERATIVE_METRICS = [faithfulness, answer_relevancy]
END_TO_END_METRICS = [answer_correctness, answer_similarity]


class BGEM3RagasEmbeddings(BaseRagasEmbeddings):
    def __init__(self, model: BGEM3FlagModel):
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


def build_eval_dataset(dataset: list[dict], llm: OpenAILike, embed: BGEM3FlagModel, client: QdrantClient,
                        reranker: FlagReranker, docstore: SimpleDocumentStore, collection_name: str,
                        retrieval_kwargs: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Ritorna (rows, timings). `timings` misura la latenza percepita dallo
    studente (condensazione + recupero + generazione, con lo stesso modello
    deployato per condensazione e generazione) — non include il tempo del
    giudice RAGAS, che è overhead di valutazione, non di produzione.
    """
    retrieval_kwargs = retrieval_kwargs or {}
    rows = []
    timings = []
    for case in dataset:
        query = case["question"]
        history = [tuple(turn) for turn in case.get("history", [])]

        
        search_query = condense_question(llm, history, query)
        chunks = retrieve(search_query, client, collection_name, embed, reranker, docstore, **retrieval_kwargs)
        t0 = time.perf_counter()
        response = generate_response(llm, query, chunks, history or None)
        t1 = time.perf_counter()

        rows.append({
            "user_input": query,
            "retrieved_contexts": [c["text"] for c in chunks] or [""],
            "response": response,
            "reference": case["expected_answer"],
        })
        timings.append({
            "id": case["id"],
            "generate_s": round(t1 - t0, 3),
        })
    return rows, timings


def timing_summary(timings: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = [t[key] for t in timings]
        return {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0], 3),
        }

    return {
        "condense_s": stats("condense_s"),
        "retrieve_s": stats("retrieve_s"),
        "generate_s": stats("generate_s"),
        "total_s": stats("total_s"),
        "per_case": timings,
    }


def results_to_json(dataset: list[dict], df, timings: list[dict]) -> dict:
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
):
    rows, timings = build_eval_dataset(dataset, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs)

    openai_client = openai.OpenAI(base_url=llm.api_base, api_key="EMPTY")

    ragas_llm = llm_factory(llm.model, provider="openai", client=openai_client, max_tokens=1024)
    ragas_emb = BGEM3RagasEmbeddings(embed)

    final_dataset = Dataset.from_list(rows)

    metrics = RETRIEVING_METRICS + GENERATIVE_METRICS + END_TO_END_METRICS

    res = ragas_evaluate(dataset=final_dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, show_progress=True)
    df = res.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_to_json(dataset, df, timings), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Risultati salvati in {output_path}")

    for metric in metrics:
        print(f"{metric.name}: {df[metric.name].mean():.2f}")

    t = timing_summary(timings)
    print(f"tempo totale per domanda (s) — media: {t['total_s']['mean']}, mediana: {t['total_s']['median']}, p95: {t['total_s']['p95']}")
    print(f"  di cui generazione — media: {t['generate_s']['mean']}, mediana: {t['generate_s']['median']}")
    return res
