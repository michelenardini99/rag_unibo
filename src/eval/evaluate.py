import json
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
                        retrieval_kwargs: dict | None = None) -> list[dict]:
    retrieval_kwargs = retrieval_kwargs or {}
    rows = []
    for case in dataset:
        query = case["question"]
        search_query = condense_question(llm, [], query)
        chunks = retrieve(search_query, client, collection_name, embed, reranker, docstore, **retrieval_kwargs)
        response = generate_response(llm, query, chunks, None)

        rows.append({
            "user_input": query,
            "retrieved_contexts": [c["text"] for c in chunks] or [""],
            "response": response,
            "reference": case["expected_answer"],
        })
    return rows


def results_to_json(dataset: list[dict], df) -> dict:
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
    rows = build_eval_dataset(dataset, llm, embed, client, reranker, docstore, collection_name, retrieval_kwargs)

    openai_client = openai.OpenAI(base_url=llm.api_base, api_key="EMPTY")

    ragas_llm = llm_factory(llm.model, provider="openai", client=openai_client, max_tokens=1024)
    ragas_emb = BGEM3RagasEmbeddings(embed)

    final_dataset = Dataset.from_list(rows)

    metrics = RETRIEVING_METRICS + GENERATIVE_METRICS + END_TO_END_METRICS

    res = ragas_evaluate(dataset=final_dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, show_progress=True)
    df = res.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_to_json(dataset, df), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Risultati salvati in {output_path}")

    for metric in metrics:
        print(f"{metric.name}: {df[metric.name].mean():.2f}")
    return res
