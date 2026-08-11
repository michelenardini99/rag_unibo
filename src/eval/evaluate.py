import openai
from datasets import Dataset
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from generation.condense import condense_question
from generation.generator import generate_response
from llama_index.llms.openai_like import OpenAILike
from qdrant_client import QdrantClient
from ragas import evaluate as ragas_evaluate
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    faithfulness,       # Misura le allucinazioni (Risposta vs Contesto)
    answer_relevancy,   # Misura la pertinenza (Risposta vs Domanda)
    context_precision,  # Misura il ranking dei chunk (Contesto vs Domanda)
    context_recall,     # Misura la completezza del recupero (Contesto vs Ground Truth)
    answer_correctness,
)
from retrieval.retriever import retrieve


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


def build_eval_dataset(dataset: dict, llm: OpenAILike, embed: BGEM3FlagModel, client: QdrantClient, reranker: FlagReranker) -> list[dict]:
    rows = []
    cases = dataset["casi"]
    for case in cases:
        query = case["domanda"]
        search_query = condense_question(llm, [], query)
        chunks = retrieve(search_query, client, "ateneo_docs", embed, reranker)
        response = generate_response(llm, query, chunks, None)

        rows.append({
            "user_input": query,
            "retrieved_contexts": [c["text"] for c in chunks] or [""],
            "response": response,
            "reference": case["risposta_attesa"],
        })
    return rows


def evaluate(dataset: dict, llm: OpenAILike, embed: BGEM3FlagModel, client: QdrantClient, reranker: FlagReranker):
    rows = build_eval_dataset(dataset, llm, embed, client, reranker)

    openai_client = openai.OpenAI(base_url=llm.api_base, api_key="EMPTY")
    ragas_llm = llm_factory(llm.model, provider="openai", client=openai_client)
    ragas_emb = BGEM3RagasEmbeddings(embed)

    final_dataset = Dataset.from_list(rows)

    metrics = [
        faithfulness,
        answer_relevancy,
        answer_correctness,
        context_precision,
        context_recall,
    ]

    res = ragas_evaluate(dataset=final_dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb, show_progress=True)
    df = res.to_pandas()
    for metric in metrics:
        print(f"{metric.name}: {df[metric.name].mean():.2f}")
    return res

