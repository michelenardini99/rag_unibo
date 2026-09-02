import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question
from generation.generator import build_llm
from generation.prompt import clean_source_name
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve


@dataclass
class RetrievalConfig:
    name: str
    chunk_max_tokens: int = settings.chunk_max_tokens
    tokenizer_model: str = settings.embedding_model_id
    mode: str = "hybrid"
    use_reranker: bool = True
    fallback_mode: str = "absolute"
    random_baseline: bool = False

    @property
    def index_key(self) -> str:
        return f"tok{self.chunk_max_tokens}-{self.tokenizer_model.replace('/', '-')}"


CONFIGS: list[RetrievalConfig] = [
    RetrievalConfig(name="hybrid"),
    RetrievalConfig(name="hybrid_no_reranker", use_reranker=False),
    RetrievalConfig(name="hybrid_relative_fallback", fallback_mode="relative"),
    RetrievalConfig(name="dense_only", mode="dense_only"),
    RetrievalConfig(name="sparse_only", mode="sparse_only"),
    RetrievalConfig(name="hybrid_no_colbert", mode="hybrid_no_colbert"),
    RetrievalConfig(name="chunk_256", chunk_max_tokens=256),
    RetrievalConfig(name="chunk_768", chunk_max_tokens=768),
    RetrievalConfig(name="random", random_baseline=True),
]


def _matches(chunks: list[dict], expected_source: str) -> tuple[bool, bool]:
    """Ritorna (hit_top1, hit_top5): il documento atteso compare tra i chunk?"""
    names = [clean_source_name(c["source_file"]) for c in chunks if c.get("source_file")]
    hit_top5 = any(n in expected_source or expected_source in n for n in names)
    hit_top1 = bool(names) and (names[0] in expected_source or expected_source in names[0])
    return hit_top1, hit_top5


def _random_chunks(client: QdrantClient, collection_name: str, docstore: SimpleDocumentStore,
                    pool_ids: list[str], k: int = 5) -> list[dict]:
    ids = random.sample(pool_ids, min(k, len(pool_ids)))
    out = []
    for node_id in ids:
        node = docstore.get_document(node_id)
        if node is None:
            continue
        out.append({
            "text": node.text,
            "headings": node.metadata.get("headings"),
            "source_file": node.metadata.get("source_file"),
            "image_paths": node.metadata.get("image_paths") or [],
        })
    return out


def evaluate_retrieval(config: RetrievalConfig, dataset: list[dict], client: QdrantClient, embed_model,
                        reranker, docstore: SimpleDocumentStore, llm, results_dir: Path) -> dict:
    collection_name = f"{settings.qdrant_collection}__{config.index_key}"

    pool_ids = None
    if config.random_baseline:
        pool_ids = list(docstore.docs.keys())

    per_case = []
    for case in dataset:
        history = [tuple(turn) for turn in case.get("history", [])]
        query = condense_question(llm, history, case["question"]) if history else case["question"]
        query_kind = "condensed" if history else "original"

        if config.random_baseline:
            chunks = _random_chunks(client, collection_name, docstore, pool_ids)
        else:
            chunks = retrieve(
                query, client, collection_name, embed_model, reranker, docstore,
                use_reranker=config.use_reranker, mode=config.mode, fallback_mode=config.fallback_mode,
            )

        hit_top1, hit_top5 = _matches(chunks, case.get("expected_source", ""))
        per_case.append({
            "id": case["id"],
            "diagnostic_tag": case["diagnostic_tag"],
            "query_kind": query_kind,
            "top1": hit_top1,
            "top5": hit_top5,
        })

    def mean(rows: list[dict], key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else float("nan")

    result = {
        "config": config.name,
        "top1": mean(per_case, "top1"),
        "top5": mean(per_case, "top5"),
        "top1_original": mean([r for r in per_case if r["query_kind"] == "original"], "top1"),
        "top5_original": mean([r for r in per_case if r["query_kind"] == "original"], "top5"),
        "top1_condensed": mean([r for r in per_case if r["query_kind"] == "condensed"], "top1"),
        "top5_condensed": mean([r for r in per_case if r["query_kind"] == "condensed"], "top5"),
        "per_case": per_case,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{config.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{config.name:26s} top1={result['top1']:.3f} top5={result['top5']:.3f}  "
          f"(originali: top1={result['top1_original']:.3f} top5={result['top5_original']:.3f} | "
          f"condensate: top1={result['top1_condensed']:.3f} top5={result['top5_condensed']:.3f})")
    return result


def run(configs: list[RetrievalConfig], dataset_path: Path, results_dir: Path, device_id: int) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    embed_model = build_embedding_model(device_id=device_id)
    reranker = build_reranker(device_id=device_id)
    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
    llm = build_llm(base_url=settings.vllm_base_url, model=settings.generation_model, context_window=settings.vllm_max_model_len)

    docstores: dict[str, SimpleDocumentStore] = {}
    for config in configs:
        if config.index_key not in docstores:
            docstores[config.index_key] = SimpleDocumentStore.from_persist_path(
                str(settings.data_chunked_dir / config.index_key / "docstore.json")
            )
        evaluate_retrieval(config, dataset, client, embed_model, reranker, docstores[config.index_key], llm, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.data_eval_dir / "qa_test_set_diagnostic_screening.json")
    parser.add_argument("--results-dir", type=Path, default=settings.data_eval_dir / "retrieval_screening")
    parser.add_argument("--device-id", type=int, default=settings.embeddings_device_id)
    args = parser.parse_args()

    run(CONFIGS, args.dataset, args.results_dir, args.device_id)
