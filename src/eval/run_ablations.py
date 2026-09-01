import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from chunking.chunker import DEFAULT_CHUNK_MAX_TOKENS, DEFAULT_TOKENIZER_MODEL
from config import settings
from embedding.embedder import build_embedding_model
from eval.evaluate import evaluate
from generation.generator import build_llm
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from indexing import indexing, IndexConfig
from retrieval.reranker import build_reranker


@dataclass
class AblationConfig:
    name: str
    chunk_max_tokens: int = settings.chunk_max_tokens
    tokenizer_model: str = settings.embedding_model_id
    llm_model: str = settings.generation_model
    llm_base_url: str = settings.vllm_base_url
    
    retrieval_kwargs: dict = field(default_factory=dict)

    @property
    def index_key(self) -> str:
        return f"tok{self.chunk_max_tokens}-{self.tokenizer_model.replace('/', '-')}"


CONFIGS: list[AblationConfig] = [
    AblationConfig(name="baseline"),
    AblationConfig(name="no_reranker", retrieval_kwargs={"use_reranker": False}),
    AblationConfig(name="no_automerging", retrieval_kwargs={"use_automerging": False}),
    AblationConfig(name="chunk_256", chunk_max_tokens=256),
    AblationConfig(name="chunk_768", chunk_max_tokens=768),
]


def _ensure_index(config: AblationConfig, force: bool) -> tuple[Path, str]:
    chunked_dir = settings.data_chunked_dir / config.index_key
    collection_name = f"{settings.qdrant_collection}__{config.index_key}"
    docstore_path = chunked_dir / "docstore.json"

    if force or not docstore_path.exists():
        print(f"[{config.index_key}] (ri)costruzione indice...")
        indexing(IndexConfig(
            root=settings.data_converted_dir,
            chunked_dir=chunked_dir,
            collection_name=collection_name,
            device_id=settings.embeddings_device_id,
            docstore=None,
            chunk_max_tokens=config.chunk_max_tokens,
            tokenizer_model=config.tokenizer_model,
            recreate=True,
        ))
    else:
        print(f"[{config.index_key}] indice già presente, riuso senza rifare chunking/embedding.")

    return chunked_dir, collection_name


def run_ablations(configs: list[AblationConfig], dataset_path: Path, results_dir: Path,
                   force_reindex: bool = False) -> None:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results_dir.mkdir(parents=True, exist_ok=True)

    embed_model = build_embedding_model(device_id=settings.embeddings_device_id)
    reranker = build_reranker(device_id=settings.embeddings_device_id)
    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)

    built_indexes: dict[str, tuple[Path, str]] = {}

    for config in configs:
        print(f"\n=== Configurazione: {config.name} ({config.index_key}) ===")

        if config.index_key not in built_indexes:
            built_indexes[config.index_key] = _ensure_index(config, force_reindex)
        chunked_dir, collection_name = built_indexes[config.index_key]

        docstore = SimpleDocumentStore.from_persist_path(str(chunked_dir / "docstore.json"))
        llm = build_llm(base_url=config.llm_base_url, model=config.llm_model,
                         context_window=settings.vllm_max_model_len)

        evaluate(
            dataset=dataset,
            llm=llm,
            embed=embed_model,
            client=client,
            reranker=reranker,
            docstore=docstore,
            collection_name=collection_name,
            retrieval_kwargs=config.retrieval_kwargs,
            output_path=results_dir / f"{config.name}.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.data_eval_dir / "qa_test_set.json")
    parser.add_argument("--results-dir", type=Path, default=settings.data_eval_dir / "ablations")
    parser.add_argument("--force-reindex", action="store_true",
                         help="Ricostruisce ogni indice anche se già presente su disco.")
    args = parser.parse_args()

    run_ablations(CONFIGS, args.dataset, args.results_dir, force_reindex=args.force_reindex)
