import argparse
import json
from pathlib import Path

from config import settings
from embedding.embedder import build_embedding_model
from eval.evaluate import evaluate
from generation.generator import build_llm
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.reranker import build_reranker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Etichetta del modello in valutazione (es. mistral_nemo)")
    parser.add_argument("--dataset", type=Path, default=settings.data_eval_dir / "qa_test_set_diagnostic.json")
    parser.add_argument("--results-dir", type=Path, default=settings.data_eval_dir / "model_comparison")
    parser.add_argument("--device-id", type=int, default=settings.embeddings_device_id)
    parser.add_argument("--max-workers", type=int, default=4, help="Casi eseguiti in parallelo nella pipeline condense/retrieve/generate")
    parser.add_argument("--ragas-max-workers", type=int, default=32, help="Concorrenza del giudice ragas (RunConfig.max_workers)")
    parser.add_argument("--llm-timeout", type=float, default=300.0,
                         help="Timeout (s) per le richieste all'LLM: con più worker concorrenti su una singola GPU "
                              "una richiesta può restare in coda dietro le altre più a lungo del default di 60s "
                              "senza che nulla sia realmente bloccato")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))

    embed_model = build_embedding_model(device_id=args.device_id)
    reranker = build_reranker(device_id=args.device_id)
    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
    llm = build_llm(base_url=settings.vllm_base_url, model=settings.generation_model,
                     context_window=settings.vllm_max_model_len, timeout=args.llm_timeout)
    docstore = SimpleDocumentStore.from_persist_path(str(settings.data_chunked_dir / "docstore.json"))
    collection_name = settings.qdrant_collection

    args.results_dir.mkdir(parents=True, exist_ok=True)
    evaluate(
        dataset=dataset,
        llm=llm,
        embed=embed_model,
        client=client,
        reranker=reranker,
        docstore=docstore,
        collection_name=collection_name,
        retrieval_kwargs={},
        output_path=args.results_dir / f"{args.name}.json",
        max_workers=args.max_workers,
        ragas_max_workers=args.ragas_max_workers,
    )
