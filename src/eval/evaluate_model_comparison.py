"""Confronto tra modelli di generazione, in due fasi:

  generate  il modello sotto esame (servito da vLLM) risponde al test set;
            domande, contesti, risposte e tempi finiscono in <name>.generations.json
  judge     il giudice (sempre lo stesso per tutti i modelli) assegna i punteggi
            ragas alle risposte salvate; risultati in <name>.json

Un solo giudice per tutti rende i punteggi confrontabili: con il giudice uguale
al modello valutato ognuno giudicava sé stesso. vLLM serve un modello alla
volta, quindi le fasi girano con modelli diversi caricati (vedi
eval/run_model_comparison.sh).
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

from config import settings
from embedding.embedder import build_embedding_model
from eval.evaluate import generate, judge
from generation.generator import build_llm
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.reranker import build_reranker

DEFAULT_JUDGE_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"


def served_model_id(base_url: str) -> str:
    """The model vLLM actually loaded behind its served name ("generation-llm")."""
    with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
        return json.load(response)["data"][0]["root"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["generate", "judge"])
    parser.add_argument("--name", required=True, help="Etichetta del modello in valutazione (es. Qwen-2.5-14B)")
    parser.add_argument("--dataset", type=Path, default=settings.data_eval_dir / "qa_test_set_diagnostic.json")
    parser.add_argument("--results-dir", type=Path, default=settings.data_eval_dir / "model_comparison")
    parser.add_argument("--device-id", type=int, default=settings.embeddings_device_id)
    parser.add_argument("--max-workers", type=int, default=4, help="Casi eseguiti in parallelo nella pipeline condense/retrieve/generate")
    parser.add_argument("--ragas-max-workers", type=int, default=32, help="Concorrenza del giudice ragas (RunConfig.max_workers)")
    parser.add_argument("--llm-timeout", type=float, default=300.0,
                         help="Timeout (s) per le richieste all'LLM: con più worker concorrenti su una singola GPU "
                              "una richiesta può restare in coda dietro le altre più a lungo del default di 60s "
                              "senza che nulla sia realmente bloccato")
    parser.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL_ID,
                        help="Modello giudice atteso in vLLM durante la fase judge: se ne è caricato un altro, si ferma")
    args = parser.parse_args()

    generations_path = args.results_dir / f"{args.name}.generations.json"
    model_id = served_model_id(settings.vllm_base_url)
    embed_model = build_embedding_model(device_id=args.device_id)

    if args.stage == "generate":
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        generate(
            dataset=dataset,
            llm=build_llm(base_url=settings.vllm_base_url, model=settings.generation_model,
                          context_window=settings.vllm_max_model_len, timeout=args.llm_timeout),
            embed=embed_model,
            client=QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True),
            reranker=build_reranker(device_id=args.device_id),
            docstore=SimpleDocumentStore.from_persist_path(str(settings.data_chunked_dir / "docstore.json")),
            output_path=generations_path,
            model_id=model_id,
            collection_name=settings.qdrant_collection,
            max_workers=args.max_workers,
        )
    else:
        if model_id != args.judge_model_id:
            sys.exit(f"vLLM serve {model_id}, non il giudice {args.judge_model_id}: carica il giudice e riprova.")
        judge(
            generations_path=generations_path,
            judge_base_url=settings.vllm_base_url,
            judge_model=settings.generation_model,
            judge_model_id=model_id,
            embed=embed_model,
            output_path=args.results_dir / f"{args.name}.json",
            ragas_max_workers=args.ragas_max_workers,
        )
