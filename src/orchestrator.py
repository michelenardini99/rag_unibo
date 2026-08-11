import json

from eval.evaluate import evaluate
from generation.condense import condense_question
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock
from generation.generator import build_llm, generate_response
from embedding.embedder import build_embedding_model
from qdrant_client import QdrantClient
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve

EXIT_COMMANDS = {"exit", "quit"}
MAX_HISTORY_TURNS = 5

def main() -> None:
    print("Caricamento modelli...")
    embed_model = build_embedding_model(device_id=2)
    reranker = build_reranker(device_id=2)
    llm = build_llm(base_url="http://localhost:8000/v1")
    client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)
    with open('datasets/eval/qa_test_set.json', "r", encoding="utf-8") as file:
        eval_dataset = json.load(file)
    evaluate(eval_dataset, llm, embed_model, client, reranker)
    history: list[tuple[str, str]] = []
    print("Assistente pronto. Scrivi 'exit' o 'quit' per uscire.\n")
    while True:
        query = input("Tu: ").strip()
        if query.lower() in EXIT_COMMANDS:
            break
        if not query:
            continue
        search_query = condense_question(llm, history, query)    
        chunks = retrieve(search_query, client, "ateneo_docs", embed_model, reranker)
        response = generate_response(llm, query, chunks, history)
        print(f"\nAssistente: {response}\n")

        history.append((query, response))
        history = history[-MAX_HISTORY_TURNS:]

if __name__ == "__main__":
    main()
