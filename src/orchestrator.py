from llama_index.core.storage.docstore import SimpleDocumentStore
from embedding.embedder import build_embedding_model, embed_nodes
from qdrant_client import QdrantClient
from embedding.qdrant_store import ensure_collection, upsert_nodes
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve

if __name__ == "__main__":
    model = build_embedding_model(device_id=2)
    reranker = build_reranker(device_id=2)
    client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)  # QDRANT_URL da .env — non ancora in config.py

    query = "Quando scade la prima rata da pagare dell'università?"

    res = retrieve(query, client, "ateneo_docs", model, reranker)

    for r in res:
        print(r["headings"])
        print(r["text"])
        print(r["source_file"])
        

