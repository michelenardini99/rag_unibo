from llama_index.core.storage.docstore import SimpleDocumentStore
from embedding.embedder import build_embedding_model, embed_nodes
from qdrant_client import QdrantClient
from embedding.qdrant_store import ensure_collection, upsert_nodes

if __name__ == "__main__":
    docstore = SimpleDocumentStore.from_persist_path("datasets/chunked/docstore.json")
    nodes = list(docstore.docs.values())

    model = build_embedding_model(device_id=2)  # o settings.embeddings_device_id
    embeddings = embed_nodes(nodes, model)

    client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)  # QDRANT_URL da .env — non ancora in config.py
    ensure_collection(client, "ateneo_docs")  # usa un nome di test, non "ateneo_docs" vero, per non sporcare la collection reale
    upsert_nodes(client, "ateneo_docs", nodes, embeddings)

