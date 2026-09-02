from dataclasses import dataclass
from pathlib import Path
import argparse
from chunking.chunker import chunk_documents, persist_nodes, find_unchunked_files
from config import settings
from embedding.embedder import build_embedding_model, embed_nodes
from embedding.qdrant_store import ensure_collection, upsert_nodes
from qdrant_client import QdrantClient
from llama_index.core.storage.docstore import SimpleDocumentStore
from config import settings


@dataclass
class IndexConfig:
    root: Path
    chunked_dir: Path
    collection_name: str
    device_id: int
    recreate: bool
    docstore: SimpleDocumentStore | None = None
    chunk_max_tokens: int = settings.chunk_max_tokens
    tokenizer_model: str = settings.embedding_model_id
    


def indexing(config: IndexConfig) -> None:
    if config.recreate:
        files = list(config.root.rglob("*.json"))
    else:
        files = find_unchunked_files(config.root, config.docstore)

    leaf_nodes, parent_nodes = chunk_documents(files, config.root, chunk_max_tokens=config.chunk_max_tokens,
                                                tokenizer_model=config.tokenizer_model)
    print(f"Chunked {len(leaf_nodes)} leaf chunk(s) + {len(parent_nodes)} parent section(s) from {config.root}.")

    persist_nodes(leaf_nodes + parent_nodes, config.chunked_dir)
    print(f"Persisted docstore to {config.chunked_dir / 'docstore.json'}.")

    embed_model = build_embedding_model(device_id=config.device_id)
    embeddings = embed_nodes(leaf_nodes, embed_model)

    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
    ensure_collection(client, config.collection_name, recreate=config.recreate)
    upsert_nodes(client, config.collection_name, leaf_nodes, embeddings)
    print(f"Indicizzati {len(leaf_nodes)} chunk in Qdrant ('{config.collection_name}').")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", help="Ricrea da 0 i vettori della collezione data")

    args = parser.parse_args()

    recreate = False

    if args.recreate:
        recreate = True

    docstore_path = settings.data_chunked_dir / "docstore.json"
    docstore = SimpleDocumentStore.from_persist_path(str(docstore_path)) if docstore_path.exists() else None

    indexing(IndexConfig(
        root=settings.data_converted_dir,
        chunked_dir=settings.data_chunked_dir,
        collection_name=settings.qdrant_collection,
        device_id=settings.embeddings_device_id,
        docstore=docstore,
        recreate=recreate
    ))
