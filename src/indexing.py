import uuid
from dataclasses import dataclass
from pathlib import Path
import argparse
from chunking.chunker import chunk_documents, persist_nodes, find_unchunked_files
from chunking.metadata import ID_NAMESPACE
from config import settings
from embedding.embedder import build_embedding_model, embed_nodes
from embedding.qdrant_store import delete_by_doc_id, ensure_collection, upsert_nodes
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
    


def _purge_changed_documents(config: IndexConfig, client: QdrantClient, changed_files: list[Path]) -> None:
    """Removes stale docstore entries and Qdrant points for documents whose
    source content changed since they were last chunked, so reprocessing
    them below replaces their chunks instead of piling up duplicates
    alongside them (see chunking.chunker.find_unchunked_files).

    Args:
        config (IndexConfig): Run configuration (needs `docstore`, `root`,
            `collection_name`, `chunked_dir`).
        client (QdrantClient): Client for the collection to purge from.
        changed_files (list[Path]): Converted JSON paths whose checksum no
            longer matches what's recorded in the docstore.
    """
    if not changed_files or config.docstore is None:
        return
    for path in changed_files:
        rel = str(path.relative_to(config.root))
        stale_ids = [
            node_id for node_id, node in config.docstore.docs.items()
            if node.metadata.get("source_path") == rel
        ]
        for node_id in stale_ids:
            config.docstore.delete_document(node_id, raise_error=False)
        doc_id = str(uuid.uuid5(ID_NAMESPACE, rel))
        delete_by_doc_id(client, config.collection_name, doc_id)
    config.docstore.persist(str(config.chunked_dir / "docstore.json"))


def indexing(config: IndexConfig) -> None:
    """Runs the offline pipeline: chunk documents, persist the docstore,
    embed leaf chunks, and upsert them into Qdrant.

    Args:
        config (IndexConfig): Paths, device and chunking parameters for the run.
    """

    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True, timeout=120)
    ensure_collection(client, config.collection_name, recreate=config.recreate)

    if config.recreate:
        files = list(config.root.rglob("*.json"))
    else:
        new_files, changed_files = find_unchunked_files(config.root, config.docstore)
        _purge_changed_documents(config, client, changed_files)
        files = new_files + changed_files

    if not files:
        print("Nessun documento nuovo o modificato da indicizzare.")
        return

    leaf_nodes, parent_nodes = chunk_documents(files, config.root, chunk_max_tokens=config.chunk_max_tokens,
                                                tokenizer_model=config.tokenizer_model)
    print(f"Chunked {len(leaf_nodes)} leaf chunk(s) + {len(parent_nodes)} parent section(s) from {config.root}.")

    persist_nodes(leaf_nodes + parent_nodes, config.chunked_dir)
    print(f"Persisted docstore to {config.chunked_dir / 'docstore.json'}.")

    if not leaf_nodes:
        print("Nessun chunk prodotto per i documenti processati: nessun vettore da aggiornare.")
        return

    embed_model = build_embedding_model(device_id=config.device_id)
    # Only leaf nodes are embedded/indexed — parent sections exist purely for
    # AutoMergingRetriever to resolve into, never searched directly.
    embeddings = embed_nodes(leaf_nodes, embed_model)

    upsert_nodes(client, config.collection_name, leaf_nodes, embeddings)
    print(f"Indicizzati {len(leaf_nodes)} chunk in Qdrant ('{config.collection_name}').")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true", help="Ricrea da 0 i vettori della collezione data")

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
