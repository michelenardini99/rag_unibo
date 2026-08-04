from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from chunking.metadata import extract_metadata


def build_node_parser() -> DoclingNodeParser:
    tokenizer = HuggingFaceTokenizer.from_pretrained("BAAI/bge-m3", max_tokens=512)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    return DoclingNodeParser(
        chunker=chunker
    )


def load_as_li_document(json_path: Path, converted_root: Path) -> Document:
    metadata = extract_metadata(json_path, converted_root)
    return Document(
        text=json_path.read_text(encoding="utf-8"),
        metadata=metadata,
    )

def _resolve_image_paths(nodes: list, converted_root: Path) -> None:
    """DoclingNodeParser drops the heavy `image`/`data` fields when it
    serializes doc_items into node.metadata (see chunker.py history), so a
    node referencing a picture only carries its self_ref (e.g. "#/pictures/3")
    with no way to reach the actual PNG. Resolve it by re-loading the
    original per-document DoclingDocument JSON (cached per source_path, since
    many nodes share the same source document) and indexing into
    `doc.pictures` at the self_ref's integer index. Needed for §3.2 of the
    architecture doc: the generative LLM gets the original image, not just
    the (sometimes too-generic) VLM caption. Mutates nodes in place.
    """
    doc_cache: dict[str, DoclingDocument] = {}

    for node in nodes:
        picture_refs = [
            item["self_ref"]
            for item in node.metadata.get("doc_items", [])
            if item.get("label") == "picture"
        ]
        if not picture_refs:
            continue

        source_path = node.metadata["source_path"]
        if source_path not in doc_cache:
            doc_cache[source_path] = DoclingDocument.load_from_json(converted_root / source_path)
        doc = doc_cache[source_path]

        image_paths = []
        for ref in picture_refs:
            index = int(ref.rsplit("/", 1)[-1])
            picture = doc.pictures[index]
            if picture.image is not None:
                image_paths.append(str(picture.image.uri))
        if image_paths:
            node.metadata["image_paths"] = image_paths


def chunk_documents(converted_root: Path) -> list[BaseNode]:
    parser = build_node_parser()
    documents = [
        load_as_li_document(p, converted_root)
        for p in sorted(converted_root.rglob("**/*.json"))
        if p.name != "_failures.json"
    ]
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    _resolve_image_paths(nodes, converted_root)
    return nodes


def persist_nodes(nodes: list[BaseNode], persist_dir: Path) -> None:
    """Writes chunked nodes to disk so later stages (embedding/Qdrant, §8
    pipeline box 3) can load them without re-running chunking from scratch.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(persist_dir / "docstore.json"))


if __name__ == "__main__":
    converted_root = Path("datasets/converted/")
    nodes = chunk_documents(converted_root)
    print(f"Chunked {len(nodes)} nodes from {converted_root}.")
    persist_nodes(nodes, Path("datasets/chunked/"))
    print("Persisted to datasets/chunked/docstore.json")
