import uuid
from collections import defaultdict
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from chunking.metadata import ID_NAMESPACE, extract_metadata, read_checksum
from config import settings


def build_node_parser(chunk_max_tokens: int = settings.chunk_max_tokens,
                       tokenizer_model: str = settings.embedding_model_id) -> DoclingNodeParser:
    """Builds the chunking pipeline that splits DoclingDocuments into LlamaIndex nodes.

    Args:
        chunk_max_tokens (int): Maximum tokens per chunk.
        tokenizer_model (str): HuggingFace model id used to count tokens
            (should match the embedding model for accurate limits).

    Returns:
        DoclingNodeParser: Parser configured with the hybrid chunker.
    """
    tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=chunk_max_tokens)
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    return DoclingNodeParser(
        chunker=chunker
    )


def load_as_li_document(json_path: Path, converted_root: Path) -> Document:
    """Loads a converted document JSON as a LlamaIndex Document with derived metadata.

    Args:
        json_path (Path): Path to the converted document JSON.
        converted_root (Path): Root of the converted-documents tree (for metadata extraction).

    Returns:
        Document: The document, ready for chunking.
    """
    metadata = extract_metadata(json_path, converted_root)
    return Document(
        text=json_path.read_text(encoding="utf-8"),
        metadata=metadata,
    )

def find_unchunked_files(root: Path, docstore: SimpleDocumentStore | None) -> tuple[list[Path], list[Path]]:
    """Lists converted documents that need (re)chunking: ones not yet present
    in the docstore, and ones whose source has changed since they were last
    chunked (detected via the "checksum" metadata field — Docling's own hash
    of the raw source file, see `chunking.metadata.read_checksum`).

    Args:
        root (Path): Root of the converted-documents tree.
        docstore (SimpleDocumentStore | None): Existing docstore, or None to
            treat every file as new.

    Returns:
        tuple[list[Path], list[Path]]: (new_files, changed_files). Callers
        that need to purge stale chunks/vectors before reprocessing only need
        to do so for `changed_files`.
    """
    existing_checksums: dict[str, str | None] = {
        node.metadata["source_path"]: node.metadata.get("checksum")
        for node in (docstore.docs.values() if docstore else [])
        if node.metadata.get("source_path") is not None
    }

    new_files: list[Path] = []
    changed_files: list[Path] = []
    for p in sorted(root.rglob("*.json")):
        if p.name == "_failures.json":
            continue
        rel = str(p.relative_to(root))
        if rel not in existing_checksums:
            new_files.append(p)
        elif read_checksum(p) != existing_checksums[rel]:
            changed_files.append(p)

    return new_files, changed_files

def _assign_deterministic_ids(leaf_nodes: list[BaseNode]) -> None:
    """Replaces LlamaIndex's random per-run node ids with ones derived from
    each document's stable "id".

    Args:
        leaf_nodes (list[BaseNode]): Leaf chunks, mutated in place. Nodes
            with no "id" metadata (shouldn't happen in practice) are left
            with their original id.
    """
    positions: dict[str, int] = defaultdict(int)
    for node in leaf_nodes:
        doc_id = node.metadata.get("id")
        if doc_id is None:
            continue
        position = positions[doc_id]
        positions[doc_id] += 1
        node.id_ = str(uuid.uuid5(ID_NAMESPACE, f"{doc_id}:{position}"))


def _resolve_image_paths(nodes: list, converted_root: Path) -> None:
    """Resolves each node's picture self_ref (e.g. "#/pictures/3") to an
    actual image path, by re-loading the source DoclingDocument.

    Args:
        nodes (list): Leaf nodes to resolve, mutated in place (adds
            "image_paths" to metadata where applicable).
        converted_root (Path): Root of the converted-documents tree.
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


def build_parent_nodes(leaf_nodes: list[BaseNode]) -> list[TextNode]:
    """Groups sibling leaf chunks (same source + headings) under a synthetic
    parent node, so AutoMergingRetriever can return the full section for
    synthesis-style questions no single chunk covers.

    Args:
        leaf_nodes (list[BaseNode]): Leaf chunks to group.

    Returns:
        list[TextNode]: One parent node per section with 2+ chunks; sections
        with a single chunk have nothing to merge and are skipped.
    """
    groups: dict[tuple, list[BaseNode]] = defaultdict(list)
    for node in leaf_nodes:
        key = (node.metadata.get("source_path"), tuple(node.metadata.get("headings") or []))
        groups[key].append(node)

    parents = []
    for (source_path, headings), children in groups.items():
        if len(children) < 2:
            continue

        first = children[0]
        doc_id = first.metadata.get("id")
        image_paths = [p for c in children for p in (c.metadata.get("image_paths") or [])]

        parent = TextNode(
            text="\n\n".join(c.text for c in children),
            metadata={
                **{key: first.metadata.get(key) for key in settings.metadata_schema},
                "anno_accademico": first.metadata.get("anno_accademico"),
                "stato": first.metadata.get("stato"),
                "id": doc_id,
                "checksum": first.metadata.get("checksum"),
                "source_file": first.metadata.get("source_file"),
                "source_path": source_path,
                "headings": list(headings),
                "image_paths": image_paths,
            },
        )
        if doc_id is not None:
            parent.id_ = str(uuid.uuid5(ID_NAMESPACE, f"{doc_id}:parent:{'/'.join(headings)}"))

        for child in children:
            child.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()
        parent.relationships[NodeRelationship.CHILD] = [c.as_related_node_info() for c in children]

        parents.append(parent)

    return parents


def chunk_documents(files: list[Path], root: Path,  chunk_max_tokens: int = settings.chunk_max_tokens,
                     tokenizer_model: str = settings.embedding_model_id) -> tuple[list[BaseNode], list[TextNode]]:
    """Chunks a set of converted documents into leaf and parent nodes.

    Args:
        files (list[Path]): Converted document JSON paths to chunk.
        root (Path): Root of the converted-documents tree.
        chunk_max_tokens (int): Maximum tokens per leaf chunk.
        tokenizer_model (str): HuggingFace model id used to count tokens.

    Returns:
        tuple[list[BaseNode], list[TextNode]]: (leaf_nodes, parent_nodes).
    """
    parser = build_node_parser(chunk_max_tokens=chunk_max_tokens, tokenizer_model=tokenizer_model)
    documents = [
        load_as_li_document(p, root)
        for p in files
        if p.name != "_failures.json"
    ]
    leaf_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    _assign_deterministic_ids(leaf_nodes)
    _resolve_image_paths(leaf_nodes, root)
    parent_nodes = build_parent_nodes(leaf_nodes)
    return leaf_nodes, parent_nodes


def persist_nodes(nodes: list[BaseNode], persist_dir: Path) -> None:
    """Writes chunked nodes to disk so later stages (embedding/Qdrant) can
    load them without re-running chunking from scratch.

    Args:
        nodes (list[BaseNode]): Leaf and/or parent nodes to persist.
        persist_dir (Path): Directory holding (or to create) the docstore.
    """
    docstore_path = persist_dir / "docstore.json"
    if docstore_path.exists():
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:        
        persist_dir.mkdir(parents=True, exist_ok=True)
        docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(persist_dir / "docstore.json"))
