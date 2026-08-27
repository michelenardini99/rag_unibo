from collections import defaultdict
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from chunking.metadata import extract_metadata
from config import settings


DEFAULT_TOKENIZER_MODEL = settings.embedding_model_id
DEFAULT_CHUNK_MAX_TOKENS = settings.chunk_max_tokens


def build_node_parser(chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
                       tokenizer_model: str = DEFAULT_TOKENIZER_MODEL) -> DoclingNodeParser:
    tokenizer = HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=chunk_max_tokens)
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

def find_unchunked_files(root: Path, docstore: SimpleDocumentStore | None) -> list[Path]:
    already_chunked = {
        node.metadata.get("source_path")
        for node in (docstore.docs.values() if docstore else [])
    }

    return [
        p for p in sorted(root.rglob("*.json"))
        if p.name != "_failures.json" and str(p.relative_to(root)) not in already_chunked
    ]

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


def build_parent_nodes(leaf_nodes: list[BaseNode]) -> list[TextNode]:
    """Groups leaf chunks that share the same document section (same
    source_path + headings breadcrumb) under a synthetic parent node, linked
    via NodeRelationship.PARENT/CHILD. Lets AutoMergingRetriever (§5
    architettura, retrieval gerarchico) merge scattered sibling chunks back
    into their full section when enough of them match a query — needed for
    synthesis-style questions where no single ~512-token chunk covers the
    whole topic. Sections with a single chunk are left as-is (nothing to merge).
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
        image_paths = [p for c in children for p in (c.metadata.get("image_paths") or [])]

        parent = TextNode(
            text="\n\n".join(c.text for c in children),
            metadata={
                **{key: first.metadata.get(key) for key in settings.metadata_schema},
                "stato": first.metadata.get("stato"),
                "source_file": first.metadata.get("source_file"),
                "source_path": source_path,
                "headings": list(headings),
                "image_paths": image_paths,
            },
        )

        for child in children:
            child.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()
        parent.relationships[NodeRelationship.CHILD] = [c.as_related_node_info() for c in children]

        parents.append(parent)

    return parents


def chunk_documents(files: list[Path], root: Path,  chunk_max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
                     tokenizer_model: str = DEFAULT_TOKENIZER_MODEL) -> tuple[list[BaseNode], list[TextNode]]:
    parser = build_node_parser(chunk_max_tokens=chunk_max_tokens, tokenizer_model=tokenizer_model)
    documents = [
        load_as_li_document(p, root)
        for p in files
        if p.name != "_failures.json"
    ]
    leaf_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    _resolve_image_paths(leaf_nodes, root)
    parent_nodes = build_parent_nodes(leaf_nodes)
    return leaf_nodes, parent_nodes


def persist_nodes(nodes: list[BaseNode], persist_dir: Path) -> None:
    """Writes chunked nodes to disk so later stages (embedding/Qdrant, §8
    pipeline box 3) can load them without re-running chunking from scratch.
    """
    docstore_path = persist_dir / "docstore.json"
    if docstore_path.exists():
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:        
        persist_dir.mkdir(parents=True, exist_ok=True)
        docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(persist_dir / "docstore.json"))
