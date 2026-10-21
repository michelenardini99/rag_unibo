import re
import uuid
from collections import defaultdict
from pathlib import Path
from llama_index.core import Document
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.node_parser.docling import DoclingNodeParser
from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer, ChunkingSerializerProvider
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument

from chunking.metadata import ID_NAMESPACE, extract_metadata, read_checksum
from config import settings


def build_tokenizer(chunk_max_tokens: int = settings.chunk_max_tokens,
                     tokenizer_model: str = settings.embedding_model_id) -> HuggingFaceTokenizer:
    """Builds the tokenizer shared by the hybrid chunker's own token budget
    and by `_merge_short_leaf_chunks`, so short-chunk detection counts tokens
    the same way the chunker itself does.

    Args:
        chunk_max_tokens (int): Maximum tokens per chunk.
        tokenizer_model (str): HuggingFace model id used to count tokens
            (should match the embedding model for accurate limits).

    Returns:
        HuggingFaceTokenizer: Tokenizer instance.
    """
    return HuggingFaceTokenizer.from_pretrained(tokenizer_model, max_tokens=chunk_max_tokens)


def build_node_parser(tokenizer: HuggingFaceTokenizer) -> DoclingNodeParser:
    """Builds the chunking pipeline that splits DoclingDocuments into LlamaIndex nodes.

    Args:
        tokenizer (HuggingFaceTokenizer): Tokenizer for the hybrid chunker's
            token budget, see `build_tokenizer`.

    Returns:
        DoclingNodeParser: Parser configured with the hybrid chunker.
    """
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True, serializer_provider=_MarkdownTableProvider())
    return DoclingNodeParser(
        chunker=chunker
    )


_MONTHS = ("gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
           "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre")
_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_TABLE_SEPARATOR_RE = re.compile(r"^\|\s*-")


def _readable_tables(text: str) -> str:
    """Rewrites the Markdown tables in a chunk (see `_MarkdownTableProvider`)
    in the words a student's question uses, leaving other text untouched:
    the header row loses bold and capitals ("**APPELLO DI LAUREA**" ->
    "appello di laurea"), and dates are spelled out ("16/11/2026" ->
    "16 novembre 2026").

    The reranker matched tables of bare dates poorly: for the graduation
    deadlines table, "...domanda di laurea tardiva per l'appello di
    ottobre?" scored 0.589 against it as is, 0.840 rewritten ("...per
    l'appello del 16 luglio 2026?" 0.630 -> 0.813).
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        if i + 1 < len(lines) and _TABLE_SEPARATOR_RE.match(lines[i + 1]):
            line = line.replace("**", "").lower()
        lines[i] = _DATE_RE.sub(_date_in_words, line)
    return "\n".join(lines)


def _date_in_words(match: re.Match) -> str:
    day, month, year = int(match.group(1)), int(match.group(2)), match.group(3)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    return f"{day} {_MONTHS[month - 1]} {year}"


class _MarkdownTableProvider(ChunkingSerializerProvider):
    """Serializes tables as (compact) Markdown in chunks, instead of the
    chunker's default "row header, column header = value" triplets.

    The triplets need the table's header row and first column to be detected
    as headers, which on scraped pages they often aren't: the graduation
    deadlines table came out as "04/06/2026, 1 = 18/06/2026. 04/06/2026, 6 =
    16/07/2026", with the column names ("domanda tardiva", "appello di
    laurea") stated once, far from the values — and the model gave the
    graduation date, then the late-application date, as the application
    deadline. A Markdown table keeps every value under its column name, and
    the chunker repeats the header row when it splits a large table
    (`HybridChunker.repeat_table_header`).
    """

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=ChunkingDocSerializer.model_fields["params"].default.model_copy(update={"compact_tables": True}),
        )


def _merge_short_leaf_chunks(leaf_nodes: list[BaseNode], tokenizer: HuggingFaceTokenizer,
                              min_tokens: int = 40, max_tokens: int | None = None) -> list[BaseNode]:
    """Merges a leaf chunk that's too short to be semantically distinctive
    (typically a one-sentence section lead-in with no peer chunk under the
    *same* heading to merge with — `HybridChunker`'s own `merge_peers` only
    merges same-heading siblings) into the next chunk of the same document,
    regardless of heading.

    Without this, a thin lead-in chunk can lose retrieval ranking against a
    long, keyword-dense chunk from an unrelated document that happens to
    share similar phrasing — observed in practice on "Iscriversi ad anni
    successivi al primo" (13-word chunk) losing to a 435-word OFA chunk
    whose subheading is almost the same phrase (REPORT_ARCHITETTURA.md §7).

    Args:
        leaf_nodes (list[BaseNode]): Leaf chunks, assumed in each document's
            original reading order (as `HybridChunker` produces them).
        tokenizer (HuggingFaceTokenizer): Tokenizer used to count tokens,
            see `build_tokenizer` — kept consistent with the chunker's own budget.
        min_tokens (int): Chunks below this many tokens get merged forward.
        max_tokens (int | None): Safety cap on the merged chunk's size, to
            avoid cascading merges ballooning past a usable chunk size on a
            document with several consecutive thin chunks. Defaults to 3x
            `tokenizer`'s configured max token budget.

    Returns:
        list[BaseNode]: Chunks with short leaves merged forward, same total
        document coverage, in the same relative order.
    """
    cap = max_tokens if max_tokens is not None else tokenizer.get_max_tokens() * 3

    by_doc: dict[str, list[BaseNode]] = defaultdict(list)
    for node in leaf_nodes:
        by_doc[node.metadata.get("source_path")].append(node)

    merged: list[BaseNode] = []
    for nodes in by_doc.values():
        i = 0
        while i < len(nodes):
            current = nodes[i]
            while (
                tokenizer.count_tokens(current.text) < min_tokens
                and i + 1 < len(nodes)
                and tokenizer.count_tokens(f"{current.text}\n\n{nodes[i + 1].text}") <= cap
            ):
                nxt = nodes[i + 1]
                current.text = f"{current.text.rstrip()}\n\n{nxt.text.lstrip()}"
                current.metadata["image_paths"] = list(dict.fromkeys(
                    (current.metadata.get("image_paths") or []) + (nxt.metadata.get("image_paths") or [])
                ))
                i += 1
            merged.append(current)
            i += 1
    return merged


def load_as_li_document(json_path: Path, converted_root: Path) -> Document:
    """Loads a converted document JSON as a LlamaIndex Document with derived metadata.

    Args:
        json_path (Path): Path to the converted document JSON.
        converted_root (Path): Root of the converted-documents tree (for metadata extraction).

    Returns:
        Document: The document, ready for chunking, without link targets
        (see `_without_hyperlinks`).
    """
    metadata = extract_metadata(json_path, converted_root)
    return Document(
        text=_without_hyperlinks(json_path.read_text(encoding="utf-8")),
        metadata=metadata,
    )


def _without_hyperlinks(doc_json: str) -> str:
    """Drops link targets from a DoclingDocument, keeping the link text, so
    chunks read "[simula l'importo delle tasse]" instead of carrying the full
    URL.

    URLs are ~26% of the corpus' tokens and carry no meaning for search: on
    the fees page a 5-item list was 411 tokens, 285 of them URLs, which
    pushed it past the chunk budget apart from the sentence introducing it
    ("Hai l'esenzione totale ... se ti trovi in una delle seguenti
    condizioni:"), so no chunk said what "ISEE pari o inferiore a 27.000 €"
    was for. They also cost context window and reranker input (512 tokens)
    and gave the model URLs to recombine into invented links. The page's own
    URL stays in the "source_url" metadata, which the app shows as source.
    """
    doc = DoclingDocument.model_validate_json(doc_json)
    for item in doc.texts:
        item.hyperlink = None
    return doc.model_dump_json()

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
                **{key: first.metadata.get(key) for key in settings.metadata_fields},
                "anno_accademico": first.metadata.get("anno_accademico"),
                "stato": first.metadata.get("stato"),
                "id": doc_id,
                "checksum": first.metadata.get("checksum"),
                "source_file": first.metadata.get("source_file"),
                "source_path": source_path,
                "source_url": first.metadata.get("source_url"),
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
    tokenizer = build_tokenizer(chunk_max_tokens=chunk_max_tokens, tokenizer_model=tokenizer_model)
    parser = build_node_parser(tokenizer)
    documents = [
        load_as_li_document(p, root)
        for p in files
        if p.name != "_failures.json"
    ]
    leaf_nodes = parser.get_nodes_from_documents(documents, show_progress=True)
    for node in leaf_nodes:
        node.text = _readable_tables(node.text)
    leaf_nodes = _merge_short_leaf_chunks(leaf_nodes, tokenizer)
    _assign_deterministic_ids(leaf_nodes)
    _resolve_image_paths(leaf_nodes, root)
    parent_nodes = build_parent_nodes(leaf_nodes)
    return leaf_nodes, parent_nodes


def persist_nodes(nodes: list[BaseNode], persist_dir: Path, replace: bool = False) -> None:
    """Writes chunked nodes to disk so later stages (embedding/Qdrant) can
    load them without re-running chunking from scratch.

    Args:
        nodes (list[BaseNode]): Leaf and/or parent nodes to persist.
        persist_dir (Path): Directory holding (or to create) the docstore.
        replace (bool): If True, `nodes` replace the whole docstore instead
            of being added to it — for a full rebuild, where nodes from a
            previous chunking (no longer in Qdrant) must not survive.
    """
    docstore_path = persist_dir / "docstore.json"
    if docstore_path.exists() and not replace:
        docstore = SimpleDocumentStore.from_persist_path(docstore_path)
    else:        
        persist_dir.mkdir(parents=True, exist_ok=True)
        docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    docstore.persist(str(persist_dir / "docstore.json"))
