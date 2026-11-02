import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question, decompose_question
from generation.generator import build_llm
from generation.hyde import hyde_query
from generation.prompt import source_title
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.query_filters import course_tipo_laurea_map, extract_query_filters, known_values_for_fields
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve_decomposed


@dataclass
class RetrievalConfig:
    name: str
    # None = the production index (`settings.qdrant_collection`, the docstore
    # app.py loads, built with `settings.chunk_max_tokens`); a number = a
    # separate index of the same corpus with that chunk size, built with
    # `python -m indexing --recreate --chunk-max-tokens N`.
    chunk_max_tokens: int | None = None
    tokenizer_model: str = settings.embedding_model_id
    mode: str = "hybrid_no_colbert"
    use_reranker: bool = True
    fallback_mode: str = "absolute"
    random_baseline: bool = False
    strip_course: bool = True

    @property
    def index_key(self) -> str:
        if self.chunk_max_tokens is None:
            return "current"
        return f"tok{self.chunk_max_tokens}-{self.tokenizer_model.replace('/', '-')}"

    @property
    def collection_name(self) -> str:
        if self.chunk_max_tokens is None:
            return settings.qdrant_collection
        return f"{settings.qdrant_collection}__{self.index_key}"

    @property
    def docstore_path(self) -> Path:
        if self.chunk_max_tokens is None:
            return settings.data_chunked_dir / "docstore.json"
        return settings.data_chunked_dir / self.index_key / "docstore.json"


CONFIGS: list[RetrievalConfig] = [
    # Production setup (app.py): hybrid without ColBERT, reranker, absolute fallback.
    RetrievalConfig(name="hybrid_no_colbert"),
    RetrievalConfig(name="hybrid", mode="hybrid"),
    RetrievalConfig(name="dense_only", mode="dense_only"),
    RetrievalConfig(name="sparse_only", mode="sparse_only"),
    RetrievalConfig(name="no_colbert_no_reranker", use_reranker=False),
    RetrievalConfig(name="no_colbert_relative_fallback", fallback_mode="relative"),
    # Course name kept in the search/rerank query (behavior before
    # `query_filters.strip_course_mention`).
    RetrievalConfig(name="no_colbert_no_strip", strip_course=False),
    # Chunk size: production is 512 tokens (hybrid_no_colbert above).
    RetrievalConfig(name="chunk_256_no_colbert", chunk_max_tokens=256),
    RetrievalConfig(name="chunk_768_no_colbert", chunk_max_tokens=768),
    RetrievalConfig(name="random", random_baseline=True),
]


def _normalize_title(title: str) -> str:
    """Case/whitespace-insensitive title, since course sites capitalize the
    same page differently ("Iscriversi al Corso" / "Iscriversi al corso")."""
    return " ".join(title.split()).casefold()


def _is_expected(chunk: dict, expected_sources: list[dict]) -> bool:
    """Whether a chunk comes from one of the expected pages: same title (as
    cited to the student, see `source_title`) and same course, so the
    same-named page of another course doesn't count.
    """
    title = _normalize_title(source_title(chunk))
    return any(title == _normalize_title(s["title"]) and chunk.get("corso") == s["corso"] for s in expected_sources)


def _normalize_text(text: str) -> str:
    """Chunk text without markdown bold and line breaks, which the crawler
    puts in the middle of sentences ("**700€**\\n.")."""
    return " ".join(text.replace("*", "").split()).casefold()


def _has_evidence(chunks: list[dict], evidence: list[str]) -> bool:
    """Whether every evidence snippet of a case (a figure, date or phrase the
    answer needs) is in the retrieved text: right page is not enough if the
    chunk with the answer was left out."""
    text = _normalize_text(" ".join(c["text"] for c in chunks))
    return all(_normalize_text(e) in text for e in evidence)


def _matches(chunks: list[dict], expected_sources: list[dict]) -> tuple[bool, bool]:
    """Checks whether an expected source document shows up among the retrieved chunks.

    Matches at the source-document level, not exact chunk, since the dataset
    has no hand-labeled chunk-id ground truth.

    Args:
        chunks (list[dict]): Retrieved chunks, ranked.
        expected_sources (list[dict]): Pages with the answer, each a {"title", "corso"}
            ("corso" None for university-wide pages); any one of them counts.

    Returns:
        tuple[bool, bool]: (hit_top1, hit_top5).
    """
    hit_top5 = any(_is_expected(c, expected_sources) for c in chunks)
    hit_top1 = bool(chunks) and _is_expected(chunks[0], expected_sources)
    return hit_top1, hit_top5


def _random_chunks(client: QdrantClient, collection_name: str, docstore: SimpleDocumentStore,
                    pool_ids: list[str], k: int = 5) -> list[dict]:
    """Samples chunks uniformly from the whole docstore, as a sanity-check baseline.

    Args:
        client (QdrantClient): Unused here, kept for signature symmetry with `retrieve()`.
        collection_name (str): Unused here, kept for signature symmetry with `retrieve()`.
        docstore (SimpleDocumentStore): Docstore to sample node ids from.
        pool_ids (list[str]): Candidate node ids to sample from (the whole
            docstore, not a relevance-filtered pool — this must be a true
            "random" control, not a reshuffle of already-good candidates).
        k (int): Number of chunks to sample.

    Returns:
        list[dict]: k randomly sampled chunks in the same shape `retrieve()` returns.
    """
    ids = random.sample(pool_ids, min(k, len(pool_ids)))
    out = []
    for node_id in ids:
        node = docstore.get_document(node_id)
        if node is None:
            continue
        out.append({
            "text": node.text,
            "headings": node.metadata.get("headings"),
            "source_file": node.metadata.get("source_file"),
            "source_url": node.metadata.get("source_url"),
            "corso": node.metadata.get("corso"),
            "image_paths": node.metadata.get("image_paths") or [],
        })
    return out


def evaluate_retrieval(config: RetrievalConfig, dataset: list[dict], client: QdrantClient, embed_model,
                        reranker, docstore: SimpleDocumentStore, llm, results_dir: Path) -> dict:
    """Runs Top-1/Top-5 retrieval evaluation for one config over a dataset,
    writes the result to `results_dir/{config.name}.json`, and prints a summary line.

    Args:
        config (RetrievalConfig): Retrieval variant to evaluate.
        dataset (list[dict]): Test cases, each with "question", "expected_sources",
            "corso" (the conversation's course), optionally "history" and "evidence"
            (snippets the retrieved text must contain, see `_has_evidence`). Cases
            with no expected source (out of scope, missing information) are skipped.
        client (QdrantClient): Qdrant client.
        embed_model: Embedding model for retrieval.
        reranker: Cross-encoder reranker.
        docstore (SimpleDocumentStore): Docstore matching `config.index_key`.
        llm: Model used to condense follow-up questions.
        results_dir (Path): Directory to write the per-config results JSON to.

    Returns:
        dict: Top-1/Top-5 means (overall and split by original/condensed query) plus per-case detail.
    """
    collection_name = config.collection_name

    pool_ids = None
    if config.random_baseline:
        pool_ids = list(docstore.docs.keys())

    # Read per-config since each chunking config has its own Qdrant collection.
    known_values = {} if config.random_baseline else known_values_for_fields(client, collection_name)
    tipo_laurea_by_corso = course_tipo_laurea_map(client, collection_name, known_values.get("corso") or [])

    per_case = []
    for case in dataset:
        if not case.get("expected_sources"):
            continue
        history = [tuple(turn) for turn in case.get("history", [])]
        query = condense_question(llm, history, case["question"]) if history else case["question"]
        # Tracked separately because condensed queries are the ones that broke
        # retrieval in practice (see condensazione-04, REPORT_ARCHITETTURA.md §6).
        query_kind = "condensed" if history else "original"

        hyde = None
        if config.random_baseline:
            chunks = _random_chunks(client, collection_name, docstore, pool_ids)
        else:
            sub_queries, reply = decompose_question(llm, query)
            # As in app.py: the course named in the message, else the one the
            # conversation is about (the session's `default_corso`, here the
            # case's "corso"); search the university-wide pages plus that course's.
            corso = extract_query_filters(llm, query, known_values, tipo_laurea_by_corso).get("corso") or case.get("corso")
            retrieval_kwargs = dict(
                priority_corso=corso, scope=[corso] if corso else [],
                use_reranker=config.use_reranker, mode=config.mode, fallback_mode=config.fallback_mode,
                strip_course=config.strip_course,
            )
            # As in app.py: a question judged a greeting or off-topic isn't searched.
            chunks = [] if reply else retrieve_decomposed(sub_queries, client, collection_name, embed_model,
                                                          reranker, docstore, **retrieval_kwargs)
            # As in app.py: retry with a HyDE query when nothing clears the
            # threshold, only with a course (with none the app asks for it).
            if not chunks and corso and not reply:
                hyde = hyde_query(llm, query)
                chunks = retrieve_decomposed([hyde], client, collection_name, embed_model, reranker, docstore,
                                             **retrieval_kwargs)

        hit_top1, hit_top5 = _matches(chunks, case["expected_sources"])
        per_case.append({
            "id": case["id"],
            "diagnostic_tag": case["diagnostic_tag"],
            "query_kind": query_kind,
            "query": query,
            "hyde": hyde,
            "top1": hit_top1,
            "top5": hit_top5,
            "evidence": _has_evidence(chunks, case["evidence"]) if case.get("evidence") else None,
            "retrieved": [f"{source_title(c)} [{c.get('corso') or 'ateneo'}]" for c in chunks],
        })

    def mean(rows: list[dict], key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else float("nan")

    result = {
        "config": config.name,
        "top1": mean(per_case, "top1"),
        "top5": mean(per_case, "top5"),
        "top1_original": mean([r for r in per_case if r["query_kind"] == "original"], "top1"),
        "top5_original": mean([r for r in per_case if r["query_kind"] == "original"], "top5"),
        "top1_condensed": mean([r for r in per_case if r["query_kind"] == "condensed"], "top1"),
        "top5_condensed": mean([r for r in per_case if r["query_kind"] == "condensed"], "top5"),
        "evidence": mean([r for r in per_case if r["evidence"] is not None], "evidence"),
        "by_tag": {
            tag: {"n": len(rows), "top1": mean(rows, "top1"), "top5": mean(rows, "top5")}
            for tag in sorted({r["diagnostic_tag"] for r in per_case})
            for rows in [[r for r in per_case if r["diagnostic_tag"] == tag]]
        },
        "per_case": per_case,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{config.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{config.name:26s} top1={result['top1']:.3f} top5={result['top5']:.3f} evidence={result['evidence']:.3f}  "
          f"(originali: top1={result['top1_original']:.3f} top5={result['top5_original']:.3f} | "
          f"condensate: top1={result['top1_condensed']:.3f} top5={result['top5_condensed']:.3f})")
    return result


def run(configs: list[RetrievalConfig], dataset_path: Path, results_dir: Path, device_id: int) -> None:
    """Loads shared models/dataset once and evaluates each config in turn,
    reusing docstores across configs that share the same `index_key`.

    Args:
        configs (list[RetrievalConfig]): Retrieval variants to evaluate.
        dataset_path (Path): Path to the test-case dataset JSON.
        results_dir (Path): Directory to write per-config results JSON to.
        device_id (int): CUDA device for the embedding and reranker models.
    """
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))

    embed_model = build_embedding_model(device_id=device_id)
    reranker = build_reranker(device_id=device_id)
    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
    llm = build_llm(base_url=settings.vllm_base_url, model=settings.generation_model, context_window=settings.vllm_max_model_len)

    docstores: dict[str, SimpleDocumentStore] = {}
    for config in configs:
        if config.index_key not in docstores:
            docstores[config.index_key] = SimpleDocumentStore.from_persist_path(str(config.docstore_path))
        evaluate_retrieval(config, dataset, client, embed_model, reranker, docstores[config.index_key], llm, results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=settings.data_eval_dir / "qa_test_set_diagnostic.json")
    parser.add_argument("--results-dir", type=Path, default=settings.data_eval_dir / "retrieval_screening")
    parser.add_argument("--device-id", type=int, default=settings.embeddings_device_id)
    parser.add_argument("--configs", nargs="+", help="Nomi delle configurazioni da valutare (default: tutte quelle di CONFIGS)")
    args = parser.parse_args()

    by_name = {c.name: c for c in CONFIGS}
    run([by_name[n] for n in args.configs] if args.configs else CONFIGS, args.dataset, args.results_dir, args.device_id)
