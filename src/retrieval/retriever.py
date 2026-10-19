import threading

from llama_index.core import QueryBundle
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from qdrant_client import QdrantClient

from retrieval.hybrid_search import embed_query, search_candidates
from retrieval.reranker import rerank
from config import settings



class HybridQdrantRetriever(BaseRetriever):
    """Wraps hybrid search + reranking as a LlamaIndex BaseRetriever so it
    plugs into AutoMergingRetriever. Returns an uncut candidate pool —
    AutoMergingRetriever needs to see whole sibling groups to decide what to
    merge; the score threshold is applied after merging, in `retrieve()`.
    """

    def __init__(self, client: QdrantClient, collection_name: str, embed_model, reranker_model,
                 docstore: SimpleDocumentStore, candidate_pool: int = settings.retrieval_candidate_limit,
                 prefetch_limit: int = settings.retrieval_prefetch_limit, use_reranker: bool = True,
                 mode: str = "hybrid_no_colbert", priority_corso: str | None = None,
                 corso_boost: float = settings.corso_priority_boost):
        self._client = client
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._docstore = docstore
        self._candidate_pool = candidate_pool
        self._prefetch_limit = prefetch_limit
        self._use_reranker = use_reranker
        self._mode = mode
        self._priority_corso = priority_corso
        self._corso_boost = corso_boost
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Runs hybrid search then optional reranking for one query, across
        the whole corpus — "corso" is never a hard filter here (see
        `hybrid_search.search_candidates`), only a score boost applied below,
        so a question only covered by a different course can still surface.

        Args:
            query_bundle (QueryBundle): LlamaIndex query wrapper.

        Returns:
            list[NodeWithScore]: Candidate nodes with their score, unfiltered.
        """
        query_embedding = embed_query(query_bundle.query_str, self._embed_model)
        candidates = search_candidates(
            self._client, self._collection_name, query_embedding,
            limit=self._candidate_pool, prefetch_limit=self._prefetch_limit, mode=self._mode,
        )

        if self._use_reranker:
            scored = rerank(query_bundle.query_str, candidates, self._reranker_model, limit=self._candidate_pool)
        else:
            scored = [(c, c.score) for c in candidates]

        results = []
        missing = 0
        for candidate, score in scored:
            node = self._docstore.get_document(str(candidate.id))
            if node is None:
                missing += 1
                continue
            score = float(score)
            if self._priority_corso and node.metadata.get("corso") == self._priority_corso:
                score += self._corso_boost
            results.append(NodeWithScore(node=node, score=score))
        print(f"[retrieve debug] priority_corso={self._priority_corso} candidates={len(candidates)} "
              f"scored={len(scored)} missing_in_docstore={missing} results={len(results)}")
        return results


def retrieve(
    query: str,
    client: QdrantClient,
    collection_name: str,
    embed_model,
    reranker,
    docstore: SimpleDocumentStore,
    *,
    use_reranker: bool = True,
    use_automerging: bool = True,
    mode: str = "hybrid_no_colbert",
    fallback_mode: str = "absolute",
    prefetch_limit: int = settings.retrieval_prefetch_limit,
    candidate_pool: int = settings.retrieval_candidate_limit,
    top_k: int = settings.rerank_top_k,
    score_threshold: float = settings.rerank_score_threshold,
    merged_fallback_threshold: float = settings.merged_threshold,
    priority_corso: str | None = None,
) -> list:
    """Retrieves relevant nodes, merging sibling chunks into their parent
    section via AutoMergingRetriever when enough of them match.

    Args:
        query (str): Search query (already condensed, if it was a follow-up).
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The collection to search.
        embed_model: Embedding model for the query.
        reranker: Cross-encoder reranker.
        docstore (SimpleDocumentStore): Docstore holding leaf and parent nodes.
        use_reranker (bool): If False, skip reranking and use the raw search
            score instead (ablation).
        use_automerging (bool): If False, skip AutoMergingRetriever and return
            leaf nodes as retrieved, without merging sections (ablation).
        mode (str): Retrieval strategy for `search_candidates` — default
            "hybrid_no_colbert" (RRF fusion, no ColBERT) measured best on this
            corpus (REPORT_ARCHITETTURA.md §6); "hybrid" (with ColBERT),
            "dense_only", "sparse_only" kept for ablation.
        fallback_mode (str): "absolute" (default) retries with
            `merged_fallback_threshold` when nothing passes `score_threshold`;
            "relative" takes the top `top_k` by score regardless of the
            absolute value.
        prefetch_limit (int): Candidates fetched per branch before fusion/rerank.
        candidate_pool (int): Candidates kept before the score-threshold filter.
        top_k (int): Maximum number of nodes returned.
        score_threshold (float): Minimum score for a node to pass on the first try.
        merged_fallback_threshold (float): Fallback threshold used when
            `fallback_mode="absolute"` and nothing passes `score_threshold`.
        priority_corso (str | None): A course to prioritize, not restrict to
            — chunks tagged with this "corso" get a score boost
            (`settings.corso_priority_boost`) instead of having every other
            course's content hard-excluded from the search (see
            REPORT_ARCHITETTURA.md §4.4: a question only covered by a
            *different* course must still be answerable).

    Returns:
        list: Selected nodes as dicts with "text", "headings", "source_file", "image_paths".
    """
    base_retriever = HybridQdrantRetriever(
        client, collection_name, embed_model, reranker, docstore,
        candidate_pool=candidate_pool, prefetch_limit=prefetch_limit, use_reranker=use_reranker, mode=mode,
        priority_corso=priority_corso,
    )

    if use_automerging:
        storage_context = StorageContext.from_defaults(docstore=docstore)
        auto_merging = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
        merged_nodes = auto_merging.retrieve(query)
    else:
        merged_nodes = base_retriever.retrieve(query)
    print(f"[retrieve debug] after automerging merged_nodes={len(merged_nodes)} "
          f"scores={[round(n.score or 0.0, 3) for n in merged_nodes]}")

    def select(threshold: float) -> list[NodeWithScore]:
        def passes(n: NodeWithScore) -> bool:
            return (n.score or 0.0) > threshold
        return sorted((n for n in merged_nodes if passes(n)), key=lambda n: n.score or 0.0, reverse=True)[:top_k]

    final_nodes = select(score_threshold)
    if not final_nodes:
        if fallback_mode == "relative":
            final_nodes = sorted(merged_nodes, key=lambda n: n.score or 0.0, reverse=True)[:top_k]
        else:
            final_nodes = select(merged_fallback_threshold)
    print(f"[retrieve debug] final_nodes={len(final_nodes)} (score_threshold={score_threshold}, "
          f"merged_fallback_threshold={merged_fallback_threshold})")
    for n in final_nodes:
        headings = " > ".join(n.node.metadata.get("headings") or [])
        print(f"[retrieve debug]   {round(n.score or 0.0, 3):.3f}  {n.node.metadata.get('source_file')}  [{headings}]")

    return [
        {
            "text": n.node.text,
            "headings": n.node.metadata.get("headings"),
            "source_file": n.node.metadata.get("source_file"),
            "source_url": n.node.metadata.get("source_url"),
            "corso": n.node.metadata.get("corso"),
            "image_paths": n.node.metadata.get("image_paths") or [],
            "score": n.score or 0.0,
        }
        for n in final_nodes
    ]


def _chunk_key(chunk: dict) -> tuple:
    """Identity used to dedupe chunks retrieved for different sub-questions
    of the same compound query (see `retrieve_decomposed`)."""
    return (chunk.get("source_file"), tuple(chunk.get("headings") or []), chunk["text"])

def _retrieve_locked(lock: threading.Lock, *args, **kwargs) -> list:
    """Wrapper for `retrieve_decomposed()` that serializes retrieval behind a
    `threading.Lock` — the shared embedding/reranker models aren't safe to
    call concurrently from multiple threads (see REPORT_ARCHITETTURA.md §7).

    Must be called via `asyncio.to_thread(_retrieve_locked, lock, ...)`, not
    awaited directly: acquiring a blocking `threading.Lock` on the event
    loop's own thread would freeze the whole loop for every session, not just
    the caller, while a second request waits on it.
    """
    with lock:
        return retrieve_decomposed(*args, **kwargs)

def retrieve_decomposed(
    queries: list[str],
    client: QdrantClient,
    collection_name: str,
    embed_model,
    reranker,
    docstore: SimpleDocumentStore,
    *,
    top_k: int = settings.rerank_top_k,
    **kwargs,
) -> list[dict]:
    """Retrieves for one or more sub-questions and merges the results,
    instead of running a single retrieval pass on a compound query where one
    sub-topic's chunks can crowd another's out of the shared top-k (see
    REPORT_ARCHITETTURA.md §5.4/§6.3 — observed in practice on a two-part
    question about "domanda di laurea" vs. "accesso al corso": both answered
    from the same document because only it survived reranking).

    Args:
        queries (list[str]): One or more standalone (sub-)questions, from
            `generation.condense.decompose_question`. A single-element list
            behaves exactly like calling `retrieve()` directly.
        client, collection_name, embed_model, reranker, docstore: Forwarded to `retrieve()`.
        top_k (int): Total chunk budget for the merged result, split evenly
            across sub-questions (at least 2 each) so a 2-3 part question
            doesn't multiply the context size by the number of parts.
        **kwargs: Forwarded to `retrieve()` (e.g. `priority_corso`, `fallback_mode`).

    Returns:
        list[dict]: Merged, deduplicated chunks (same shape as `retrieve()`),
        in sub-question order — first the chunks for `queries[0]`, etc.
    """
    if len(queries) == 1:
        return retrieve(queries[0], client, collection_name, embed_model, reranker, docstore, top_k=top_k, **kwargs)

    per_query_top_k = max(2, top_k // len(queries))
    seen: set[tuple] = set()
    merged: list[dict] = []
    for q in queries:
        for chunk in retrieve(
            q, client, collection_name, embed_model, reranker, docstore, top_k=per_query_top_k, **kwargs,
        ):
            key = _chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
    return merged
