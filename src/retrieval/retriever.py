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
                 mode: str = "hybrid_no_colbert", filters: dict[str, str] | None = None):
        self._client = client
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._docstore = docstore
        self._candidate_pool = candidate_pool
        self._prefetch_limit = prefetch_limit
        self._use_reranker = use_reranker
        self._mode = mode
        self._filters = filters
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Runs hybrid search then optional reranking for one query.

        Args:
            query_bundle (QueryBundle): LlamaIndex query wrapper.

        Returns:
            list[NodeWithScore]: Candidate nodes with their score, unfiltered.
        """
        query_embedding = embed_query(query_bundle.query_str, self._embed_model)
        candidates = search_candidates(
            self._client, self._collection_name, query_embedding,
            limit=self._candidate_pool, prefetch_limit=self._prefetch_limit, mode=self._mode,
            filters=self._filters,
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
            results.append(NodeWithScore(node=node, score=float(score)))
        print(f"[retrieve debug] filters={self._filters} candidates={len(candidates)} "
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
    filters: dict[str, str] | None = None,
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
        filters (dict[str, str] | None): Extra metadata constraints (e.g.
            {"corso": "...", "anno_accademico": "..."}), see
            `retrieval.query_filters.extract_query_filters`.

    Returns:
        list: Selected nodes as dicts with "text", "headings", "source_file", "image_paths".
    """
    base_retriever = HybridQdrantRetriever(
        client, collection_name, embed_model, reranker, docstore,
        candidate_pool=candidate_pool, prefetch_limit=prefetch_limit, use_reranker=use_reranker, mode=mode,
        filters=filters,
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

    return [
        {
            "text": n.node.text,
            "headings": n.node.metadata.get("headings"),
            "source_file": n.node.metadata.get("source_file"),
            "image_paths": n.node.metadata.get("image_paths") or [],
        }
        for n in final_nodes
    ]
