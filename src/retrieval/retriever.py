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
    """Wraps the custom BGE-M3 hybrid search (dense+sparse+colbert) + BGE
    cross-encoder reranking as a LlamaIndex BaseRetriever, so it can plug into
    AutoMergingRetriever (§5 architettura, retrieval gerarchico Parent-Child)
    while keeping the existing hybrid retrieval logic untouched.

    Returns a wide-ish candidate pool (candidate_pool, not just the final
    top-k) *without* the score cutoff: AutoMergingRetriever needs to see
    multiple siblings of the same section to decide whether to merge them —
    filtering too early would remove exactly the borderline chunks the merge
    is meant to rescue. The score threshold is applied after merging instead
    (see `retrieve()` below).
    """

    def __init__(self, client: QdrantClient, collection_name: str, embed_model, reranker_model,
                 docstore: SimpleDocumentStore, candidate_pool: int = settings.retrieval_candidate_limit,
                 prefetch_limit: int = settings.retrieval_prefetch_limit, use_reranker: bool = True):
        self._client = client
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._docstore = docstore
        self._candidate_pool = candidate_pool
        self._prefetch_limit = prefetch_limit
        self._use_reranker = use_reranker
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = embed_query(query_bundle.query_str, self._embed_model)
        candidates = search_candidates(
            self._client, self._collection_name, query_embedding,
            limit=self._candidate_pool, prefetch_limit=self._prefetch_limit,
        )

        if self._use_reranker:
            scored = rerank(query_bundle.query_str, candidates, self._reranker_model, limit=self._candidate_pool)
        else:
            scored = [(c, c.score) for c in candidates]

        results = []
        for candidate, score in scored:
            node = self._docstore.get_document(str(candidate.id))
            if node is None:
                continue
            results.append(NodeWithScore(node=node, score=float(score)))
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
    prefetch_limit: int = settings.retrieval_prefetch_limit,
    candidate_pool: int = settings.retrieval_candidate_limit,
    top_k: int = settings.rerank_top_k,
    score_threshold: float = settings.rerank_score_threshold,
    merged_fallback_threshold: float = settings.merged_threshold,
) -> list:
    """
    Retrieves relevant nodes (merging sibling chunks into their parent
    section when enough of them match, via AutoMergingRetriever) for the
    given query.

    Args:
        query (str): The query string.
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to search in.
        embed_model: The embedding model used to embed the query.
        reranker: The reranker model used to rerank the candidates.
        docstore (SimpleDocumentStore): docstore holding both leaf and parent
            nodes (datasets/chunked/docstore.json), needed by
            AutoMergingRetriever to resolve parents by id.
        use_reranker: se False, salta lo step di cross-encoder reranking e usa
            direttamente lo score della ricerca ibrida (ablation).
        use_automerging: se False, salta AutoMergingRetriever e ritorna i nodi
            foglia così come recuperati, senza fondere le sezioni (ablation).
        prefetch_limit: quanti candidati prelevare per ciascun ramo (denso/sparso)
            prima del passaggio ColBERT in `search_candidates`.
        candidate_pool: quanti candidati passare al reranker/tenere prima del
            filtro per punteggio finale.
        top_k: numero massimo di nodi restituiti.
        score_threshold: soglia minima di punteggio per i nodi foglia (e per i
            nodi fusi, al primo tentativo).
        merged_fallback_threshold: soglia di fallback per i soli nodi fusi,
            usata quando `score_threshold` non fa passare nulla.

    Returns:
        list: A list of relevant nodes retrieved from the collection.
    """
    base_retriever = HybridQdrantRetriever(
        client, collection_name, embed_model, reranker, docstore,
        candidate_pool=candidate_pool, prefetch_limit=prefetch_limit, use_reranker=use_reranker,
    )

    if use_automerging:
        storage_context = StorageContext.from_defaults(docstore=docstore)
        auto_merging = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
        merged_nodes = auto_merging.retrieve(query)
    else:
        merged_nodes = base_retriever.retrieve(query)

    def select(merged_threshold: float) -> list[NodeWithScore]:
        def passes(n: NodeWithScore) -> bool:
            is_merged = bool(n.node.child_nodes)
            threshold = merged_threshold if is_merged else score_threshold
            return (n.score or 0.0) > threshold
        return sorted((n for n in merged_nodes if passes(n)), key=lambda n: n.score or 0.0, reverse=True)[:top_k]

    final_nodes = select(score_threshold)
    if not final_nodes:
        final_nodes = select(merged_fallback_threshold)

    return [
        {
            "text": n.node.text,
            "headings": n.node.metadata.get("headings"),
            "source_file": n.node.metadata.get("source_file"),
            "image_paths": n.node.metadata.get("image_paths") or [],
        }
        for n in final_nodes
    ]
