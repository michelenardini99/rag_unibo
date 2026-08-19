from llama_index.core import QueryBundle
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from qdrant_client import QdrantClient

from retrieval.hybrid_search import embed_query, search_candidates
from retrieval.reranker import rerank

RERANK_CANDIDATE_POOL = 20
TOP_K = 6
SCORE_THRESHOLD = 0.3
# Soglia di fallback, usata SOLO per le sezioni fuse e SOLO quando la soglia
# standard non fa passare nulla (vedi retrieve()) — per domande di sintesi
# dove nessun singolo chunk/sezione è mai giudicato molto pertinente dal
# reranker, ma è comunque meglio di "nessuna informazione pertinente". Le
# foglie restano sempre a SCORE_THRESHOLD: non si abbassa mai la precisione
# sulle domande fattuali/numeriche, che sono la priorità del progetto.
MERGED_FALLBACK_THRESHOLD = 0.12


class HybridQdrantRetriever(BaseRetriever):
    """Wraps the custom BGE-M3 hybrid search (dense+sparse+colbert) + BGE
    cross-encoder reranking as a LlamaIndex BaseRetriever, so it can plug into
    AutoMergingRetriever (§5 architettura, retrieval gerarchico Parent-Child)
    while keeping the existing hybrid retrieval logic untouched.

    Returns a wide-ish candidate pool (RERANK_CANDIDATE_POOL, not just the
    final top-k) *without* the score cutoff: AutoMergingRetriever needs to see
    multiple siblings of the same section to decide whether to merge them —
    filtering too early would remove exactly the borderline chunks the merge
    is meant to rescue. The score threshold is applied after merging instead
    (see `retrieve()` below).
    """

    def __init__(self, client: QdrantClient, collection_name: str, embed_model, reranker_model,
                 docstore: SimpleDocumentStore, candidate_pool: int = RERANK_CANDIDATE_POOL):
        self._client = client
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._docstore = docstore
        self._candidate_pool = candidate_pool
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_embedding = embed_query(query_bundle.query_str, self._embed_model)
        candidates = search_candidates(self._client, self._collection_name, query_embedding)
        reranked = rerank(query_bundle.query_str, candidates, self._reranker_model, limit=self._candidate_pool)

        results = []
        for candidate, score in reranked:
            node = self._docstore.get_document(str(candidate.id))
            if node is None:
                continue
            results.append(NodeWithScore(node=node, score=float(score)))
        return results


def retrieve(query: str, client: QdrantClient, collection_name: str, embed_model, reranker,
              docstore: SimpleDocumentStore) -> list:
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

    Returns:
        list: A list of relevant nodes retrieved from the collection.
    """
    base_retriever = HybridQdrantRetriever(client, collection_name, embed_model, reranker, docstore)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    auto_merging = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
    merged_nodes = auto_merging.retrieve(query)

    def select(merged_threshold: float) -> list[NodeWithScore]:
        def passes(n: NodeWithScore) -> bool:
            is_merged = bool(n.node.child_nodes)
            threshold = merged_threshold if is_merged else SCORE_THRESHOLD
            return (n.score or 0.0) > threshold
        return sorted((n for n in merged_nodes if passes(n)), key=lambda n: n.score or 0.0, reverse=True)[:TOP_K]

    final_nodes = select(SCORE_THRESHOLD)
    if not final_nodes:
        final_nodes = select(MERGED_FALLBACK_THRESHOLD)

    return [
        {
            "text": n.node.text,
            "headings": n.node.metadata.get("headings"),
            "source_file": n.node.metadata.get("source_file"),
            "image_paths": n.node.metadata.get("image_paths") or [],
        }
        for n in final_nodes
    ]