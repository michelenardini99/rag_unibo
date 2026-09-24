from FlagEmbedding.inference import BGEM3FlagModel
from qdrant_client import QdrantClient, models


def embed_query(query: str, model: BGEM3FlagModel) -> dict:
    """Embeds a query into dense, sparse and colbert vectors.

    Args:
        query (str): Query string to embed.
        model (BGEM3FlagModel): Loaded embedding model.

    Returns:
        dict: BGE-M3 output with "dense_vecs", "lexical_weights", "colbert_vecs".
    """
    return model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )


def search_candidates(client: QdrantClient, collection_name: str, query_embedding: dict, limit: int = 20,
                       prefetch_limit: int = 50, mode: str = "hybrid_no_colbert",
                       filters: dict[str, str] | None = None) -> list:
    """Searches candidate nodes in Qdrant.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The collection to search.
        query_embedding (dict): Output of `embed_query()`.
        limit (int): Number of candidates to return.
        prefetch_limit (int): Candidates fetched per branch (dense/sparse)
            before fusion/reranking, for modes that prefetch.
        mode (str): Retrieval strategy — default "hybrid_no_colbert" (RRF
            fusion) measured best on this corpus; "hybrid" (adds a ColBERT
            stage), "dense_only", "sparse_only" kept for ablation.
        filters (dict[str, str] | None): Extra payload field -> value
            constraints, ANDed with the always-on "stato" filter (see
            `retrieval.query_filters.extract_query_filters`) — restricts
            search to (e.g.) one course/subject/academic year so a
            topically-similar chunk from a different one can't outrank the
            actually relevant one.

    Returns:
        list: Candidate points from Qdrant, ranked by the chosen strategy.
    """
    dense_vec = query_embedding["dense_vecs"][0].tolist()
    weights = query_embedding["lexical_weights"][0]
    sparse_vec = models.SparseVector(indices=[int(k) for k in weights], values=list(weights.values()))

    conditions = [models.FieldCondition(key="stato", match=models.MatchValue(value="vigente"))]
    for field, value in (filters or {}).items():
        conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    search_filter = models.Filter(must=conditions)

    if mode == "dense_only":
        result = client.query_points(
            collection_name, query=dense_vec, using="dense",
            query_filter=search_filter, limit=limit, with_payload=True,
        )
        return result.points

    if mode == "sparse_only":
        result = client.query_points(
            collection_name, query=sparse_vec, using="sparse",
            query_filter=search_filter, limit=limit, with_payload=True,
        )
        return result.points

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=search_filter),
        models.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit, filter=search_filter),
    ]

    if mode == "hybrid_no_colbert":
        result = client.query_points(
            collection_name, prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True,
        )
        return result.points

    colbert_vec = query_embedding["colbert_vecs"][0].tolist()
    result = client.query_points(
        collection_name,
        prefetch=prefetch,
        query=colbert_vec,
        using="colbert",
        limit=limit,
        with_payload=True
    )

    return result.points