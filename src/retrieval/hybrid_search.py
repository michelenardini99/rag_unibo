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
                       scope: list[str] | None = None) -> list:
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
        scope (list[str] | None): Which documents may be searched: None for
            the whole corpus; otherwise the university-wide ones
            ("ambito" = "ateneo") plus those of the listed courses — [] for
            university-wide only. The chat restricts every search to the
            student's course and the university pages (see `app.py`), so
            near-identical pages of other courses can't stand in for theirs.

    Returns:
        list: Candidate points from Qdrant, ranked by the chosen strategy.
    """
    dense_vec = query_embedding["dense_vecs"][0].tolist()
    weights = query_embedding["lexical_weights"][0]
    sparse_vec = models.SparseVector(indices=[int(k) for k in weights], values=list(weights.values()))

    # "stato" is always a hard filter (document validity). "corso" is one only
    # when `scope` is given; without it every course is searched.
    search_filter = models.Filter(must=[models.FieldCondition(key="stato", match=models.MatchValue(value="vigente"))])
    if scope is not None:
        search_filter.should = [models.FieldCondition(key="ambito", match=models.MatchValue(value="ateneo"))] + [
            models.FieldCondition(key="corso", match=models.MatchValue(value=corso)) for corso in scope
        ]

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