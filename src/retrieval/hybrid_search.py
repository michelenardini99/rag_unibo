from FlagEmbedding.inference import BGEM3FlagModel
from qdrant_client import QdrantClient, models


def embed_query(query: str, model: BGEM3FlagModel) -> dict:
    """Embeds a query string using the provided embedding model."""
    return model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True
    )


def search_candidates(client: QdrantClient, collection_name: str, query_embedding: dict, limit: int = 20,
                       prefetch_limit: int = 50, mode: str = "hybrid") -> list:
    """
    Searches for candidate nodes in a Qdrant collection based on the provided query embedding.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to search in.
        query_embedding (dict): A dictionary containing the query embedding.
        limit (int): The number of top candidates to retrieve.
        prefetch_limit (int): The number of candidates to prefetch for further processing.
        mode: strategia di retrieval, per isolare il contributo di ciascun segnale
            (ablation di retrieval puro, non usato in produzione):
            - "hybrid" (default): dense+sparse in prefetch, riordinati per score ColBERT
              (comportamento originale).
            - "dense_only": solo il branch denso.
            - "sparse_only": solo il branch sparso (pesi lessicali BGE-M3, non BM25 classico).
            - "hybrid_no_colbert": fonde dense+sparse via RRF nativo di Qdrant, senza lo
              stage ColBERT finale.

    Returns:
        list: A list of candidate nodes retrieved from the collection.
    """
    dense_vec = query_embedding["dense_vecs"][0].tolist()
    weights = query_embedding["lexical_weights"][0]
    sparse_vec = models.SparseVector(indices=[int(k) for k in weights], values=list(weights.values()))

    vigente_filter = models.Filter(
        must=[models.FieldCondition(key="stato", match=models.MatchValue(value="vigente"))]
    )

    if mode == "dense_only":
        result = client.query_points(
            collection_name, query=dense_vec, using="dense",
            query_filter=vigente_filter, limit=limit, with_payload=True,
        )
        return result.points

    if mode == "sparse_only":
        result = client.query_points(
            collection_name, query=sparse_vec, using="sparse",
            query_filter=vigente_filter, limit=limit, with_payload=True,
        )
        return result.points

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense", limit=prefetch_limit, filter=vigente_filter),
        models.Prefetch(query=sparse_vec, using="sparse", limit=prefetch_limit, filter=vigente_filter),
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