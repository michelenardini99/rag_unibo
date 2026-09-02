from FlagEmbedding import FlagReranker


def build_reranker(device_id: int):
    """Builds the cross-encoder reranker model used by the retriever.

    Args:
        device_id (int): CUDA device index to load the model on.

    Returns:
        FlagReranker: Loaded reranker, scores normalized to [0,1].
    """
    return FlagReranker(
        'BAAI/bge-reranker-v2-m3',
        devices=f"cuda:{device_id}",
        normalize=True  # sigmoid to [0,1] — required for rerank_score_threshold to be meaningful
    )

def rerank(query: str, candidates: list, reranker: FlagReranker, limit: int = 5) -> list:
    """Reranks candidate nodes by relevance to the query using a cross-encoder.

    Args:
        query (str): The query string.
        candidates (list): Candidate nodes to rerank, each with a "text" payload.
        reranker (FlagReranker): The reranker model instance.
        limit (int): Maximum number of results to return.

    Returns:
        list: (candidate, score) pairs, sorted by score descending, truncated to `limit`.
    """
    pairs = [(query, c.payload["text"]) for c in candidates]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:limit]