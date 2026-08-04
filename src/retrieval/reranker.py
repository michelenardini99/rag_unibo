from FlagEmbedding import FlagReranker


def build_reranker(device_id: int):
    """Builds the reranker model for the pipeline."""
    
    return FlagReranker(
        'BAAI/bge-reranker-v2-m3',
        devices=f"cuda:{device_id}",
        normalize=True
    )

def rerank(query: str, candidates: list, reranker: FlagReranker, limit: int = 5) -> list:
    """
    Reranks a list of candidate nodes based on their relevance to the query.

    Args:
        query (str): The query string.
        candidates (list): A list of candidate nodes to rerank.
        model (FlagReranker): The reranker model instance.

    Returns:
        list: A list of reranked candidate nodes.
    """
    pairs = [(query, c.payload["text"]) for c in candidates]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:limit]]