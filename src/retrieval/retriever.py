from qdrant_client import QdrantClient
from retrieval.hybrid_search import embed_query, search_candidates
from retrieval.reranker import rerank


def retrieve(query: str, client: QdrantClient, collection_name: str, embed_model, reranker) -> list:
    """
    Retrieves relevant nodes from a Qdrant collection based on the provided query.

    Args:
        query (str): The query string.
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): The name of the collection to search in.
        embed_model: The embedding model used to embed the query.
        reranker: The reranker model used to rerank the candidates.

    Returns:
        list: A list of relevant nodes retrieved from the collection.
    """
    query_embedding = embed_query(query, embed_model)
    candidates = search_candidates(client, collection_name, query_embedding)
    reranked_candidates = rerank(query, candidates, reranker)
    final_candidates = [c for c, s in reranked_candidates if s > 0.6]

    return [
        {
            "text": c.payload["text"],
            "headings": c.payload.get("headings"),
            "source_file": c.payload.get("source_file"),
            "image_paths": c.payload.get("image_paths") or [],
        }
        for c in final_candidates
    ]