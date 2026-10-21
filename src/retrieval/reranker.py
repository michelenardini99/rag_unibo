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

def rerank_text(payload: dict) -> str:
    """The text a candidate is judged on: its heading path, then its text.

    The embedding already sees the headings (they are embed metadata, see
    `embedding.embedder`); without them the cross-encoder judges list- and
    table-only chunks blind — "l'indicatore ... è pari o inferiore a
    27.000 €" never says it's the threshold for a fee exemption, its heading
    "Avere l'esenzione e pagare meno tasse" does (0.733 -> 0.860 for "Sotto
    quale ISEE si ha l'esenzione?"; the bare list of instalment dates
    0.129 -> 0.946 for "Quali sono le scadenze delle rate?").
    """
    headings = " > ".join(payload.get("headings") or [])
    return f"{headings}\n{payload['text']}" if headings else payload["text"]


def rerank(query: str, candidates: list, reranker: FlagReranker, limit: int = 5) -> list:
    """Reranks candidate nodes by relevance to the query using a cross-encoder.

    Args:
        query (str): The query string.
        candidates (list): Candidate nodes to rerank, each with a "text"
            (and optionally "headings") payload, see `rerank_text`.
        reranker (FlagReranker): The reranker model instance.
        limit (int): Maximum number of results to return.

    Returns:
        list: (candidate, score) pairs, sorted by score descending, truncated to `limit`.
    """
    if not candidates:
        return []
    pairs = [(query, rerank_text(c.payload)) for c in candidates]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:limit]