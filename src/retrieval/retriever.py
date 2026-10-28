import threading

from llama_index.core import QueryBundle
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.storage_context import StorageContext
from qdrant_client import QdrantClient

from retrieval.glossary import expand_acronyms
from retrieval.hybrid_search import embed_query, search_candidates
from retrieval.query_filters import strip_course_mention
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
                 mode: str = "hybrid_no_colbert", priority_corso: str | None = None,
                 corso_boost: float = settings.corso_priority_boost, scope: list[str] | None = None):
        self._client = client
        self._collection_name = collection_name
        self._embed_model = embed_model
        self._reranker_model = reranker_model
        self._docstore = docstore
        self._candidate_pool = candidate_pool
        self._prefetch_limit = prefetch_limit
        self._use_reranker = use_reranker
        self._mode = mode
        self._priority_corso = priority_corso
        self._corso_boost = corso_boost
        self._scope = scope
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Runs hybrid search then optional reranking for one query, across
        the whole corpus — "corso" is never a hard filter here (see
        `hybrid_search.search_candidates`), only a score boost applied below,
        so a question only covered by a different course can still surface.

        Args:
            query_bundle (QueryBundle): LlamaIndex query wrapper.

        Returns:
            list[NodeWithScore]: Candidate nodes with their score, unfiltered.
        """
        query_embedding = embed_query(query_bundle.query_str, self._embed_model)
        candidates = search_candidates(
            self._client, self._collection_name, query_embedding,
            limit=self._candidate_pool, prefetch_limit=self._prefetch_limit, mode=self._mode, scope=self._scope,
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
            score = float(score)
            if self._priority_corso and node.metadata.get("corso") == self._priority_corso:
                score += self._corso_boost
            results.append(NodeWithScore(node=node, score=score))
        print(f"[retrieve debug] priority_corso={self._priority_corso} scope={self._scope} candidates={len(candidates)} "
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
    priority_corso: str | None = None,
    strip_course: bool = True,
    scope: list[str] | None = None,
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
        priority_corso (str | None): A course to prioritize, not restrict to
            — chunks tagged with this "corso" get a score boost
            (`settings.corso_priority_boost`) instead of having every other
            course's content hard-excluded from the search (see
            REPORT_ARCHITETTURA.md §4.4: a question only covered by a
            *different* course must still be answerable).
        scope (list[str] | None): Documents to search, see
            `hybrid_search.search_candidates`: None for all, [] for the
            university-wide ones only, [corso] for that course's and the
            university-wide ones.
        strip_course (bool): If True (default) and `priority_corso` is set,
            search and rerank on the query with the course mention removed
            (`query_filters.strip_course_mention`) — the course is already
            honored by the boost, and its name in the query otherwise
            dominates the reranker. False keeps the query as-is (ablation).

    Returns:
        list: Selected nodes as dicts with "text", "headings", "source_file", "image_paths".
    """
    base_retriever = HybridQdrantRetriever(
        client, collection_name, embed_model, reranker, docstore,
        candidate_pool=candidate_pool, prefetch_limit=prefetch_limit, use_reranker=use_reranker, mode=mode,
        priority_corso=priority_corso, scope=scope,
    )

    if strip_course:
        query = strip_course_mention(query, priority_corso)
    query = expand_acronyms(query)

    if use_automerging:
        storage_context = StorageContext.from_defaults(docstore=docstore)
        auto_merging = AutoMergingRetriever(base_retriever, storage_context, verbose=False)
        merged_nodes = auto_merging.retrieve(query)
    else:
        merged_nodes = base_retriever.retrieve(query)
    print(f"[retrieve debug] after automerging merged_nodes={len(merged_nodes)} "
          f"scores={[round(n.score or 0.0, 3) for n in merged_nodes]}")

    def select(threshold: float) -> list[NodeWithScore]:
        # Ties go to the university-wide copy (no "corso"): it applies to
        # every course, while the course pages copying it say the same thing.
        ranked = sorted(
            (n for n in merged_nodes if (n.score or 0.0) > threshold),
            key=lambda n: (n.score or 0.0, not n.node.metadata.get("corso")),
            reverse=True,
        )
        ranked = _keep_priority_course(ranked, priority_corso)
        return _dedupe_by_text(ranked, lambda n: n.node.text)[:top_k]

    final_nodes = select(score_threshold)
    if not final_nodes:
        if fallback_mode == "relative":
            final_nodes = sorted(merged_nodes, key=lambda n: n.score or 0.0, reverse=True)[:top_k]
        else:
            final_nodes = select(merged_fallback_threshold)
    print(f"[retrieve debug] final_nodes={len(final_nodes)} (score_threshold={score_threshold}, "
          f"merged_fallback_threshold={merged_fallback_threshold})")
    for n in final_nodes:
        headings = " > ".join(n.node.metadata.get("headings") or [])
        print(f"[retrieve debug]   {round(n.score or 0.0, 3):.3f}  {n.node.metadata.get('source_file')}  [{headings}]")

    return [
        {
            "text": n.node.text,
            "headings": n.node.metadata.get("headings"),
            "source_file": n.node.metadata.get("source_file"),
            "source_url": n.node.metadata.get("source_url"),
            "corso": n.node.metadata.get("corso"),
            "image_paths": n.node.metadata.get("image_paths") or [],
            "score": n.score or 0.0,
        }
        for n in final_nodes
    ]


def _keep_priority_course(ranked: list[NodeWithScore], priority_corso: str | None,
                          ratio: float = settings.corso_ambiguity_ratio) -> list[NodeWithScore]:
    """Drops other courses' nodes once the prioritized course genuinely
    answers — its best node reaches `ratio` × the overall best — keeping its
    own and university-wide (no "corso") nodes.

    The ranking boost alone isn't enough when every course has its own
    version of a page: on "requisiti per la domanda di laurea" with
    Ingegneria chosen, Ingegneria's page ranked first but the other 4 context
    slots went to Psicologia's and Infermieristica's equally relevant pages,
    and the answer was built (and cited) from those. When the prioritized
    course has nothing competitive ("Che ruolo ha Franco Callegati?" with
    Psicologia known), nothing is dropped, so a question covered only by
    another course still gets answered (REPORT_ARCHITETTURA.md §4.4).

    Short of that, other courses' versions of a page the prioritized course
    also has (same page title) are dropped anyway: they can outscore it by
    wording alone — Psicologia's deadlines table names its "Invernale"
    session, Ingegneria's only lists dates — and the model then answered an
    Ingegneria student with Psicologia's winter deadline, even told the
    course. Pages only another course has are kept.

    Args:
        ranked (list[NodeWithScore]): Nodes, best first.
        priority_corso (str | None): The course being prioritized, if any.
        ratio (float): See `settings.corso_ambiguity_ratio`.

    Returns:
        list[NodeWithScore]: `ranked`, possibly without other courses' nodes.
    """
    if not priority_corso or not ranked:
        return ranked
    top = ranked[0].score or 0.0
    best_priority = max(
        ((n.score or 0.0) for n in ranked if n.node.metadata.get("corso") == priority_corso), default=0.0,
    )
    if best_priority >= ratio * top:
        return [n for n in ranked if n.node.metadata.get("corso") in (priority_corso, None)]
    own_titles = {_page_title(n) for n in ranked if n.node.metadata.get("corso") == priority_corso} - {""}
    return [
        n for n in ranked
        if n.node.metadata.get("corso") in (priority_corso, None) or _page_title(n) not in own_titles
    ]


def _page_title(node: NodeWithScore) -> str:
    """A node's page title (its first heading), normalized for comparison."""
    headings = node.node.metadata.get("headings") or []
    return " ".join(headings[0].lower().split()) if headings else ""


def _text_key(text: str) -> str:
    """Whitespace- and case-insensitive identity of a chunk's text."""
    return " ".join(text.split()).lower()


def _dedupe_by_text(items: list, get_text) -> list:
    """Keeps only the first of the items sharing the same text — many course
    pages copy a university-wide page (or each other) verbatim, and without
    this the same paragraph from 3-4 sources fills the whole top-k context.
    Order is preserved, so callers pass items best-first."""
    seen: set[str] = set()
    kept = []
    for item in items:
        key = _text_key(get_text(item))
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return kept

def _retrieve_locked(lock: threading.Lock, *args, **kwargs) -> list:
    """Wrapper for `retrieve_decomposed()` that serializes retrieval behind a
    `threading.Lock` — the shared embedding/reranker models aren't safe to
    call concurrently from multiple threads (see REPORT_ARCHITETTURA.md §7).

    Must be called via `asyncio.to_thread(_retrieve_locked, lock, ...)`, not
    awaited directly: acquiring a blocking `threading.Lock` on the event
    loop's own thread would freeze the whole loop for every session, not just
    the caller, while a second request waits on it.
    """
    with lock:
        return retrieve_decomposed(*args, **kwargs)

def retrieve_decomposed(
    queries: list[str],
    client: QdrantClient,
    collection_name: str,
    embed_model,
    reranker,
    docstore: SimpleDocumentStore,
    *,
    top_k: int = settings.rerank_top_k,
    subquery_top_k: int = settings.subquery_top_k,
    decomposed_top_k: int = settings.decomposed_top_k,
    **kwargs,
) -> list[dict]:
    """Retrieves for one or more sub-questions and merges the results,
    instead of running a single retrieval pass on a compound query where one
    sub-topic's chunks can crowd another's out of the shared top-k (see
    REPORT_ARCHITETTURA.md §5.4/§6.3 — observed in practice on a two-part
    question about "domanda di laurea" vs. "accesso al corso": both answered
    from the same document because only it survived reranking).

    Args:
        queries (list[str]): One or more standalone (sub-)questions, from
            `generation.condense.decompose_question`. A single-element list
            behaves exactly like calling `retrieve()` directly.
        client, collection_name, embed_model, reranker, docstore: Forwarded to `retrieve()`.
        top_k (int): Chunk budget for a single question.
        subquery_top_k (int): Chunks per sub-question of a compound one.
        decomposed_top_k (int): Total chunk budget for a compound question,
            lowering the per-sub-question budget (to at least 2) when there
            are many parts, so the context can't grow with their number.
            Splitting the single-question budget (5 // 2 = 2 each) was too
            little: on "i requisiti sono uguali se sono iscritto al primo
            anno o al secondo?" the second-year exemption section ranked 4th
            behind near-identical sections of the same page (0.941 vs
            0.975-0.965) and never reached the model, which answered with
            the "fuori corso" rules instead.
        **kwargs: Forwarded to `retrieve()` (e.g. `priority_corso`, `fallback_mode`).

    Returns:
        list[dict]: Merged, deduplicated chunks (same shape as `retrieve()`),
        in sub-question order — first the chunks for `queries[0]`, etc.
    """
    if len(queries) == 1:
        return retrieve(queries[0], client, collection_name, embed_model, reranker, docstore, top_k=top_k, **kwargs)

    per_query_top_k = max(2, min(subquery_top_k, decomposed_top_k // len(queries)))
    merged = [
        chunk
        for q in queries
        for chunk in retrieve(q, client, collection_name, embed_model, reranker, docstore, top_k=per_query_top_k, **kwargs)
    ]
    return _dedupe_by_text(merged, lambda c: c["text"])
