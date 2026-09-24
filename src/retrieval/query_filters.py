import difflib
import re
import unicodedata

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike
from qdrant_client import QdrantClient, models

NO_VALUE = "NESSUNO"

CATEGORICAL_FIELDS = ["corso"]

_YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*[/\-]\s*(?:20)?(\d{2})\b")

_YEAR_LABELED_RE = re.compile(r"\banno\s+accademic\w*\s+(?:\D{0,10})?(20\d{2})\b", re.IGNORECASE)


def list_known_values(client: QdrantClient, collection_name: str, field: str, limit: int = 200) -> list[str]:
    """Lists the distinct values a payload field takes in the collection, so
    the LLM extraction step can be constrained to a real, closed vocabulary
    instead of guessing free-form values.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): Collection to read the facet from.
        field (str): Payload field to facet on (needs a keyword index).
        limit (int): Maximum distinct values to return.

    Returns:
        list[str]: Known values, or [] if the collection/field isn't ready.
    """
    try:
        response = client.facet(collection_name, key=field, limit=limit, exact=True)
    except Exception:
        return []
    return [str(hit.value) for hit in response.hits if hit.value is not None]


def known_values_for_fields(
    client: QdrantClient, collection_name: str, fields: list[str] = CATEGORICAL_FIELDS
) -> dict[str, list[str]]:
    """Fetches the closed vocabulary for each of `fields` in one call, meant
    to be cached at startup (fetching per-query would add a Qdrant round trip
    per message for values that essentially never change mid-session).

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): Collection to read facets from.
        fields (list[str]): Payload fields to fetch vocabularies for.

    Returns:
        dict[str, list[str]]: field -> known values.
    """
    return {field: list_known_values(client, collection_name, field) for field in fields}


def course_tipo_laurea_map(client: QdrantClient, collection_name: str, courses: list[str]) -> dict[str, str]:
    """Looks up each course's tipo_laurea directly from the data (one point
    per course is enough, since the relationship is 1:1), so it can be
    derived from a matched "corso" instead of guessed independently — see
    the note on `CATEGORICAL_FIELDS`.

    Args:
        client (QdrantClient): The Qdrant client instance.
        collection_name (str): Collection to read from.
        courses (list[str]): Known "corso" values, see `list_known_values`.

    Returns:
        dict[str, str]: corso -> tipo_laurea, missing an entry for any
            course with no indexed point (yet) to read it from.
    """
    mapping: dict[str, str] = {}
    for course in courses:
        try:
            points, _ = client.scroll(
                collection_name,
                scroll_filter=models.Filter(must=[models.FieldCondition(key="corso", match=models.MatchValue(value=course))]),
                limit=1,
                with_payload=["tipo_laurea"],
                with_vectors=False,
            )
        except Exception:
            continue
        if points and points[0].payload.get("tipo_laurea") is not None:
            mapping[course] = points[0].payload["tipo_laurea"]
    return mapping


def extract_academic_year(query: str) -> str | None:
    """Regex-extracts an explicit academic year mention (e.g. "2025/2026",
    "2025-26", "anno accademico 2026") and normalizes it to
    "<start_year>/<start_year + 1>", matching the format
    `chunking.metadata.current_academic_year` writes into the index.

    Args:
        query (str): The (condensed, standalone) question.

    Returns:
        str | None: The normalized academic year, or None if the query
            doesn't clearly name one.
    """
    match = _YEAR_RANGE_RE.search(query)
    if match:
        start = int(match.group(1))
        return f"{start}/{start + 1}"
    match = _YEAR_LABELED_RE.search(query)
    if match:
        start = int(match.group(1))
        return f"{start}/{start + 1}"
    return None


FILTER_EXTRACTION_SYSTEM_PROMPT = (
    "Il tuo compito è identificare, per la domanda di uno studente, a quale valore "
    "corrisponde ciascuno dei campi elencati sotto, scegliendo tra i valori possibili "
    "indicati per quel campo.\n\n"
    "I valori possibili sono identificativi tecnici (es. nomi di corso concatenati senza "
    "spazi, tipo \"PsicologiaScolasticaComunita\"): la domanda li nominerà quasi sempre in "
    "linguaggio naturale (es. \"psicologia scolastica e di comunità\", \"laurea magistrale "
    "in psicologia comunità\"). Riconosci il riferimento per SIGNIFICATO, non aspettarti che "
    "compaia identico nella domanda, e copia poi il valore ESATTAMENTE come è scritto "
    "nell'elenco.\n\n"
    "Esempio: se tra i valori possibili di CORSO c'è \"PsicologiaScolasticaComunita\" e la "
    "domanda chiede \"a chi mi rivolgo per i tirocini di psicologia scolastica e comunità?\", "
    "rispondi CORSO: PsicologiaScolasticaComunita.\n\n"
    f"Se la domanda non menziona né implica chiaramente un valore per un campo, scrivi "
    f"esattamente \"{NO_VALUE}\" per quel campo. Non inventare valori che non sono elencati.\n\n"
    "Rispondi con ESATTAMENTE questa riga, senza aggiungere altro testo:\n"
    "CORSO: <valore>"
)

def _normalize(text: str) -> str:
    """Folds accents/case/punctuation away for loose comparison, e.g.
    "Psicologia Scolastica e di Comunità" and "PsicologiaScolasticaComunita"
    both normalize to "psicologiascolasticaecomunita"-ish token soup.
    """
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def closest_known_value(candidate: str, known: list[str]) -> str | None:
    """Matches the model's raw answer for one field against its closed
    vocabulary, tolerating near-misses (accents, spacing, minor rewording)
    instead of requiring byte-for-byte equality — a model that gets the
    reference right but not the exact on-disk spelling shouldn't silently
    lose the filter.

    Args:
        candidate (str): The model's raw answer for this field.
        known (list[str]): That field's closed vocabulary.

    Returns:
        str | None: The matching entry from `known`, or None if nothing is
            close enough (fail open: no filter rather than a wrong one).
    """
    if not candidate or candidate.strip().upper() == NO_VALUE:
        return None
    if candidate in known:
        return candidate

    normalized_candidate = _normalize(candidate)
    normalized_to_known = {_normalize(value): value for value in known}
    if normalized_candidate in normalized_to_known:
        return normalized_to_known[normalized_candidate]

    close = difflib.get_close_matches(normalized_candidate, normalized_to_known.keys(), n=1, cutoff=0.6)
    return normalized_to_known[close[0]] if close else None


_FIELD_LINE_PATTERNS = {
    "corso": re.compile(r"CORSO:\s*(.+)", re.IGNORECASE),
}


def build_filter_extraction_prompt(query: str, known_values: dict[str, list[str]]) -> list[ChatMessage]:
    """Builds the chat messages for the categorical-filter classification call.

    Args:
        query (str): The (condensed, standalone) question.
        known_values (dict[str, list[str]]): Closed vocabulary per field, see
            `known_values_for_fields`.

    Returns:
        list[ChatMessage]: System + user messages for `llm.chat()`.
    """
    def block(label: str, field: str) -> str:
        values = known_values.get(field) or []
        listed = "\n".join(f"  - {v}" for v in values) if values else "  (nessun valore disponibile)"
        return f"{label} (valori possibili):\n{listed}"

    vocab = "\n\n".join(
        block(field.upper(), field) for field in CATEGORICAL_FIELDS
    )
    text = f"{vocab}\n\nDOMANDA: {query}\n\nRisposta:"
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=FILTER_EXTRACTION_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)]),
    ]


def extract_categorical_filters(llm: OpenAILike, query: str, known_values: dict[str, list[str]]) -> dict[str, str]:
    """Classifies the query against each categorical field's closed
    vocabulary in one LLM call, so retrieval can be scoped to (e.g.) one
    course/subject and avoid pulling in a topically-similar chunk from a
    different one (e.g. two courses both having a "chi contatto per i
    tirocini" page with a different, course-specific answer).

    Args:
        llm (OpenAILike): Model used for the classification.
        query (str): The (condensed, standalone) question.
        known_values (dict[str, list[str]]): Closed vocabulary per field, see
            `known_values_for_fields`.

    Returns:
        dict[str, str]: Only fields with a confidently-matched value — a
        field with no match, an unparseable line, or an empty vocabulary is
        omitted (fail open: no filter rather than a wrong one).
    """
    if not any(known_values.get(field) for field in CATEGORICAL_FIELDS):
        return {}

    response = llm.chat(build_filter_extraction_prompt(query, known_values), max_tokens=128, temperature=0.0)
    text = response.message.content or ""

    filters: dict[str, str] = {}
    for field, pattern in _FIELD_LINE_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1).strip()
        matched = closest_known_value(value, known_values.get(field) or [])
        if matched is not None:
            filters[field] = matched
    return filters


def extract_query_filters(
    llm: OpenAILike, query: str, known_values: dict[str, list[str]], tipo_laurea_by_corso: dict[str, str] | None = None
) -> dict[str, str]:
    """Extracts every metadata filter a query implies: "corso" via a
    constrained LLM classification call, with "tipo_laurea" derived from it
    (never guessed independently — see the note on `CATEGORICAL_FIELDS`).

    Args:
        llm (OpenAILike): Model used for the categorical classification.
        query (str): The (condensed, standalone) question.
        known_values (dict[str, list[str]]): Closed vocabularies per
            categorical field, see `known_values_for_fields`.
        tipo_laurea_by_corso (dict[str, str] | None): corso -> tipo_laurea,
            see `course_tipo_laurea_map`.

    Returns:
        dict[str, str]: field -> value, ready to AND into a Qdrant filter
        (see `retrieval.hybrid_search.search_candidates`).
    """
    filters = extract_categorical_filters(llm, query, known_values)
    corso = filters.get("corso")
    if corso is not None:
        tipo_laurea = (tipo_laurea_by_corso or {}).get(corso)
        if tipo_laurea is not None:
            filters["tipo_laurea"] = tipo_laurea
    return filters
