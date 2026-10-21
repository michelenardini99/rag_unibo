import re
from pathlib import Path
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock

from config import settings

SYSTEM_PROMPT = (
    f"Sei un assistente virtuale per {settings.assistant_institution}, specializzato in domande "
    f"su {settings.assistant_scope}. Rispondi "
    f"sempre in {settings.assistant_language}.\n\n"
    f"Se la domanda non riguarda argomenti pertinenti, dillo esplicitamente e spiega che puoi aiutare solo con "
    f"questioni relative a {settings.assistant_institution}.\n\n"
    "Per le domande pertinenti, usa esclusivamente le informazioni presenti nel CONTESTO "
    "fornito. Se il contesto non contiene la risposta, dillo esplicitamente invece di "
    "inventare o rispondere con conoscenza generale. Riporta date, importi e numeri "
    "esattamente come appaiono nel contesto, senza arrotondare o parafrasare.\n\n"
    "Non riutilizzare informazioni dalle tue risposte precedenti in questa conversazione: "
    "se il turno precedente riguardava un corso, un argomento o un documento diverso, "
    "quell'informazione non vale per la domanda attuale a meno che non sia ripetuta anche "
    "nel CONTESTO qui sotto. In caso di dubbio, tratta ogni domanda come indipendente dalla "
    "cronologia e basati solo sul CONTESTO fornito per questo turno.\n\n"
    "Se nel CONTESTO la risposta cambia a seconda di una condizione che la domanda non "
    "specifica (ad esempio un periodo, una sede o una categoria di studenti):\n"
    "- se i casi sono due o tre e la risposta per ciascuno è breve, rispondi per ciascun "
    "caso, indicando chiaramente a quale caso si riferisce ogni parte;\n"
    "Il titolo di ogni blocco di CONTESTO (dopo il numero) dice a quale sezione, e quindi "
    "a quale caso, si riferisce il suo contenuto: usalo per non mescolare regole di casi "
    "diversi in un'unica lista.\n"
    "- se i casi sono di più, o la risposta completa sarebbe troppo lunga, non rispondere "
    "e non elencare i casi: la tua intera risposta deve essere solo 'CHIARIMENTO:' seguito "
    "da una sola domanda breve che chieda allo studente quale caso lo riguarda, elencando "
    "le alternative presenti nel CONTESTO.\n"
    "Le righe di una tabella (ad esempio più date o scadenze, una per riga) sono casi "
    "diversi: se la domanda non indica a quale riga si riferisce, non scegliere la prima, "
    "applica le stesse regole.\n"
    "Se il CONTESTO contiene un solo caso, o la domanda specifica già la condizione, "
    "rispondi direttamente.\n\n"
    "Il titolo di ogni blocco indica anche se viene dalle pagine di un corso di laurea o "
    "da quelle generali di ateneo. Se è indicato il CORSO DELLO STUDENTE, usa solo i "
    "blocchi di quel corso e quelli generali di ateneo; se l'informazione richiesta "
    "c'è solo per un altro corso, dillo esplicitamente, indicando a quale corso si "
    "riferisce.\n\n"
    "Ogni blocco di CONTESTO inizia con il suo numero tra parentesi quadre. Per ogni "
    "affermazione cita il numero del blocco da cui l'hai presa, sempre tra parentesi "
    "quadre, uno per coppia di parentesi (più blocchi: uno dopo l'altro). Non scrivere "
    "URL, link o nomi di file: l'elenco delle fonti viene aggiunto automaticamente "
    "dopo la tua risposta."
)

CLARIFICATION_MARKER = "CHIARIMENTO:"


class ClarificationDetector:
    """Watches the start of a streamed answer for `CLARIFICATION_MARKER`
    (the model asking the student which case applies, see SYSTEM_PROMPT),
    holding tokens back only until the marker is confirmed or ruled out, so
    it is never shown.

    After the stream, `is_clarification` tells whether the answer was a
    question: if so, the app shows no sources list under it.
    """

    def __init__(self):
        self._head = ""
        self.is_clarification: bool | None = None

    def feed(self, token: str) -> str:
        """Returns the part of the stream that can be passed on."""
        if self.is_clarification is not None:
            return token
        self._head += token
        head = self._head.lstrip()
        if len(head) < len(CLARIFICATION_MARKER) and CLARIFICATION_MARKER.startswith(head.upper()):
            return ""
        self.is_clarification = head.upper().startswith(CLARIFICATION_MARKER)
        return head[len(CLARIFICATION_MARKER):].lstrip() if self.is_clarification else self._head

    def flush(self) -> str:
        """Returns whatever was held back, once the stream has ended."""
        if self.is_clarification is None:
            self.is_clarification = False
            return self._head
        return ""


def strip_clarification_marker(text: str) -> str:
    """`text` without a leading `CLARIFICATION_MARKER`, if it has one."""
    stripped = text.lstrip()
    if stripped.upper().startswith(CLARIFICATION_MARKER):
        return stripped[len(CLARIFICATION_MARKER):].lstrip()
    return text


# One citation: "[3]", or the list/range forms the model also writes despite
# the prompt, "[3, 4, 5]" and "[3-5]".
_CITATION = r"\[\s*\d+(?:\s*[,;\-–]\s*\d+)*\s*\]"
_CITATION_RE = re.compile(_CITATION)
# What `strip_citations` removes: a run of citations with its leading space
# (" [1]", " [1], [2], [3]", " [1][3]", " [3, 4, 5]"), or a clarification
# marker the model put mid-answer — it sometimes lists the cases and then
# ends with "CHIARIMENTO: ...?", where the question is fine but the marker
# isn't. Uppercase only, so a plain "per un chiarimento: contatta..." stays.
_CITATION_GROUP_RE = re.compile(
    rf"\s*{_CITATION}(?:\s*,?\s*{_CITATION})*|" + re.escape(CLARIFICATION_MARKER) + r"[ \t]*"
)
# A text tail that may still turn into one of those once more tokens arrive.
_PENDING_CITATION_RE = re.compile(
    rf"(?:\s*{_CITATION}\s*,?)*\s*(?:\[[\d\s,;\-–]*)?$|" + re.escape(CLARIFICATION_MARKER) + r"\s*$|(?:" + "|".join(
        re.escape(CLARIFICATION_MARKER[:i]) for i in range(len(CLARIFICATION_MARKER), 0, -1)
    ) + r")$"
)


def _cited_numbers(answer: str) -> list[int]:
    """Block numbers cited in `answer`, in order: "[2]" -> 2, "[3, 5]" ->
    3, 5, "[3-5]" -> 3, 4, 5."""
    numbers = []
    for citation in _CITATION_RE.findall(answer):
        for part in re.split(r"\s*[,;]\s*", citation.strip("[] ")):
            bounds = [int(b) for b in re.split(r"\s*[\-–]\s*", part)]
            numbers.extend(range(bounds[0], bounds[-1] + 1) if bounds[-1] - bounds[0] < 20 else bounds)
    return numbers


def strip_citations(text: str) -> str:
    """Removes "[n]" citations (and a stray mid-answer clarification marker)
    from an answer. The model still cites (the code needs it to know which
    sources to list, see `format_sources`), but the student only sees the
    sources list: several chunks of one page made the answer read
    "... 29 ottobre 2026 [1], [2], [3]."."""
    return _CITATION_GROUP_RE.sub("", text)


class CitationStripper:
    """`strip_citations` for a token stream: holds back a tail that could
    still be (part of) a citation run, so "[", "1", "]" arriving as separate
    tokens are never shown."""

    def __init__(self):
        self._pending = ""

    def feed(self, token: str) -> str:
        """Returns the part of the stream so far that is safe to show."""
        text = self._pending + token
        cut = _PENDING_CITATION_RE.search(text).start()
        self._pending = text[cut:]
        return strip_citations(text[:cut])

    def flush(self) -> str:
        """Returns whatever was held back, once the stream has ended."""
        text, self._pending = self._pending, ""
        return strip_citations(text)


def clean_source_name(source_file: str) -> str:
    """Turns a raw filename into a readable citation title, e.g. "Tasse ed
    esenzioni_ importi — Università di Bologna.json" -> "Tasse ed esenzioni: importi".
    Drops the site/course suffix after the separator, restores "_ " to ": "
    (colons get sanitized to underscores in scraped filenames).
    """
    name = Path(source_file).stem
    name = name.split(settings.source_title_separator)[0]
    name = re.sub(r"_\s", ": ", name)
    return re.sub(r"\s+", " ", name).strip()


def question_for_answer(search_query: str, sub_queries: list[str]) -> str:
    """The question the generation model answers: the condensed query, plus
    its sub-questions when `decompose_question` split it.

    The condensed query alone can blur a compound question: "Entro quando si
    pagano le tasse per una laurea magistrale? e per quelle a ciclo unico?"
    was condensed to "...per una laurea magistrale e a ciclo unico?", which
    the model read as "laurea magistrale a ciclo unico" and answered with
    that deadline only — though the context held both. The sub-questions
    ("...laurea magistrale?", "...ciclo unico?") keep the parts apart.

    Args:
        search_query (str): The condensed, standalone question.
        sub_queries (list[str]): Its sub-questions (one element if not split).

    Returns:
        str: `search_query`, followed by the numbered sub-questions if any.
    """
    if len(sub_queries) < 2:
        return search_query
    parts = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_queries, 1))
    return f"{search_query}\nLa domanda ha più parti: rispondi a ciascuna separatamente.\n{parts}"


def source_title(chunk: dict) -> str:
    """Readable title of a chunk's source: the page title (first heading)
    for crawled web pages, the cleaned filename for documents."""
    headings = chunk.get("headings") or []
    if chunk.get("source_url") and headings:
        return headings[0]
    return clean_source_name(chunk["source_file"])


def build_context_block(chunks: list[dict]) -> str:
    """Formats retrieved chunks as numbered "[n] title > section" blocks for
    the prompt.

    Blocks carry no URL: asked to cite URLs, the model rewrote them from
    memory ("iscrizioni-taxe-altre-procedure", ".../prova-final"), producing
    broken links. It cites the number instead, and `format_sources` turns
    the numbers back into the real links.

    Args:
        chunks (list[dict]): Chunks with "text", "headings", "source_file"
            keys, and optionally "source_url".

    Returns:
        str: Chunks joined into a single context string.
    """
    parts = []
    for i, c in enumerate(chunks, 1):
        section = " > ".join((c.get("headings") or [])[1:])
        label = f"{source_title(c)} > {section}" if section else source_title(c)
        scope = f"corso: {course_display_name(c['corso'])}" if c.get("corso") else "generale di ateneo"
        parts.append(f"[{i}] ({scope}) {label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def course_display_name(corso: str) -> str:
    """Readable form of a course identifier: "IngegneriaScienzeInformatiche"
    -> "Ingegneria Scienze Informatiche"."""
    return re.sub(r"(?<=[a-zà-ù])(?=[A-Z])", " ", corso)


def format_sources(answer: str, chunks: list[dict]) -> str:
    """Builds the sources list shown under an answer, from the code's own
    record of which chunk each "[n]" citation refers to (see
    `build_context_block`), so every link is the chunk's real URL.

    Only cited chunks' pages are listed, each once (several chunks of one
    page are one source to the student), as title + link — the "[n]"
    numbers themselves are stripped from the shown answer, see
    `strip_citations`. If the answer cites nothing, every page the answer
    was built from is listed instead.

    Args:
        answer (str): The model's answer, with "[n]" citations.
        chunks (list[dict]): The chunks passed to the model, in prompt order.

    Returns:
        str: A Markdown "Fonti" section, or "" if there are no chunks.
    """
    cited = [n for n in _cited_numbers(answer) if 1 <= n <= len(chunks)]
    numbers = list(dict.fromkeys(cited)) or list(range(1, len(chunks) + 1))
    sources = dict.fromkeys((source_title(chunks[n - 1]), chunks[n - 1].get("source_url")) for n in numbers)
    if not sources:
        return ""
    lines = [f"- [{title}]({url})" if url else f"- {title}" for title, url in sources]
    return "\n\n**Fonti:**\n" + "\n".join(lines)

def build_prompt(query: str, chunks: list[dict], history: list[tuple[str, str]],
                 corso: str | None = None) -> list[ChatMessage]:
    """Constructs the full chat prompt: system instructions, prior turns, then
    the current query with its retrieved context (and images, if supported).

    Args:
        query (str): The user's query.
        chunks (list[dict]): Retrieved context chunks, with "image_paths" if any.
        history (list[tuple[str, str]]): Prior (question, answer) turns.
        corso (str | None): The student's course, if known. The ranking boost
            and `_keep_priority_course` favour its chunks, but other courses'
            near-identical pages can still be in the context (on a question
            whose answer only another course has), and the model answered
            with Psicologia's graduation deadline to a question that named
            Ingegneria. Telling it the course lets it pick the right blocks,
            or say the information is another course's.

    Returns:
        list[ChatMessage]: Messages ready for `llm.chat()`.
    """
    course_line = f"CORSO DELLO STUDENTE: {course_display_name(corso)}\n\n" if corso else ""
    prompt_text = (
        f"CONTESTO:\n{build_context_block(chunks)}\n\n{course_line}"
        f"DOMANDA ({settings.assistant_audience.upper()}):\n{query}"
    )
    blocks = [TextBlock(text=prompt_text)]
    # Only attach raw images if the deployed model actually supports vision —
    # a non-multimodal model rejects ImageBlock content with a 400.
    if settings.generation_model_vision:
        seen = set()
        for c in chunks:
            for img_path in c.get("image_paths") or []:
                if img_path not in seen:
                    blocks.append(ImageBlock(path=Path(img_path)))
                    seen.add(img_path)
    messages = [ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=SYSTEM_PROMPT)])]
    for past_query, past_answer in history or []:
        messages.append(ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=past_query)]))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, blocks=[TextBlock(text=past_answer)]))
    messages.append(ChatMessage(role=MessageRole.USER, blocks=blocks))
    return messages