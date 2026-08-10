import re
from pathlib import Path
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock

SYSTEM_PROMPT = (
    "Sei un assistente virtuale dell'Università di Bologna, specializzato in domande "
    "su regolamenti, tasse, iscrizioni, corsi e procedure amministrative. Rispondi "
    "sempre in italiano.\n\n"
    "Se la domanda non riguarda argomenti universitari (saluti, conversazione generica, "
    "richieste non pertinenti), dillo esplicitamente e spiega che puoi aiutare solo con "
    "questioni relative all'Università di Bologna — non tentare comunque di rispondere "
    "nel merito.\n\n"
    "Per le domande pertinenti, usa esclusivamente le informazioni presenti nel CONTESTO "
    "fornito. Se il contesto non contiene la risposta, dillo esplicitamente invece di "
    "inventare o rispondere con conoscenza generale. Riporta date, importi e numeri "
    "esattamente come appaiono nel contesto, senza arrotondare o parafrasare.\n\n"
    "Ogni blocco di CONTESTO inizia con '[Fonte: <nome>]': per ogni affermazione, cita "
    "tra parentesi quadre il <nome> del blocco da cui l'hai presa, sostituendolo con "
    "quello reale — non copiare esempi. Se il contesto contiene più fonti, cita tutte "
    "le fonti rilevanti."
)


def clean_source_name(source_file: str) -> str:
    """Turns a raw stage-1 filename (e.g. "Tasse ed esenzioni_ importi e
    scadenze — Università di Bologna.json") into a readable title for
    citation display ("Tasse ed esenzioni: importi e scadenze").

    Scraped Unibo page titles follow "<titolo> — <sito/corso...>.json";
    dropping everything from the em-dash on removes the site/course suffix,
    and "_ " gets restored to ": " since scraped filenames sanitize colons
    to underscores. Plain filenames without an em-dash (e.g. CamelCase PDF
    names) pass through unchanged — no general fix for those without a
    lookup table.
    """
    name = Path(source_file).stem
    name = name.split(" — ")[0]
    name = re.sub(r"_\s", ": ", name)
    return re.sub(r"\s+", " ", name).strip()


def build_context_block(chunks: list[dict]) -> str:
    """
    Constructs a context block from a list of chunks.

    Args:
        chunks (list[dict]): A list of chunks, where each chunk is a dictionary containing 'text', 'headings', 'source_file', and 'image_paths'.
    """

    parts = []
    for c in chunks:
        headings = " > ".join(c.get("headings") or [])
        source_name = clean_source_name(c["source_file"])
        parts.append(f"[Fonte: {source_name} - {headings}\n{c["text"]}]")
    return "\n\n---\n\n".join(parts)


def build_sources_footer(chunks: list[dict]) -> str:
    """Deterministic 'Fonti consultate' list built directly from the chunks
    actually retrieved — not left to the model to reproduce. Since we
    already know with certainty which sources were used, generating this in
    code guarantees a correct, complete citation list regardless of how
    reliably the model followed the inline-citation instruction above.
    """
    if not chunks:
        return ""
    seen: list[str] = []
    for c in chunks:
        name = clean_source_name(c["source_file"])
        if name not in seen:
            seen.append(name)
    return "Fonti consultate: " + "; ".join(seen)

def build_prompt(query: str, chunks: list[dict], history: list[tuple[str, str]]) -> list[ChatMessage]:
    """
    Constructs a prompt for the LLM based on the query and context chunks.

    Args:
        query (str): The user's query.
        chunks (list[dict]): A list of context chunks.

    Returns:
        list[ChatMessage]: A list of ChatMessage objects representing the prompt.
    """
    prompt_text = f"CONTESTO:\n{build_context_block(chunks)}\n\nDOMANDA DELLO STUDENTE:\n{query}"
    blocks = [TextBlock(text=prompt_text)]
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