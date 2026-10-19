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
    "Ogni blocco di CONTESTO inizia con '[Fonte: <nome>]': per ogni affermazione, cita "
    "tra parentesi quadre il <nome> del blocco da cui l'hai presa, sostituendolo con "
    "quello reale — non copiare esempi. Se il contesto contiene più fonti, cita tutte "
    "le fonti rilevanti."
)


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


def build_context_block(chunks: list[dict]) -> str:
    """Formats retrieved chunks as labeled "[Fonte: ...]" blocks for the prompt.

    Args:
        chunks (list[dict]): Chunks with "text", "headings", "source_file"
            keys, and optionally "source_url" (only crawled web pages have
            one — PDF/DOCX/PPTX documents never do).

    Returns:
        str: Chunks joined into a single context string.
    """
    parts = []
    for c in chunks:
        source_url = c.get("source_url")
        if source_url:
            label = source_url
        else:
            headings = " > ".join(c.get("headings") or [])
            source_name = clean_source_name(c["source_file"])
            label = f"{source_name} - {headings}"
        parts.append(f"[Fonte: {label}\n{c["text"]}]")
    return "\n\n---\n\n".join(parts)

def build_prompt(query: str, chunks: list[dict], history: list[tuple[str, str]]) -> list[ChatMessage]:
    """Constructs the full chat prompt: system instructions, prior turns, then
    the current query with its retrieved context (and images, if supported).

    Args:
        query (str): The user's query.
        chunks (list[dict]): Retrieved context chunks, with "image_paths" if any.
        history (list[tuple[str, str]]): Prior (question, answer) turns.

    Returns:
        list[ChatMessage]: Messages ready for `llm.chat()`.
    """
    prompt_text = f"CONTESTO:\n{build_context_block(chunks)}\n\nDOMANDA ({settings.assistant_audience.upper()}):\n{query}"
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