import re
from pathlib import Path
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock, ImageBlock

SYSTEM_PROMPT = (
    "Sei un assistente virtuale dell'Università di Bologna. Rispondi in italiano "
    "usando esclusivamente le informazioni presenti nel CONTESTO fornito. "
    "Se il contesto non contiene la risposta, dillo esplicitamente invece di inventare. "
    "Riporta date, importi e numeri esattamente come appaiono nel contesto, senza "
    "arrotondare o parafrasare. Per ogni affermazione, cita la fonte tra parentesi "
    "quadre, es. [Fonte: nome_file]. Se il contesto contiene più fonti, cita tutte le fonti rilevanti."
    " Un esempio di risposta corretta sarebbe: 'La scadenza per la prima rata è il 30 settembre [Fonte: tasse]'."
)

def build_context_block(chunks: list[dict]) -> str:
    """
    Constructs a context block from a list of chunks.

    Args:
        chunks (list[dict]): A list of chunks, where each chunk is a dictionary containing 'text', 'headings', 'source_file', and 'image_paths'.
    """

    parts = []
    for c in chunks:
        headings = " > ".join(c.get("headings") or [])
        parts.append(f"[Fonte: {c["source_file"]} - {headings}\n{c["text"]}]")
    return "\n\n---\n\n".join(parts)

def build_prompt(query: str, chunks: list[dict]) -> list[ChatMessage]:
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
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=blocks)
    ]