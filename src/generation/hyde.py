from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike

from config import settings

HYDE_SYSTEM_PROMPT = (
    f"Ricevi la domanda di uno studente dell'{settings.assistant_institution}. Scrivi il breve "
    "paragrafo (2-3 frasi) di una pagina web dell'ateneo che risponderebbe alla domanda, con la "
    "terminologia amministrativa e accademica ufficiale che quella pagina userebbe (es. \"learning "
    "agreement\", \"autocertificazione\", \"rinnovo dell'iscrizione\").\n\n"
    "Se non conosci i dettagli (date, importi, nomi), inventali in modo plausibile: il paragrafo "
    "serve solo a cercare la pagina vera, non viene mostrato allo studente.\n\n"
    "Non nominare corsi di laurea specifici. Non aggiungere premesse, avvertenze o titoli: "
    "restituisci solo il paragrafo, in italiano."
)


def hyde_query(llm: OpenAILike, question: str) -> str:
    """Search query for a question whose retrieval found nothing: the
    question followed by a hypothetical answer passage (HyDE).

    Students and pages name things differently ("test d'ingresso" / "prova
    di ammissione"): the dense search still ranks the right chunk first, but
    the reranker scores it 0.02 against the question and 0.99 against the
    page's wording, below the threshold, so nothing is returned. The
    hypothetical passage brings the pages' vocabulary into the query; its
    made-up facts don't matter, only its wording. The question stays in, so
    the reranker still judges relevance to what was asked.

    Args:
        llm (OpenAILike): Model that writes the passage.
        question (str): The (condensed) question that found nothing.

    Returns:
        str: The question and the passage, or the question alone if the
        model returns nothing.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=HYDE_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=question)]),
    ]
    # Short on purpose: the reranker sees query + chunk within its 512-token window.
    response = llm.chat(messages, max_tokens=120, temperature=0.0)
    passage = (response.message.content or "").strip()
    return f"{question}\n{passage}" if passage else question
