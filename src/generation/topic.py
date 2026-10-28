from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike

from config import settings

OFF_TOPIC = "FUORI TEMA"

TOPIC_SYSTEM_PROMPT = (
    f"Sei il filtro di un assistente per gli studenti di {settings.assistant_institution}. Ricevi una "
    "domanda di uno studente e devi stabilire se è chiaramente estranea all'università.\n\n"
    "Considera PERTINENTE tutto ciò che può riguardare la vita di uno studente universitario: corsi, "
    "esami, lezioni e qualunque argomento di studio (anche tecnico o teorico), persone che possono "
    "essere docenti o personale dell'ateneo, tasse, borse, iscrizioni, procedure, servizi, alloggi, "
    "trasporti, mense e agevolazioni nella città universitaria.\n\n"
    f"Rispondi {OFF_TOPIC} solo se la domanda non ha alcun legame possibile con l'università o con la "
    "vita da studente. Nel dubbio rispondi PERTINENTE.\n\n"
    f"Rispondi con una sola parola: PERTINENTE oppure {OFF_TOPIC}."
)

OFF_TOPIC_REPLY = (
    f"Posso aiutarti solo con questioni relative all'{settings.assistant_institution}: "
    f"{settings.assistant_scope}."
)


def is_off_topic(llm: OpenAILike, question: str) -> bool:
    """True if `question` has nothing to do with the university.

    Asked only when retrieval found nothing to answer from: the app would
    otherwise ask the student for their course ("Chi ha vinto la Serie A?"
    got "Qual è il tuo corso?"), since with no course known it can't tell a
    course-specific question from an unrelated one.

    Args:
        llm (OpenAILike): The generation model.
        question (str): The (condensed) question.

    Returns:
        bool: True only on an explicit off-topic verdict — an unclear reply
        counts as on-topic, falling back to asking for the course.
    """
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=TOPIC_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=question)]),
    ]
    reply = llm.chat(messages, max_tokens=5, temperature=0.0).message.content or ""
    return OFF_TOPIC in reply.upper()
