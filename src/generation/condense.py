from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike

CONDENSE_SYSTEM_PROMPT = (
    "Il tuo unico compito è riformulare l'ultima domanda dello studente in una "
    "domanda autonoma e completa, comprensibile senza la cronologia della "
    "conversazione. Non rispondere alla domanda, non aggiungere spiegazioni: "
    "restituisci solo la domanda riformulata, in italiano."
)

def build_condense_prompt(history: list[tuple[str, str]], query: str) -> list[ChatMessage]:
    turns = "\n".join(f"Studente: {q}\nAssistente: {a}" for q, a in history)
    text = f"CRONOLOGIA:\n{turns}\n\nULTIMA DOMANDA: {query}\n\nDomanda riformulata:"
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=CONDENSE_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)]),
    ]

def condense_question(llm: OpenAILike, history: list[tuple[str, str]], query: str) -> str:
    if not history:
        return query  # nothing to condense against, skip the extra LLM call
    response = llm.chat(build_condense_prompt(history, query), max_tokens=64, temperature=0.0)
    return response.message.content.strip()