from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike

CONDENSE_SYSTEM_PROMPT = (
    """Il tuo compito è riformulare l'ultima domanda dello studente in una 
    domanda autonoma e completa, comprensibile senza la cronologia della conversazione.

    Regole:
    - Mantieni il significato e la struttura originale della domanda precedente
    - Sostituisci SOLO l'elemento che cambia esplicitamente nella nuova domanda
    - Non introdurre significati diversi da quelli impliciti nella conversazione
    - Non rispondere alla domanda, non aggiungere spiegazioni
    - Restituisci solo la domanda riformulata, in italiano
    - Prima di riformulare, identifica se la nuova domanda contiene una negazione 
        (es. "non", "senza", "eccetto", "a differenza di", "diverso da"). 
        Se presente, assicurati che la negazione sia esplicitamente preservata 
        nella domanda riformulata.
    - Se la nuova domanda ha una struttura diversa da quella precedente, mantieni comunque nell'domanda riformulata 
        l'argomento/le entità principali della conversazione (es. nome di corso, procedura, programma) necessarie a capirla da sola.

    Esempi:

    Cronologia:
    Utente: Entro quando scade il pagamento della prima rata per gli studenti iscritti a corsi di laurea magistrale a ciclo unico?
    Assistente: [risposta]
    Nuova domanda: E per quelli non a ciclo unico?
    Domanda riformulata: Entro quando scade il pagamento della prima rata per gli studenti NON iscritti a corsi di laurea magistrale a ciclo unico?

    Cronologia:
    Utente: Entro quando scade il pagamento della prima rata per le lauree magistrali a ciclo unico?
    Assistente: [risposta]
    Nuova domanda: E per chi non è a ciclo unico?
    Domanda riformulata: Entro quando scade il pagamento della prima rata per chi non è iscritto a lauree magistrali a ciclo unico?

    Cronologia:
    Utente: Qual è la procedura per richiedere l'equipollenza di un titolo di laurea conseguito all'estero?
    Assistente: [risposta]
    Utente: E chi valuta la documentazione presentata?
    Assistente: [risposta]
    Nuova domanda: E in quanto tempo arriva la risposta?
    Domanda riformulata: In quanto tempo arriva la risposta alla richiesta di equipollenza di un titolo di laurea conseguito all'estero?

    Ora riformula:

    Cronologia: {cronologia}

    Nuova domanda: {query}
    Domanda riformulata:"""
)

def build_condense_prompt(history: list[tuple[str, str]], query: str) -> list[ChatMessage]:
    turns = "\n".join(f"Studente: {q}\nAssistente: {a}" for q, a in history)
    text = f"CRONOLOGIA:\n{turns}\n\nNUOVA DOMANDA: {query}\n\nDomanda riformulata:"
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=CONDENSE_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)]),
    ]

def condense_question(llm: OpenAILike, history: list[tuple[str, str]], query: str) -> str:
    if not history:
        return query
    response = llm.chat(build_condense_prompt(history, query), max_tokens=64, temperature=0.0)
    return response.message.content.strip()