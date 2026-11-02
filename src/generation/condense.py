from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock
from llama_index.llms.openai_like import OpenAILike

from config import settings

CONDENSE_SYSTEM_PROMPT = (
    """Il tuo compito è riformulare l'ultima domanda dello studente in una 
    domanda autonoma e completa, comprensibile senza la cronologia della conversazione, SOLO
    se la domanda fatta dall'utente risulta incompleta e non comprensibile, altrimenti mantieni la
    la domanda cosi com'è.

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
    - Non aggiungere il nome di un corso di laurea che lo studente non ha scritto nella nuova domanda: il corso
        dello studente è gestito a parte.
    - Se il nuovo messaggio non chiede nulla (un saluto, un ringraziamento, un congedo o una conferma come
        "grazie", "ok perfetto", "a presto"), restituiscilo invariato: non è un seguito della domanda precedente.

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
    """Builds the chat messages for the condensation call.

    Args:
        history (list[tuple[str, str]]): Prior (question, answer) turns.
        query (str): The latest, possibly context-dependent, question.

    Returns:
        list[ChatMessage]: System + user messages for `llm.chat()`.
    """
    turns = "\n".join(f"Studente: {q}\nAssistente: {a}" for q, a in history)
    text = f"CRONOLOGIA:\n{turns}\n\nNUOVA DOMANDA: {query}\n\nDomanda riformulata:"
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=CONDENSE_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)]),
    ]

def condense_question(llm: OpenAILike, history: list[tuple[str, str]], query: str) -> str:
    """Rewrites a possibly context-dependent follow-up into a standalone
    question, using the conversation history.

    Args:
        llm (OpenAILike): Model used to condense the question.
        history (list[tuple[str, str]]): Prior (question, answer) turns.
        query (str): The latest question, as typed by the user.

    Returns:
        str: `query` unchanged if `history` is empty (nothing to condense);
        otherwise the model's standalone rewrite.
    """
    if not history:
        return query
    # 64 was sized for a short follow-up ("e per il 2026/27?"); a multi-part
    # compound question condensed against history can legitimately need more
    # room than that — a tighter budget was observed to silently truncate the
    # rewrite mid-sentence, dropping a whole clause before decompose_question
    # ever saw it (see REPORT_ARCHITETTURA.md §5.4/§7).
    response = llm.chat(build_condense_prompt(history, query), max_tokens=200, temperature=0.0)
    return response.message.content.strip()


COURTESY = "CORTESIA"
OFF_TOPIC = "FUORI TEMA"

OFF_TOPIC_REPLY = (
    f"Posso aiutarti solo con questioni relative all'{settings.assistant_institution}: "
    f"{settings.assistant_scope}."
)
COURTESY_REPLY = "Se hai domande sull'università, chiedi pure."

DECOMPOSE_SYSTEM_PROMPT = (
    """Il tuo compito è analizzare un messaggio di uno studente.

    Prima stabilisci se il messaggio contiene almeno una richiesta:
    - Se è solo un saluto, un ringraziamento, un congedo o una conferma (es. "ciao", "grazie mille",
      "ok perfetto", "a presto"), rispondi su un'unica riga con "CORTESIA:" seguito da una breve risposta
      cordiale in italiano, che inviti a fare domande sull'università.
    - Se chiede qualcosa che non ha alcun legame possibile con l'università o con la vita da studente
      (es. "chi ha vinto la Serie A?", "come si cucina la carbonara?"), rispondi solo "FUORI TEMA".
      È invece una domanda tutto ciò che può riguardare uno studente universitario: corsi, esami, lezioni e
      qualunque argomento di studio (anche tecnico o teorico), persone che possono essere docenti o personale
      dell'ateneo, tasse, borse, iscrizioni, procedure, servizi, alloggi, trasporti, mense e agevolazioni nella
      città universitaria. Nel dubbio, trattalo come una domanda.
    - Se il messaggio contiene una domanda insieme a un saluto o a un ringraziamento (es. "grazie, e quando
      scade la seconda rata?"), ignora il saluto e analizza la domanda come indicato sotto.

    Se è una domanda, determina se contiene
    più richieste informative distinte (es. unite da "e", "oppure", o elencate in sequenza),
    che nel corpus documentale dell'ateneo potrebbero essere coperte da documenti diversi.

    Regole:
    - Se la domanda riguarda un solo argomento/richiesta, restituisci SOLO quella domanda,
      invariata, su un'unica riga.
    - Se la domanda contiene più richieste distinte, dividila in altrettante domande autonome
      e complete, una per riga, ciascuna comprensibile da sola: ripeti in ciascuna il
      soggetto/corso/procedura/contesto necessario, senza pronomi o riferimenti che rimandano alle altre righe.
    - Attenzione alle richieste che, nella domanda originale, dipendono implicitamente dalla
      procedura o dall'argomento nominato in una clausola precedente (es. "...per presentare X,
      quanto costa Y" dove Y è inteso come "Y di/per X", non Y in generale): la sotto-domanda
      risultante deve nominare esplicitamente quella procedura, non solo il corso, altrimenti
      diventa una domanda diversa e più generica di quella posta.
    - Non rispondere alle domande, non aggiungere spiegazioni, numerazione, elenchi puntati o altro testo.
    - Mantieni la lingua originale (italiano).

    Esempio 1 (una sola richiesta):
    Domanda: Qual è il voto minimo per superare la prova finale?
    Output:
    Qual è il voto minimo per superare la prova finale?

    Esempio 2 (due richieste distinte, argomenti già autonomi):
    Domanda: Quali sono i requisiti per presentare domanda di laurea in Psicologia scolastica e di comunità, e quali invece per poter accedere al corso?
    Output:
    Quali sono i requisiti per presentare domanda di laurea nel corso di laurea magistrale in Psicologia scolastica e di comunità?
    Quali sono i requisiti per accedere al corso di laurea magistrale in Psicologia scolastica e di comunità?

    Esempio 3 (una richiesta dipende implicitamente dalla procedura nominata nella clausola precedente):
    Domanda: Quali sono i requisiti per presentare domanda di laurea in Psicologia scolastica e di comunità, quanto costa la marca da bollo, e qual è la scala di voto della prova finale?
    Output:
    Quali sono i requisiti per presentare domanda di laurea nel corso di laurea magistrale in Psicologia scolastica e di comunità?
    Quanto costa la marca da bollo per la domanda di laurea nel corso di laurea magistrale in Psicologia scolastica e di comunità?
    Qual è la scala di voto della prova finale nel corso di laurea magistrale in Psicologia scolastica e di comunità?

    Esempio 4 (nessuna richiesta):
    Domanda: grazie mille, gentilissimo
    Output:
    CORTESIA: Prego! Se hai altre domande sull'università, chiedi pure.

    Esempio 5 (estraneo all'università):
    Domanda: mi consigli una serie tv da guardare stasera?
    Output:
    FUORI TEMA

    Ora analizza:

    Domanda: {query}
    Output:"""
)


def build_decompose_prompt(query: str) -> list[ChatMessage]:
    """Builds the chat messages for the query-decomposition call.

    Args:
        query (str): The (condensed, standalone) question to analyze.

    Returns:
        list[ChatMessage]: System + user messages for `llm.chat()`.
    """
    text = f"Domanda: {query}\nOutput:"
    return [
        ChatMessage(role=MessageRole.SYSTEM, blocks=[TextBlock(text=DECOMPOSE_SYSTEM_PROMPT)]),
        ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text=text)]),
    ]


def decompose_question(llm: OpenAILike, query: str, max_subquestions: int = 3) -> tuple[list[str], str | None]:
    """Splits a question into independently-retrievable sub-questions when it
    bundles multiple distinct informational needs that a single combined
    query's retrieval pass could fail to cover jointly (see
    REPORT_ARCHITETTURA.md §5.4/§6.3: one compound query reranked as a whole
    can let one sub-topic's chunks crowd the other out of the top-k).

    The same call recognizes messages that aren't questions, so no extra model
    call is needed for them: greetings and thanks get the model's own short
    reply, and questions unrelated to the university a fixed one. Before, a
    word list caught greetings ahead of retrieval and a separate classifier
    caught off-topic questions after it, when nothing was found; the list
    missed any other wording, and the classifier, asked about a question,
    judged most greetings pertinent, so "ciao" got "Qual è il tuo corso?".

    Args:
        llm (OpenAILike): Model used to decompose the question.
        query (str): The (condensed, standalone) question to analyze.
        max_subquestions (int): Upper bound on how many sub-questions are
            kept, to cap the retrieval fan-out cost.

    Returns:
        tuple[list[str], str | None]: The sub-questions, and None — or, if
        the message isn't a question, `[query]` and the reply to send instead
        of searching. The sub-questions are `[query]` if the model judges it
        a single request (or on an unparseable/empty response — fail safe to
        the original behavior); otherwise the standalone sub-questions,
        capped at `max_subquestions`.
    """
    response = llm.chat(build_decompose_prompt(query), max_tokens=200, temperature=0.0)
    lines = [line.strip() for line in (response.message.content or "").strip().splitlines() if line.strip()]
    if not lines:
        return [query], None
    label = lines[0].upper()
    if label.startswith(COURTESY):
        return [query], lines[0][len(COURTESY):].lstrip(" :") or COURTESY_REPLY
    if label.startswith(OFF_TOPIC):
        return [query], OFF_TOPIC_REPLY
    return lines[:max_subquestions], None
