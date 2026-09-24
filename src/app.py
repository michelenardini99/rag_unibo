import asyncio

from chainlit import AskUserMessage, Message, on_chat_start, on_message, user_session, Step, Text
from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question
from generation.generator import build_llm, generate_response
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.query_filters import closest_known_value, course_tipo_laurea_map, extract_query_filters, known_values_for_fields
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve


EMBED_MODEL = build_embedding_model(device_id=settings.embeddings_device_id)
RERANKER = build_reranker(device_id=settings.embeddings_device_id)
LLM = build_llm(
    base_url=settings.vllm_base_url,
    model=settings.generation_model,
    context_window=settings.vllm_max_model_len,
)
VECTOR_STORE = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
DOCSTORE = SimpleDocumentStore.from_persist_path(settings.data_chunked_dir / "docstore.json")
KNOWN_VALUES = known_values_for_fields(VECTOR_STORE, settings.qdrant_collection)
TIPO_LAUREA_BY_CORSO = course_tipo_laurea_map(VECTOR_STORE, settings.qdrant_collection, KNOWN_VALUES.get("corso") or [])

@on_chat_start
async def on_chat_start():
    """Initializes a fresh conversation history and asks the student's course
    once, so later questions that don't name a course (e.g. "quanti iscritti
    quest'anno?") default to it instead of a Qdrant filter risking mixing
    same-shaped stats from other courses (see retrieve()'s filters and
    query_filters.py's "fail open, no filter" default).
    """
    history: list[tuple[str, str]] = []
    user_session.set("history", history)

    default_filters: dict[str, str] = {}
    known_courses = KNOWN_VALUES.get("corso") or []
    if known_courses:
        course_list = "\n".join(f"- {c}" for c in known_courses)
        res = await AskUserMessage(
            content=(
                "Ciao! Sono il tuo assistente per l'università di Bologna. "
                f"Per rispondere in modo più preciso, dimmi il tuo corso di laurea tra questi:\n{course_list}"
            ),
            timeout=120,
        ).send()
        answer = (res or {}).get("output", "")
        matched = closest_known_value(answer, known_courses)
        if matched:
            default_filters["corso"] = matched
            tipo_laurea = TIPO_LAUREA_BY_CORSO.get(matched)
            if tipo_laurea:
                default_filters["tipo_laurea"] = tipo_laurea
            await Message(content=f"Corso impostato: **{matched}**. Chiedimi pure.").send()
        else:
            await Message(
                content=(
                    f"Non ho riconosciuto \"{answer}\" tra i corsi disponibili — procederò senza corso di "
                    "riferimento, quindi specificalo direttamente nelle domande dove serve per non rischiare "
                    "di mischiare dati di corsi diversi."
                )
            ).send()
        print(f"[corso onboarding] answer={answer!r} -> {default_filters}")
    else:
        await Message(content="Ciao! Sono il tuo assistente per l'università di Bologna. Come posso aiutarti oggi?").send()

    user_session.set("default_filters", default_filters)


@on_message
async def on_message(msg: Message):
    """Handles one user message end-to-end: condense, retrieve, generate,
    stream the answer back, and update the session's conversation history.

    Args:
        msg (Message): Incoming Chainlit message.
    """
    query = msg.content
    history = user_session.get("history")

    old_source_elem = user_session.get("source_elements") or []
    for elem in old_source_elem:
        await elem.remove()
    async with Step(name="Analisi e Condensazione query") as step_condense:
        step_condense.input = query
        # condense_question is a blocking HTTP call; to_thread keeps it off the event loop.
        search_query = await asyncio.to_thread(condense_question, LLM, history, query)

        step_condense.output = f"Query ottimizzata per il Vector DB:\n`{search_query}`"

        extracted_filters = await asyncio.to_thread(extract_query_filters, LLM, search_query, KNOWN_VALUES, TIPO_LAUREA_BY_CORSO)
        default_filters = user_session.get("default_filters") or {}
        filters = extracted_filters or default_filters
        print(f"[metadata filters] query={search_query!r} -> extracted={extracted_filters} used={filters}")
        if filters:
            step_condense.output += f"\nFiltri applicati: `{filters}`"

    async with Step(name="Ricerca Documenti") as step_retrieve:
        step_retrieve.input = search_query
        chunks = await asyncio.to_thread(
            retrieve, search_query, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
            filters=filters,
        )

        step_retrieve.output = f"Recuperati e riordinati **{len(chunks)} chunk** rilevanti da `{settings.qdrant_collection}`."

    source_elem = []
    for i, chunk in enumerate(chunks):
        source_file = chunk["source_file"]
        text = chunk["text"]

        source_elem.append(
            Text(
                name = f"Fonte {i+1}: {source_file}",
                content = text,
                display = 'side'
            )
        )

    user_session.set("source_elements", source_elem)

    final_response = Message(content="", elements=user_session.get("source_elements"))

    # search_query (condensed), not the raw query: self-contained and
    # consistent with what was actually retrieved for it.
    response_text = await asyncio.to_thread(generate_response, LLM, search_query, chunks, history)

    for token in response_text.split(" "):
        await final_response.stream_token(token + " ")
        await asyncio.sleep(0.03)

    await final_response.send()

    history.append((search_query, response_text))
    history = history[-settings.max_history_turns:]
    user_session.set("history", history)


