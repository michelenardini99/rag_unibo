import asyncio

from chainlit import AskUserMessage, Message, on_chat_start, on_message, user_session, Step, Text
import chainlit as cl
from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question, decompose_question
from generation.generator import build_llm, generate_response
from generation.prompt import build_prompt, clean_source_name
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.query_filters import closest_known_value, course_tipo_laurea_map, extract_query_filters, known_values_for_fields, needs_course_disambiguation
from retrieval.reranker import build_reranker
from retrieval.retriever import _retrieve_locked
import threading


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

lock = threading.Lock()


class _NullStep:
    """No-op stand-in for `chainlit.Step`, used in Standard mode so the
    pipeline logic inside each `async with` block doesn't need to branch on
    the mode itself — `.input`/`.output` assignments are silently absorbed
    instead of rendered, and nothing is shown to the student beyond the
    final answer and its sources.
    """

    def __init__(self):
        self.input = None
        self.output = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _step(name: str, debug: bool):
    """A real `Step` in Debug Mode (visible pipeline stage), or a `_NullStep`
    in Standard mode (same code path, nothing rendered)."""
    return Step(name=name) if debug else _NullStep()


def _format_debug_prompt(messages) -> str:
    """Renders the chat messages actually sent to the generation LLM as
    readable text, for the "Debug Mode" prompt-inspection step — images are
    shown as a placeholder, not inlined.

    Args:
        messages: `ChatMessage` list from `generation.prompt.build_prompt`.

    Returns:
        str: One section per message, role-labeled.
    """
    parts = []
    for m in messages:
        role = getattr(m.role, "value", m.role)
        block_texts = [getattr(b, "text", None) or f"<{type(b).__name__}>" for b in m.blocks]
        parts.append(f"**[{role}]**\n" + "\n".join(block_texts))
    return "\n\n---\n\n".join(parts)



@on_chat_start
async def on_chat_start():
    """Initializes a fresh conversation history. The student's course is no
    longer asked upfront — it's resolved lazily, per message, only when a
    question actually needs it to disambiguate which course's (near-duplicate)
    document is the right one (see `on_message`'s disambiguation step and
    REPORT_ARCHITETTURA.md §4.4).
    """
    user_session.set("history", [])
    user_session.set("default_corso", None)

    await cl.context.emitter.set_modes([
        cl.Mode(
            id="debug_selector",
            name="Modalità di Esecuzione",
            options=[
                cl.ModeOption(
                    id="standard", 
                    name="Standard", 
                    description="Esecuzione normale dell'assistente", 
                    icon="bot", 
                    default=True
                ),
                cl.ModeOption(
                    id="debug", 
                    name="Debug Mode", 
                    description="Mostra prompt intermedi e log tecnici", 
                    icon="bug"
                )
            ]
        )
    ])
    await Message(content="Ciao! Sono il tuo assistente per l'università di Bologna. Come posso aiutarti oggi?").send()


@on_message
async def on_message(msg: Message):
    """Handles one user message end-to-end: condense, retrieve, generate,
    stream the answer back, and update the session's conversation history.

    Args:
        msg (Message): Incoming Chainlit message.
    """
    query = msg.content
    history = user_session.get("history")

    debug = msg.modes.get("debug_selector") == "debug"

    async with _step("Analisi e Condensazione query", debug) as step_condense:
        step_condense.input = query
        # condense_question is a blocking HTTP call; to_thread keeps it off the event loop.
        search_query = await asyncio.to_thread(condense_question, LLM, history, query)

        step_condense.output = f"Query ottimizzata per il Vector DB:\n`{search_query}`"

        sub_queries = await asyncio.to_thread(decompose_question, LLM, search_query)
        if len(sub_queries) > 1:
            step_condense.output += "\nDomanda scomposta in:\n" + "\n".join(f"- `{q}`" for q in sub_queries)

        extracted_filters = await asyncio.to_thread(extract_query_filters, LLM, search_query, KNOWN_VALUES, TIPO_LAUREA_BY_CORSO)
        extracted_corso = extracted_filters.get("corso")
        default_corso = user_session.get("default_corso")
        print(f"[metadata] query={search_query!r} -> extracted_corso={extracted_corso} default_corso={default_corso}")
        if extracted_corso:
            step_condense.output += f"\nCorso riconosciuto nel testo: `{extracted_corso}`"

    async with _step("Ricerca Documenti", debug) as step_retrieve:
        step_retrieve.input = search_query

        if extracted_corso:
            priority_corso = extracted_corso
            user_session.set("default_corso", priority_corso)
            chunks = await asyncio.to_thread(
                _retrieve_locked, lock, sub_queries, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                priority_corso=priority_corso,
            )
        else:
            probe_chunks = await asyncio.to_thread(
                _retrieve_locked, lock, sub_queries, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                priority_corso=None,
            )

            if needs_course_disambiguation(probe_chunks, default_corso):
                known_courses = KNOWN_VALUES.get("corso") or []
                course_list = "\n".join(f"- {c}" for c in known_courses)
                res = await AskUserMessage(
                    content=(
                        "Questa domanda potrebbe riguardare più corsi — per rispondere con precisione "
                        f"ho bisogno di sapere il tuo corso di laurea, tra questi:\n{course_list}"
                    ),
                    timeout=120,
                ).send()
                answer = (res or {}).get("output", "")
                matched = closest_known_value(answer, known_courses)
                print(f"[corso disambiguation] answer={answer!r} -> matched={matched}")
                if matched:
                    user_session.set("default_corso", matched)
                    step_retrieve.output = f"Corso impostato: `{matched}`. Ripeto la ricerca.\n"
                    chunks = await asyncio.to_thread(
                        _retrieve_locked, lock, sub_queries, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                        priority_corso=matched,
                    )
                else:
                    chunks = probe_chunks
                    await Message(
                        content=(
                            f"Non ho riconosciuto \"{answer}\" tra i corsi disponibili — rispondo senza dare "
                            "priorità a un corso specifico."
                        )
                    ).send()
            elif default_corso:
                chunks = await asyncio.to_thread(
                    _retrieve_locked, lock, sub_queries, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                    priority_corso=default_corso,
                )
            else:
                chunks = probe_chunks

        step_retrieve.output = (step_retrieve.output or "") + \
            f"Recuperati e riordinati **{len(chunks)} chunk** rilevanti da `{settings.qdrant_collection}`."

    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source_url = chunk.get("source_url")
        if source_url:
            label = source_url
            content = f"{source_url}\n\n{text}"
        else:
            label = clean_source_name(chunk["source_file"])
            content = text

    final_response = Message(content="", elements=user_session.get("source_elements"))

    if debug:
        async with Step(name="Prompt completo inviato al modello") as step_prompt:
            step_prompt.input = search_query
            step_prompt.output = _format_debug_prompt(build_prompt(search_query, chunks, history))

    response_text = await asyncio.to_thread(generate_response, LLM, search_query, chunks, history)

    for token in response_text.split(" "):
        await final_response.stream_token(token + " ")
        await asyncio.sleep(0.03)

    await final_response.send()

    history.append((search_query, response_text))
    history = history[-settings.max_history_turns:]
    user_session.set("history", history)


