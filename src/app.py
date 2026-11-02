import asyncio

from chainlit import AskUserMessage, Message, on_chat_start, on_message, user_session, Step, Text
import chainlit as cl
from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question, decompose_question
from generation.hyde import hyde_query
from generation.generator import astream_response, build_llm
from generation.prompt import (CitationStripper, build_prompt, course_display_name, format_sources, question_for_answer,
                               strip_citations)
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.query_filters import (course_tipo_laurea_map, extract_query_filters, known_values_for_fields,
                                     match_course_answer, names_course_heads)
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

        # The course travels through the session, not through the rewrite: a
        # rewrite naming a course the student didn't name now, other than the
        # session's, is dropped for the student's own message (it once revived
        # "Ingegneria Informatica" from four turns back, against the session's
        # Psicologia, and the answer refused over the mismatch).
        known_courses = KNOWN_VALUES.get("corso") or []
        allowed = names_course_heads(query, known_courses) | {user_session.get("default_corso")}
        if names_course_heads(search_query, known_courses) - allowed:
            print(f"[condense] rewrite {search_query!r} names a course the student didn't -> using {query!r}")
            search_query = query

        step_condense.output = f"Query ottimizzata per il Vector DB:\n`{search_query}`"

        # Both depend only on search_query, not on each other: run them
        # concurrently so their latencies overlap instead of adding up.
        (sub_queries, reply), extracted_filters = await asyncio.gather(
            asyncio.to_thread(decompose_question, LLM, search_query),
            # The course must be named by the student (`query`), not only by the
            # condensed rewrite; one named earlier is kept as default_corso.
            asyncio.to_thread(extract_query_filters, LLM, search_query, KNOWN_VALUES, TIPO_LAUREA_BY_CORSO, query),
        )
        if len(sub_queries) > 1:
            step_condense.output += "\nDomanda scomposta in:\n" + "\n".join(f"- `{q}`" for q in sub_queries)
        if reply:
            step_condense.output += "\nNessuna domanda sull'università: nessuna ricerca."

        extracted_corso = extracted_filters.get("corso")
        default_corso = user_session.get("default_corso")
        print(f"[metadata] query={search_query!r} -> extracted_corso={extracted_corso} default_corso={default_corso}")
        if extracted_corso:
            step_condense.output += f"\nCorso riconosciuto nel testo: `{extracted_corso}`"

    # Greetings, thanks and off-topic questions: a reply without searching (a
    # "buongiorno" searched found a medical-specialisation page, listed as its
    # source), not added to history (where the condensation took the reply
    # for a question), and the course named, if any, isn't remembered.
    if reply:
        await Message(content=reply).send()
        return

    async with _step("Ricerca Documenti", debug) as step_retrieve:
        step_retrieve.input = search_query

        # A course the student names becomes the session's course, kept for the
        # rest of the conversation until they name another one.
        if extracted_corso:
            user_session.set("default_corso", extracted_corso)
        corso = user_session.get("default_corso")

        # Per question, so a repeated search (course chosen after "nothing
        # found") asks the model only once.
        hyde: dict[str, str] = {}

        async def retrieve_in(course: str | None, queries: list[str]) -> list[dict]:
            # Only the university-wide documents, plus the course's if one is known.
            return await asyncio.to_thread(
                _retrieve_locked, lock, queries, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                priority_corso=course, scope=[course] if course else [],
            )

        async def search(course: str | None) -> tuple[list[dict], list[str]]:
            """Retrieves for the sub-questions; if nothing clears the threshold,
            retries once with a HyDE query (`generation.hyde`), which brings the
            pages' wording to a colloquial question.

            Only with a course: with none, nothing found means the answer may be
            a course's, and the student is asked for it — a made-up passage
            found loosely related university pages instead ("quando c'è il test
            d'ingresso?" answered from the masters' and ISEE pages).

            Returns:
                tuple[list[dict], list[str]]: The chunks and the queries that found them.
            """
            chunks = await retrieve_in(course, sub_queries)
            if chunks or not course:
                return chunks, sub_queries
            if search_query not in hyde:
                hyde[search_query] = await asyncio.to_thread(hyde_query, LLM, search_query)
            queries = [hyde[search_query]]
            chunks = await retrieve_in(course, queries)
            print(f"[hyde] {search_query!r} -> {len(chunks)} chunk con {hyde[search_query]!r}")
            step_retrieve.output = (step_retrieve.output or "") + \
                f"Nessun risultato: ricerca ripetuta con HyDE:\n`{hyde[search_query]}`\n"
            return chunks, queries

        chunks, searched = await search(corso)
        nothing_found = not chunks
        if not corso and chunks:
            # With no course yet, some university page nearly always clears the
            # threshold ("Conciliare studio e lavoro" for "chi è il coordinatore
            # del corso?", the fees page for "scadenze delle domande di laurea"):
            # the university has nothing for the question when the courses'
            # pages are more relevant than its best one.
            all_chunks = await asyncio.to_thread(
                _retrieve_locked, lock, searched, VECTOR_STORE, settings.qdrant_collection, EMBED_MODEL, RERANKER, DOCSTORE,
                priority_corso=None, scope=None,
            )
            best_course = max((c["score"] for c in all_chunks if c.get("corso")), default=0.0)
            nothing_found = best_course > max(c["score"] for c in chunks)
        if nothing_found:
            # Nothing in scope: the answer may be another course's (or, with no
            # course yet, a course's rather than the university's).
            if corso:
                others = "\n".join(f"- {course_display_name(c)}" for c in known_courses if c != corso)
                ask = (
                    f"Non ho trovato informazioni su questo argomento per il corso di "
                    f"{course_display_name(corso)}. La domanda riguardava un altro corso? Se sì, "
                    f"indicalo tra questi:\n{others}\n\nAltrimenti rispondi \"no\"."
                )
            else:
                courses = "\n".join(f"- {course_display_name(c)}" for c in known_courses)
                ask = (
                    "Non ho trovato questa informazione tra i documenti generali dell'ateneo: potrebbe "
                    f"dipendere dal tuo corso di laurea. Qual è il tuo corso, tra questi?\n{courses}"
                )
            res = await AskUserMessage(content=ask, timeout=120).send()
            answer = (res or {}).get("output", "")
            matched = match_course_answer(answer, known_courses)
            print(f"[corso] nothing found for corso={corso}; answer={answer!r} -> matched={matched}")
            if matched and matched != corso:
                user_session.set("default_corso", matched)
                corso = matched
                # The rewrite may name the previous course ("Che ruolo ha Franco
                # Callegati nel corso di Psicologia...?"): search and answer the
                # student's own words for the new one.
                search_query, sub_queries = query, [query]
                step_retrieve.output = f"Corso impostato: `{matched}`. Ripeto la ricerca.\n"
                chunks, _ = await search(matched)
            elif not corso and chunks:
                # No course given: answer from the university pages found.
                pass
            else:
                chunks = []

        step_retrieve.output = (step_retrieve.output or "") + \
            f"Recuperati e riordinati **{len(chunks)} chunk** rilevanti da `{settings.qdrant_collection}`."

    if not chunks:
        where = f"per il corso di {course_display_name(corso)}" if corso else "nei documenti disponibili"
        await Message(content=f"Non ho trovato informazioni su questo argomento {where}.").send()
        return

    final_response = Message(content="")
    answer_question = question_for_answer(search_query, sub_queries)
    # The course retrieval just prioritized, if any (extracted, chosen or remembered).
    student_corso = user_session.get("default_corso")

    if debug:
        async with Step(name="Prompt completo inviato al modello") as step_prompt:
            step_prompt.input = answer_question
            step_prompt.output = _format_debug_prompt(build_prompt(answer_question, chunks, history, student_corso))

    response_text = ""
    stripper = CitationStripper()
    async for token in astream_response(LLM, answer_question, chunks, history, student_corso):
        response_text += token
        if visible := stripper.feed(token):
            await final_response.stream_token(visible)
    if tail := stripper.flush():
        await final_response.stream_token(tail)
    # Listed by the code, not the model: links come from the chunks' metadata.
    sources = format_sources(response_text, chunks)
    if sources:
        await final_response.stream_token(sources)

    await final_response.send()

    # Without "[n]": next turn's numbers refer to different chunks.
    history.append((search_query, strip_citations(response_text)))
    history = history[-settings.max_history_turns:]
    user_session.set("history", history)


