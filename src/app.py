import asyncio

from chainlit import AskUserMessage, Message, on_chat_start, on_message, user_session, Step, Text
from config import settings
from embedding.embedder import build_embedding_model
from generation.condense import condense_question
from generation.generator import build_llm, generate_response
from llama_index.core.storage.docstore import SimpleDocumentStore
from qdrant_client import QdrantClient
from retrieval.reranker import build_reranker
from retrieval.retriever import retrieve



@on_chat_start
async def on_chat_start():
    embed_model = await asyncio.to_thread(build_embedding_model, device_id=settings.embeddings_device_id)
    reranker = await asyncio.to_thread(build_reranker, device_id=settings.embeddings_device_id)
    llm = build_llm(
        base_url=settings.vllm_base_url,
        model=settings.generation_model,
        context_window=settings.vllm_max_model_len,
    )
    client = QdrantClient(url=settings.qdrant_url, grpc_port=settings.qdrant_grpc_port, prefer_grpc=True)
    docstore = await asyncio.to_thread(SimpleDocumentStore.from_persist_path, settings.data_chunked_dir)
    history: list[tuple[str, str]] = []
    user_session.set("embed", embed_model)
    user_session.set("reranker", reranker)
    user_session.set("llm", llm)
    user_session.set("vector_store", client)
    user_session.set("docstore", docstore)
    user_session.set("history", history)


    await Message(content="Ciao! Sono il tuo assistente per l'università di Bologna. Come posso aiutarti oggi?").send()


@on_message
async def on_message(msg: Message):
    query = msg.content
    history = user_session.get("history")

    old_source_elem = user_session.get("source_elements") or []
    for elem in old_source_elem:
        await elem.remove()
    async with Step(name="Analisi e Condensazione query") as step_condense:
        step_condense.input = query
        llm = user_session.get("llm")
        search_query = await asyncio.to_thread(condense_question, llm, history, query)

        step_condense.output = f"Query ottimizzata per il Vector DB:\n`{search_query}`"

    async with Step(name="Ricerca Documenti") as step_retrieve:
        step_retrieve.input = search_query
        vector_store = user_session.get("vector_store")
        embed_model = user_session.get("embed")
        reranker = user_session.get("reranker")
        docstore = user_session.get("docstore")
        chunks = await asyncio.to_thread(retrieve, search_query, vector_store, settings.qdrant_collection, embed_model, reranker, docstore)

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

    response_text = await asyncio.to_thread(generate_response, llm, query, chunks, history)

    for token in response_text.split(" "):
        await final_response.stream_token(token + " ")
        await asyncio.sleep(0.03)

    await final_response.send()

    history.append((query, response_text))
    history = history[-settings.max_history_turns:]
    user_session.set("history", history)


