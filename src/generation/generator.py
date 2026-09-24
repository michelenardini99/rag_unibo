from llama_index.llms.openai_like import OpenAILike
from generation.prompt import build_prompt
from config import settings

NO_CONTEXT_RESPONSE = (
    f"Non ho trovato informazioni pertinenti per rispondere a questa domanda. "
    f"Ti consiglio di consultare il sito ufficiale di {settings.assistant_institution} "
    f"o di contattare l'ufficio competente."
)


def build_llm(base_url: str, model: str = "generation-llm", context_window: int = 8192, timeout: float = 60.0) -> OpenAILike:
    """
    Builds an OpenAILike LLM instance.

    Args:
        base_url (str): The base URL for the LLM API.
        model (str): The model name to use. Default is "generation-llm".
        context_window (int): The context window size. Default is 8192.
        timeout (float): Request timeout in seconds (llama_index default is
            60s). Batch/eval callers running several requests concurrently
            against a single-GPU server should raise this — under load a
            request can legitimately queue behind others for longer than 60s
            without anything being actually stuck.

    Returns:
        OpenAILike: An instance of the OpenAILike LLM.
    """
    return OpenAILike(
        api_base=base_url,
        model=model,
        api_key="not-needed",
        is_chat_model=True,
        context_window=context_window,
        timeout=timeout,
    )

def generate_response(llm: OpenAILike, query: str, chunks: list[dict], history: list[tuple[str, str]] | None = None) -> str:
    """Generates a grounded response from the LLM given the query and retrieved context.

    Args:
        llm (OpenAILike): The LLM instance to use for generation.
        query (str): The (condensed, standalone) question to answer — pass
            the same query used for retrieval, so the model isn't asked to
            resolve a raw follow-up fragment (e.g. "e per il 2026/27?")
            against `chunks` that were retrieved for its condensed form.
        chunks (list[dict]): Retrieved context chunks.
        history (list[tuple[str, str]] | None): Prior (question, answer) turns,
            included in the prompt for conversational continuity.

    Returns:
        str: The generated response.
    """
    if not chunks:
        # Don't hand the model an empty CONTESTO block and trust it to admit
        # that: observed in practice to instead fabricate a plausible-looking
        # answer (even a fake "[Fonte: ...]" citation) rather than say so, the
        # system prompt's instruction to the contrary notwithstanding. An
        # empty retrieval result is answered deterministically instead.
        return f"{NO_CONTEXT_RESPONSE}\n"

    # Deterministic on purpose: factual QA over retrieved context doesn't need
    # creativity, and a fixed temperature keeps answers (and eval scores) reproducible.
    response = llm.chat(build_prompt(query, chunks, history), temperature=0.0)
    answer = response.message.content
    return f"{answer}\n"