from llama_index.llms.openai_like import OpenAILike
from generation.prompt import build_prompt
from config import settings

def build_llm(base_url: str, model: str = "generation-llm", context_window: int = 8192) -> OpenAILike:
    """
    Builds an OpenAILike LLM instance.

    Args:
        base_url (str): The base URL for the LLM API.
        model (str): The model name to use. Default is "generation-llm".
        context_window (int): The context window size. Default is 8192.

    Returns:
        OpenAILike: An instance of the OpenAILike LLM.
    """
    return OpenAILike(
        api_base=base_url,
        model=model,
        api_key="not-needed",
        is_chat_model=True,
        context_window=context_window
    )

def generate_response(llm: OpenAILike, query: str, chunks: list[dict], history: list[tuple[str, str]] | None = None) -> str:
    """Generates a grounded response from the LLM given the query and retrieved context.

    Args:
        llm (OpenAILike): The LLM instance to use for generation.
        query (str): The user's original query (not the condensed search query).
        chunks (list[dict]): Retrieved context chunks.
        history (list[tuple[str, str]] | None): Prior (question, answer) turns,
            included in the prompt for conversational continuity.

    Returns:
        str: The generated response.
    """

    # Deterministic on purpose: factual QA over retrieved context doesn't need
    # creativity, and a fixed temperature keeps answers (and eval scores) reproducible.
    response = llm.chat(build_prompt(query, chunks, history), temperature=0.0)
    answer = response.message.content
    return f"{answer}\n"