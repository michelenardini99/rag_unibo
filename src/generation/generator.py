from llama_index.llms.openai_like import OpenAILike
from generation.prompt import build_prompt

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

def generate_response(llm: OpenAILike, query: str, chunks: list[dict]) -> str:
    """
    Generates a response from the LLM based on the provided query and context chunks.

    Args:
        llm (OpenAILike): The LLM instance to use for generation.
        query (str): The user's query.
        chunks (list[dict]): A list of context chunks.

    Returns:
        str: The generated response from the LLM.
    """
    response = llm.chat(build_prompt(query, chunks))
    return response.message.content