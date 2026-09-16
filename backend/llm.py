"""
One place that builds the chat model and the embedding model.

Interview line: "The rest of the app never imports a provider SDK. It calls
get_llm(). Swapping Mistral for Gemini or a local Ollama model is one line in
.env, because every provider already implements the same LangChain Runnable
interface."

@lru_cache means the client is built once per process, not on every request.
"""

from functools import lru_cache

from backend import config


@lru_cache(maxsize=None)
def get_llm():
    """Return the chat model chosen by LLM_PROVIDER in .env."""
    provider = config.LLM_PROVIDER

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=config.MISTRAL_MODEL,
            temperature=config.TEMPERATURE,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GOOGLE_MODEL,
            temperature=config.TEMPERATURE,
        )

    if provider == "ollama":
        # Fully offline. No API key, no rate limit, no internet.
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL,
            temperature=config.TEMPERATURE,
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        endpoint = HuggingFaceEndpoint(
            repo_id=config.HF_REPO_ID,
            task="text-generation",
            temperature=config.TEMPERATURE,
            max_new_tokens=512,
        )
        return ChatHuggingFace(llm=endpoint)

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Use one of: mistral, google, ollama, huggingface."
    )


@lru_cache(maxsize=None)
def get_embeddings():
    """
    Return the embedding model chosen by EMBEDDING_PROVIDER in .env.

    Default is local (all-MiniLM-L6-v2, ~80MB, 384 dimensions). Embeddings run
    on every answer we score, so an API round-trip there would be the latency
    bottleneck and would cost money per call.

    On Vercel, set EMBEDDING_PROVIDER=google to avoid the 80MB download.
    """
    if config.EMBEDDING_PROVIDER == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=config.GOOGLE_EMBEDDING_MODEL)

    # Local sentence-transformers — only works when the package is installed.
    # Not available on Vercel (requires C++ native libs).
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.LOCAL_EMBEDDING_MODEL)
    except ImportError:
        # Fallback to Google embeddings if local model is unavailable
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=config.GOOGLE_EMBEDDING_MODEL)
