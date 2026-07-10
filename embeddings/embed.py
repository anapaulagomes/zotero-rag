import logging
import os

# Quiet LiteLLM before importing it: its import-time pre-load warnings about optional
# backends fire during the import itself, so the logger level must be set first (same
# reasoning as retrieval/llm.py).
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from litellm import embedding  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.drop_params = True

# Maps EMBED_PROVIDER to the model prefix LiteLLM uses for routing. `openai` also
# covers any OpenAI-compatible server (a local MLX/LM Studio endpoint, text-embedding-3,
# etc.) via EMBED_BASE_URL.
_PROVIDER_PREFIX = {
    "ollama": "ollama",
    "openai": "openai",
}

# Task prefixes (document, query) keyed by base model name. Some embedding models are
# trained with task instructions prepended to the text; omitting them puts queries and
# documents in slightly different subspaces and measurably hurts retrieval. Models not
# listed here (bge-m3, mxbai-embed-large, ...) correctly take no prefix.
_TASK_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
}

DEFAULT_EMBED_DIM = 768


def embedding_dim() -> int:
    """Vector width for the configured model — used to build the LanceDB schema.

    Read explicitly from EMBED_DIM rather than auto-probed: the dimension is baked into
    the table at creation time, so it's a deliberate, validated choice. It must match
    EMBED_MODEL's real output; _embed() asserts this on every call. Changing the model
    (and thus the dimension) requires a full re-ingest into a fresh table.
    """
    return int(os.environ.get("EMBED_DIM", DEFAULT_EMBED_DIM))


def _resolve() -> tuple[str, dict]:
    """Build the LiteLLM model string and per-provider kwargs from the environment."""
    provider = os.environ.get("EMBED_PROVIDER", "ollama").lower()
    prefix = _PROVIDER_PREFIX.get(provider)
    if prefix is None:
        supported = ", ".join(sorted(_PROVIDER_PREFIX))
        raise ValueError(f"Unsupported EMBED_PROVIDER {provider!r}. Supported: {supported}.")

    model = f"{prefix}/{os.environ['EMBED_MODEL']}"

    kwargs: dict = {}
    if provider == "ollama":
        kwargs["api_base"] = os.environ["OLLAMA_HOST"]
    elif provider == "openai":
        base_url = os.environ.get("EMBED_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["api_base"] = base_url

    return model, kwargs


def _task_prefixes() -> tuple[str, str]:
    # Ollama tags look like "nomic-embed-text:latest"; key on the bare model name.
    name = os.environ["EMBED_MODEL"].split(":")[0]
    return _TASK_PREFIXES.get(name, ("", ""))


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages for storage, applying the model's document task prefix."""
    document_prefix = _task_prefixes()[0]
    return _embed([f"{document_prefix}{text}" for text in texts])


def embed_query(text: str) -> list[float]:
    """Embed a single query, applying the model's query task prefix."""
    query_prefix = _task_prefixes()[1]
    return _embed([f"{query_prefix}{text}"])[0]


# Retry only transient transport failures (connection refused, timeouts): a bad model
# name or malformed request raises a non-transient error and should fail fast instead.
_TRANSIENT = (litellm.exceptions.APIConnectionError, litellm.exceptions.Timeout)


@retry(
    retry=retry_if_exception_type(_TRANSIENT),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=30),
    reraise=True,
)
def _embed(inputs: list[str]) -> list[list[float]]:
    model, kwargs = _resolve()
    response = embedding(model=model, input=inputs, **kwargs)
    vectors = [item["embedding"] for item in response.data]

    expected = embedding_dim()
    for vector in vectors:
        if len(vector) != expected:
            raise ValueError(
                f"{model} returned {len(vector)}-dim vectors but EMBED_DIM={expected}. "
                "Set EMBED_DIM to the model's real dimension and re-ingest from scratch."
            )
    return vectors
