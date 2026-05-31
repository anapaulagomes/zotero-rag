import logging
import os
from collections.abc import AsyncIterator

# Quiet LiteLLM before importing it: its import-time pre-load warnings about
# optional AWS backends (bedrock/sagemaker, which we don't use) fire during the
# import itself, so the logger level must be set first.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

import litellm  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from litellm import acompletion  # noqa: E402

load_dotenv()

litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.drop_params = True

# Maps the LLM_PROVIDER env var to the model prefix LiteLLM uses for routing.
# `ollama_chat` hits Ollama's /api/chat endpoint (better for instruct models than
# the /api/generate one behind the plain `ollama` prefix).
_PROVIDER_PREFIX = {
    "ollama": "ollama_chat",
    "claude": "anthropic",
    "openai": "openai",
}


def _resolve() -> tuple[str, dict]:
    """Build the LiteLLM model string and per-provider kwargs from the environment."""
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    prefix = _PROVIDER_PREFIX.get(provider)
    if prefix is None:
        supported = ", ".join(sorted(_PROVIDER_PREFIX))
        raise ValueError(
            f"Unsupported LLM_PROVIDER {provider!r}. Supported: {supported}."
        )

    model = f"{prefix}/{os.environ['LLM_MODEL']}"

    kwargs: dict = {}
    if provider == "ollama":
        kwargs["api_base"] = os.environ["OLLAMA_HOST"]
    elif provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["api_base"] = base_url

    return model, kwargs


async def stream_chat(system: str, user: str) -> AsyncIterator[str]:
    """Stream the LLM response token by token for the configured provider.

    The provider (Ollama, Claude or any OpenAI-compatible endpoint) is selected
    via LLM_PROVIDER / LLM_MODEL; API keys are read from the environment by
    LiteLLM. Raises litellm.exceptions.* (e.g. APIConnectionError,
    AuthenticationError) on the first iteration if the call fails.
    """
    model, kwargs = _resolve()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    response = await acompletion(
        model=model,
        messages=messages,
        stream=True,
        **kwargs,
    )

    async for chunk in response:
        if not chunk.choices:
            continue
        token = chunk.choices[0].delta.content or ""
        if token:
            yield token
