# llms/factory.py

import os
from pathlib import Path
from typing import Any, cast
from .base import LLMClient
from .openai_client import OpenAIChatClient


def _dotenv_candidate_paths() -> tuple[Path, ...]:
    # factory.py lives at Democritus_OpenAI/llms/factory.py, so:
    #   parents[1] = Democritus_OpenAI/   (local override, if the user keeps one here)
    #   parents[2] = workspace root containing FunctorFlow/ as a sibling repo
    here = Path(__file__).resolve()
    democritus_root = here.parents[1]
    workspace_root = here.parents[2]
    return (
        democritus_root / ".env",
        workspace_root / "FunctorFlow" / ".env",
    )


def _load_dotenv_if_present() -> None:
    # Populate os.environ from the first readable .env found in the candidate
    # list. Mirrors FunctorFlow/democritus.py::_load_dotenv_if_present — shell
    # env wins (we never overwrite a key the user already set). This lets
    # Democritus_OpenAI pick up DEMOC_LLM_PROVIDER and ANTHROPIC_API_KEY from
    # the single FunctorFlow/.env without the user having to export anything.
    for env_path in _dotenv_candidate_paths():
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def make_llm_client(**kwargs: Any) -> LLMClient:
    """
    Construct the default LLM backend for Democritus v1.5.

    kwargs are forwarded to the selected client, e.g.:

        make_llm_client(max_tokens=128, max_batch_size=16)

    Provider selection:
      DEMOC_LLM_PROVIDER    default "openai"; set to "anthropic" to route
                            through llms.anthropic_client.AnthropicChatClient.

    Env-based defaults (OpenAI path):
      OPENAI_API_KEY
      DEMOC_LLM_BASE_URL
      DEMOC_LLM_MODEL
      DEMOC_LLM_MAX_TOKENS
      DEMOC_LLM_TEMPERATURE
      DEMOC_LLM_BATCH_SIZE

    Env-based defaults (Anthropic path):
      ANTHROPIC_API_KEY
      ANTHROPIC_BASE_URL
      ANTHROPIC_MODEL
      ANTHROPIC_API_VERSION
      DEMOC_LLM_MAX_TOKENS, DEMOC_LLM_TEMPERATURE, DEMOC_LLM_BATCH_SIZE are shared.

    Before dispatching, this function loads a .env file if one exists at
    Democritus_OpenAI/.env or at the sibling FunctorFlow/.env, mirroring
    FunctorFlow's loader. Shell-exported values always take precedence.
    """
    _load_dotenv_if_present()
    provider = os.getenv("DEMOC_LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return cast(LLMClient, OpenAIChatClient(**kwargs))
    if provider == "anthropic":
        # Imported lazily so OpenAI-only users pay nothing for the Anthropic path.
        from .anthropic_client import AnthropicChatClient
        return cast(LLMClient, AnthropicChatClient(**kwargs))
    raise RuntimeError(
        f"Unknown DEMOC_LLM_PROVIDER={provider!r}. Supported values: 'openai', 'anthropic'."
    )
