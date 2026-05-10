from __future__ import annotations

import os
import time
from typing import Any

import anthropic

# Load .env automatically when python-dotenv is present.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles on each retry (1 s, 2 s, 4 s)

_RETRIABLE_ERRORS = (anthropic.RateLimitError, anthropic.InternalServerError)


class LLMClient:
    """Thin wrapper around the Anthropic SDK.

    Responsibilities:
    - Loads ANTHROPIC_API_KEY from the environment (or .env via python-dotenv).
    - Applies exponential-backoff retry for rate-limit / server errors (max 3 tries).
    - Exposes two methods:
        complete()            → plain-text response (str)
        complete_structured() → parsed Pydantic model instance
    """

    def __init__(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example → .env and add your key, "
                "or export it as an environment variable."
            )
        self._client = anthropic.Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        max_tokens: int = 1024,
        model: str = _DEFAULT_MODEL,
    ) -> str:
        """Send a prompt and return the assistant's text response."""
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        return self._with_retries(
            lambda: self._client.messages.create(**kwargs).content[0].text
        )

    def chat(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        max_tokens: int = 1024,
        model: str = _DEFAULT_MODEL,
    ) -> str:
        """Multi-turn conversation. *messages* is a list of role/content dicts."""
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        return self._with_retries(
            lambda: self._client.messages.create(**kwargs).content[0].text
        )

    def complete_structured(
        self,
        prompt: str,
        output_format: type,
        *,
        system_prompt: str = "",
        max_tokens: int = 2048,
        model: str = _DEFAULT_MODEL,
    ):
        """Structured-output call. Returns the parsed Pydantic model instance."""
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_format=output_format,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        return self._with_retries(
            lambda: self._client.messages.parse(**kwargs).parsed_output
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _with_retries(self, fn):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except _RETRIABLE_ERRORS as exc:
                last_exc = exc
                time.sleep(_BASE_DELAY * (2 ** attempt))
            except anthropic.APIError:
                raise  # non-retriable (auth errors, bad requests, etc.)
        raise last_exc  # type: ignore[misc]
