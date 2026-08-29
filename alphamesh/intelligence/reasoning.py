"""LLM provider abstraction for the AI reasoning council.

The council is advisory. Providers return *text*, which the agents parse into
strictly validated Pydantic models; anything that fails validation is discarded
and the deterministic fallback is used instead. No provider output ever reaches
contract selection, sizing, risk approval or order construction.

Tests never require a paid LLM call: ``ScriptedProvider`` and ``NullProvider``
cover the success, malformed and unavailable paths offline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

import httpx

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMUnavailableError(RuntimeError):
    """The provider could not produce a response at all."""


class MalformedAIOutputError(ValueError):
    """The provider responded, but not with usable structured output."""


@runtime_checkable
class ReasoningProvider(Protocol):
    """Minimal contract every reasoning backend satisfies."""

    name: str

    def available(self) -> bool:
        """True when this provider can be called right now."""
        ...

    def complete_json(self, *, system: str, user: str, max_tokens: int = 900) -> dict[str, Any]:
        """Return a parsed JSON object, or raise."""
        ...


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models sometimes wrap JSON in prose or fences. We take the outermost
    brace-delimited span and parse it; anything else is malformed.
    """
    if not text or not text.strip():
        raise MalformedAIOutputError("empty response")
    match = _JSON_BLOCK.search(text)
    if not match:
        raise MalformedAIOutputError("no JSON object found in response")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise MalformedAIOutputError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise MalformedAIOutputError("top-level JSON value is not an object")
    return parsed


class NullProvider:
    """No LLM configured. Every call fails, forcing the heuristic fallback."""

    name = "null"

    def available(self) -> bool:
        return False

    def complete_json(self, *, system: str, user: str, max_tokens: int = 900) -> dict[str, Any]:
        raise LLMUnavailableError("no reasoning provider is configured")


class ScriptedProvider:
    """Deterministic provider for tests and offline dry runs.

    Responses are consumed in order. When exhausted, it raises
    ``LLMUnavailableError`` so the fallback path can be exercised too.
    """

    name = "scripted"

    def __init__(self, responses: list[dict[str, Any] | str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return True

    def complete_json(self, *, system: str, user: str, max_tokens: int = 900) -> dict[str, Any]:
        self.calls.append((system, user))
        if not self._responses:
            raise LLMUnavailableError("scripted provider exhausted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, str):
            return extract_json(item)
        return item


class AnthropicProvider:
    """Anthropic Messages API backend.

    The API key is read once at construction and never logged. Network and
    protocol failures are normalised into ``LLMUnavailableError`` so callers
    have exactly one failure mode to handle.
    """

    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-5",
        timeout: float = 25.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self._timeout = timeout
        self._client = client

    def available(self) -> bool:
        return bool(self._api_key)

    def complete_json(self, *, system: str, user: str, max_tokens: int = 900) -> dict[str, Any]:
        if not self.available():
            raise LLMUnavailableError("ANTHROPIC_API_KEY is not set")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        try:
            if self._client is not None:
                response = self._client.post(ANTHROPIC_URL, json=payload, headers=headers)
            else:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(ANTHROPIC_URL, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Anthropic request failed: {type(exc).__name__}") from exc
        except ValueError as exc:
            raise LLMUnavailableError("Anthropic returned a non-JSON body") from exc

        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        return extract_json(text)


def build_provider(api_key: str, model: str) -> ReasoningProvider:
    """Pick a provider from configuration. Absent a key, the council runs
    entirely on its deterministic heuristics rather than failing."""
    if api_key:
        return AnthropicProvider(api_key=api_key, model=model)
    return NullProvider()


__all__ = [
    "ANTHROPIC_URL",
    "AnthropicProvider",
    "LLMUnavailableError",
    "MalformedAIOutputError",
    "NullProvider",
    "ReasoningProvider",
    "ScriptedProvider",
    "build_provider",
    "extract_json",
]
