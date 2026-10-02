"""LLMClient for any OpenAI-compatible /chat/completions server (Ollama, vLLM, LM Studio...)."""

import json
import logging
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from voice_to_order.domain import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    match = _FENCE.match(text)
    return (match.group(1) if match else text).strip()


class OpenAICompatibleClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 4_096,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        json_schema = schema.model_json_schema()
        native = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": json_schema, "strict": True},
        }
        try:
            content = await self._complete(system, prompt, native)
        except _FormatRejected:
            logger.info("server rejected response_format; falling back to prompt-injected schema")
            injected = (
                f"{system}\n\nRespond with only a JSON object (no prose, no code fences) that "
                f"conforms to this JSON Schema:\n{json.dumps(json_schema)}"
            )
            content = await self._complete(injected, prompt, None)

        try:
            return schema.model_validate_json(strip_code_fence(content))
        except ValidationError as exc:
            raise LLMError(
                f"response did not match {schema.__name__}: {exc.error_count()} validation errors"
            ) from exc

    async def _complete(
        self, system: str, prompt: str, response_format: dict[str, Any] | None
    ) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            response = await self._client.post(self._url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"could not reach the LLM server at {self._url}: {type(exc).__name__}"
            ) from exc

        if response.status_code in (400, 422) and response_format is not None:
            raise _FormatRejected
        if response.status_code >= 400:
            raise LLMError(
                f"LLM server returned {response.status_code}: {response.text[:200].strip()}"
            )
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("LLM server returned an unexpected response shape") from exc
        if choice.get("finish_reason") == "length":
            raise LLMError("response was cut off at max_tokens")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM server returned an empty response")
        return content


class _FormatRejected(Exception):
    """The server does not support response_format with a JSON schema."""
