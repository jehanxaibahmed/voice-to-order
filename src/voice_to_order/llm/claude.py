"""Claude implementation of LLMClient using structured outputs."""

import logging
from typing import Literal, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from voice_to_order.domain import LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Server-side fallback: if a safety classifier declines the request, the API re-runs it on
# Anthropic's recommended fallback model inside the same call.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeClient:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        max_tokens: int = 16_000,
    ) -> None:
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._max_tokens = max_tokens

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        try:
            response = await self._client.beta.messages.parse(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_format=schema,
                output_config={"effort": self._effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            raise LLMError("rate limited by the Claude API") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API returned {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("could not reach the Claude API") from exc
        except ValidationError as exc:
            raise LLMError(f"response did not match {schema.__name__}") from exc
        except TypeError as exc:
            # The SDK raises TypeError at request time when it finds no credentials.
            if "authentication" not in str(exc):
                raise
            raise LLMError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY (or VTO_ANTHROPIC_API_KEY)"
            ) from exc

        if response.stop_reason == "refusal":
            raise LLMError("the request was declined by the model")
        if response.stop_reason == "max_tokens":
            raise LLMError("response was cut off at max_tokens")
        if response.parsed_output is None:
            raise LLMError("response contained no structured output")
        return response.parsed_output
