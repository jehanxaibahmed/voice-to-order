from typing import Any, TypeVar

import litellm
from pydantic import BaseModel

from voice_to_order.domain import LLMError
from voice_to_order.llm.base import LLMClient

T = TypeVar("T", bound=BaseModel)


class LiteLLMClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        kwargs: dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        kwargs["response_format"] = schema

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await litellm.acompletion(
                model=self.model, messages=messages, timeout=self.timeout, **kwargs
            )

            content = response.choices[0].message.content
            if not content:
                raise LLMError("Empty response from model")

            return schema.model_validate_json(content)
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
