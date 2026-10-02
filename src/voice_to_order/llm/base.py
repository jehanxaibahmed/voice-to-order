from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        """Return the model's answer parsed into ``schema``. Raise LLMError on failure."""
        ...
