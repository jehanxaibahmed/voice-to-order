"""A scripted LLMClient for tests and offline runs."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic import BaseModel

from voice_to_order.domain import LLMError

T = TypeVar("T", bound=BaseModel)

Responder = Callable[[str, str, type[BaseModel]], BaseModel | dict[str, object] | Exception]


@dataclass
class Call:
    system: str
    prompt: str
    schema: type[BaseModel]


@dataclass
class ScriptedLLM:
    """Answers each call with ``responder(system, prompt, schema)``.

    The responder may return a model instance, a dict to validate against the schema, or an
    exception to raise. Every call is recorded in ``calls``.
    """

    responder: Responder
    calls: list[Call] = field(default_factory=list)

    async def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        self.calls.append(Call(system, prompt, schema))
        answer = self.responder(system, prompt, schema)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, dict):
            return schema.model_validate(answer)
        if not isinstance(answer, schema):
            raise LLMError(f"scripted answer is {type(answer).__name__}, not {schema.__name__}")
        return answer
