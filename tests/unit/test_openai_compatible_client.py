import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from voice_to_order.domain import LLMError
from voice_to_order.llm import OpenAICompatibleClient


class Answer(BaseModel):
    text: str


def reply(content: str, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]},
    )


def make(handler: Any, **kwargs: Any) -> tuple[OpenAICompatibleClient, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def record(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://llm.test/v1/chat/completions"
        bodies.append(json.loads(request.content))
        bodies[-1]["_auth"] = request.headers.get("authorization")
        return handler(request)  # type: ignore[no-any-return]

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return OpenAICompatibleClient(
        client, base_url="http://llm.test/v1/", model="m", **kwargs
    ), bodies


async def test_sends_json_schema_response_format() -> None:
    llm, bodies = make(lambda _: reply('{"text": "hi"}'), api_key="k")
    assert await llm.structured(system="sys", prompt="p", schema=Answer) == Answer(text="hi")

    body = bodies[0]
    assert body["model"] == "m"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "p"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    assert body["_auth"] == "Bearer k"


async def test_no_auth_header_without_key() -> None:
    llm, bodies = make(lambda _: reply('{"text": "hi"}'))
    await llm.structured(system="s", prompt="p", schema=Answer)
    assert bodies[0]["_auth"] is None


async def test_strips_code_fences() -> None:
    llm, _ = make(lambda _: reply('```json\n{"text": "fenced"}\n```'))
    assert (await llm.structured(system="s", prompt="p", schema=Answer)).text == "fenced"


async def test_falls_back_to_prompt_injected_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "response_format" in json.loads(request.content):
            return httpx.Response(400, json={"error": "response_format unsupported"})
        return reply('```\n{"text": "ok"}\n```')

    llm, bodies = make(handler)
    assert (await llm.structured(system="sys", prompt="p", schema=Answer)).text == "ok"

    assert len(bodies) == 2
    assert "response_format" not in bodies[1]
    assert json.dumps(Answer.model_json_schema()) in bodies[1]["messages"][0]["content"]
    assert bodies[1]["messages"][0]["content"].startswith("sys")


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(500, text="boom"), "returned 500: boom"),
        (httpx.Response(200, json={"nope": 1}), "unexpected response shape"),
        (reply(""), "empty response"),
        (reply('{"text": "x"', "length"), "cut off"),
        (reply("not json"), "did not match Answer"),
        (reply('{"wrong": 1}'), "did not match Answer"),
    ],
)
async def test_errors_become_llm_errors(response: httpx.Response, match: str) -> None:
    llm, _ = make(lambda _: response)
    with pytest.raises(LLMError, match=match):
        await llm.structured(system="s", prompt="p", schema=Answer)


async def test_unreachable_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm, _ = make(handler)
    with pytest.raises(LLMError, match="could not reach"):
        await llm.structured(system="s", prompt="p", schema=Answer)
