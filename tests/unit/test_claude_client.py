import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from voice_to_order.domain import LLMError
from voice_to_order.llm import ClaudeClient


class Answer(BaseModel):
    text: str


def message(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}] if text else [],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def client_for(handler: Any) -> tuple[ClaudeClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)  # type: ignore[no-any-return]

    sdk = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(record)),
    )
    return ClaudeClient(sdk), seen


async def test_parses_structured_output_and_sends_fallback() -> None:
    client, seen = client_for(lambda _: httpx2.Response(200, json=message('{"text": "hi"}')))
    answer = await client.structured(system="sys", prompt="p", schema=Answer)

    assert answer == Answer(text="hi")
    body = json.loads(seen[0].content)
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "server-side-fallback-2026-07-01" in seen[0].headers["anthropic-beta"]


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx2.Response(200, json=message("", "refusal")), "declined"),
        (httpx2.Response(200, json=message('{"text": "hi', "max_tokens")), "cut off|match"),
        (httpx2.Response(200, json=message('{"wrong": 1}')), "did not match"),
        (
            httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error"}}),
            "rate",
        ),
        (httpx2.Response(500, json={"type": "error", "error": {"type": "api_error"}}), "500"),
    ],
)
async def test_failures_become_llm_error(response: httpx2.Response, match: str) -> None:
    client, _ = client_for(lambda _: response)
    with pytest.raises(LLMError, match=match):
        await client.structured(system="s", prompt="p", schema=Answer)


async def test_missing_credentials_is_a_clear_llm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An empty home directory means no `ant auth login` profile on disk either.
    for var in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    sdk = anthropic.AsyncAnthropic(
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(500))
        ),
    )
    with pytest.raises(LLMError, match="no Anthropic credentials"):
        await ClaudeClient(sdk).structured(system="s", prompt="p", schema=Answer)
