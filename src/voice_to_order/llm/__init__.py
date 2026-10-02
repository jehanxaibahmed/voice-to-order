"""LLM access behind a small protocol, so pipeline stages never import a vendor SDK."""

from voice_to_order.llm.base import LLMClient
from voice_to_order.llm.claude import ClaudeClient
from voice_to_order.llm.fake import ScriptedLLM
from voice_to_order.llm.openai_compatible import OpenAICompatibleClient

__all__ = ["ClaudeClient", "LLMClient", "OpenAICompatibleClient", "ScriptedLLM"]
