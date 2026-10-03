with open("src/voice_to_order/llm/litellm_client.py", "r") as f:
    content = f.read()
content = content.replace("from voice_to_order.errors import LLMError", "from voice_to_order.domain import LLMError")
with open("src/voice_to_order/llm/litellm_client.py", "w") as f:
    f.write(content)
