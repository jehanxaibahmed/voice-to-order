with open("tests/unit/test_local_whisper.py", "r") as f:
    content = f.read()
import re
content = re.sub(r'from voice_to_order\.llm import ClaudeClient, OpenAICompatibleClient\n', 'from voice_to_order.llm import LiteLLMClient\n', content)
content = re.sub(r'def test_build_llm_selects_provider\(\) -> None:.*?# type: ignore\[call-arg\]', '', content, flags=re.DOTALL)
with open("tests/unit/test_local_whisper.py", "w") as f:
    f.write(content)
