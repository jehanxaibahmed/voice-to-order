from voice_to_order.consensus import LLMReconciler, RoverReconciler
from voice_to_order.consensus.llm import build_prompt
from voice_to_order.domain import LLMError, Transcript, TranscriptionResult
from voice_to_order.llm import ScriptedLLM


def result(*texts: str) -> TranscriptionResult:
    return TranscriptionResult(
        transcripts=[Transcript(provider=f"p{i}", text=t) for i, t in enumerate(texts)]
    )


async def test_rover_majority_fixes_single_engine_errors() -> None:
    consensus = await RoverReconciler().reconcile(
        result(
            "please send four cases of oat milk",
            "please send for cases of oat milk",
            "please send four cases of goat milk",
        )
    )
    assert consensus.text == "please send four cases of oat milk"
    assert consensus.method == "rover"
    assert consensus.uncertain_spans == []


async def test_rover_handles_dropped_and_inserted_words() -> None:
    consensus = await RoverReconciler().reconcile(
        result(
            "hi this is sam from the deli",
            "this is sam from the deli",
            "hi this is sam um from the deli",
        )
    )
    assert consensus.text == "hi this is sam from the deli"


async def test_rover_flags_slots_without_majority() -> None:
    consensus = await RoverReconciler().reconcile(
        result(
            "deliver on tuesday", "deliver on thursday", "deliver on tuesday", "deliver on today"
        )
    )
    assert consensus.text == "deliver on tuesday"
    assert consensus.uncertain_spans == ["thursday/today/tuesday"]


async def test_rover_single_transcript_passthrough() -> None:
    consensus = await RoverReconciler().reconcile(result("Just One."))
    assert (consensus.text, consensus.method) == ("just one", "single")


async def test_llm_reconciler_uses_llm_answer() -> None:
    llm = ScriptedLLM(lambda *_: {"text": " four cases of OM-12 ", "uncertain_spans": ["OM-12"]})
    consensus = await LLMReconciler(llm).reconcile(
        result("for cases of om twelve", "four cases of OM-12"), vocabulary=["OM-12 Oat milk"]
    )
    assert consensus.method == "llm"
    assert consensus.text == "four cases of OM-12"
    assert consensus.uncertain_spans == ["OM-12"]
    assert consensus.sources == ["p0", "p1"]
    prompt = llm.calls[0].prompt
    assert '<transcript engine="p0">' in prompt
    assert "OM-12 Oat milk" in prompt


async def test_llm_reconciler_falls_back_to_rover() -> None:
    llm = ScriptedLLM(lambda *_: LLMError("down"))
    consensus = await LLMReconciler(llm).reconcile(result("a b c", "a b c", "a x c"))
    assert consensus.method == "rover"
    assert consensus.text == "a b c"


async def test_llm_not_called_for_single_transcript() -> None:
    llm = ScriptedLLM(lambda *_: AssertionError("should not be called"))
    consensus = await LLMReconciler(llm).reconcile(result("Only One."))
    assert consensus.text == "Only One."
    assert llm.calls == []


def test_prompt_includes_confidence_when_known() -> None:
    res = TranscriptionResult(transcripts=[Transcript(provider="dg", text="x", confidence=0.912)])
    assert 'confidence="0.91"' in build_prompt(res, [])
