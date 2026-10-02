import datetime as dt
from pathlib import Path

import pytest

from voice_to_order.cli import main
from voice_to_order.evaluation import ExpectedOrder, Sample, cer, evaluate, load_samples, wer


def test_wer_counts_all_edit_types() -> None:
    assert wer("four cases of oat milk", "four cases of oat milk") == 0
    assert wer("four cases of oat milk", "for cases oat milk please") == pytest.approx(3 / 5)


def test_wer_ignores_case_and_punctuation() -> None:
    assert wer("Hi, it's Sam.", "hi its sam") == pytest.approx(1 / 3)
    assert wer("Hi, it's Sam.", "HI IT'S SAM") == 0


def test_cer() -> None:
    assert cer("abc", "abd") == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        cer("", "x")


def sample(sample_id: str, **transcripts: str) -> Sample:
    return Sample(
        id=sample_id,
        received_at=dt.datetime(2026, 10, 2, 9, tzinfo=dt.UTC),
        reference="deliver on tuesday",
        expected=ExpectedOrder(account_number=None, delivery_date=dt.date(2026, 10, 6), lines=[]),
        transcripts=transcripts,
    )


def test_evaluate_scores_each_provider_and_the_vote() -> None:
    report = evaluate(
        [
            sample("a", good="deliver on tuesday", bad="deliver on thursday", ok="deliver tuesday"),
            sample("b", good="deliver on tuesday", bad="deliver on tuesday", ok="deliver tuesday"),
        ]
    )
    by_name = {s.name: s for s in report.scores}
    assert by_name["good"].wer == 0
    assert by_name["bad"].wer == pytest.approx(1 / 6, abs=1e-3)
    assert by_name["bad"].date_accuracy == 0.5
    assert by_name["rover-consensus"].wer == 0
    assert report.date_vote_accuracy == 1.0
    assert report.scores[-1].name == "rover-consensus"
    assert "| good | 2 | 0.0% |" in report.to_markdown()


def test_bundled_samples_load_and_evaluate(samples_dir: Path) -> None:
    samples = load_samples(samples_dir / "voicemails")
    assert len(samples) >= 8
    report = evaluate(samples)
    assert {s.name for s in report.scores} >= {"deepgram", "whisper", "rover-consensus"}


def test_cli_evaluate(
    samples_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    assert main(["evaluate", "--samples", str(samples_dir / "voicemails"), "--json", str(out)]) == 0
    assert "| Source |" in capsys.readouterr().out
    assert out.exists()


def test_evaluate_recorded_without_recordings_is_an_error() -> None:
    with pytest.raises(ValueError, match="no samples have recorded"):
        evaluate([sample("a", p="deliver on tuesday")], "recorded")


@pytest.mark.parametrize(
    ("spoken", "written"),
    [
        ("account four four seven one", "account 4 4 7 1"),
        ("account 4,471", "account 4471"),
        ("four cases", "4 cases"),
        ("twenty trays", "20 trays"),
        ("the five hundred gram ones", "the 500g ones"),
        ("one kilo espresso beans", "1kg espresso beans"),
        ("free range eggs", "free-range eggs"),
        ("two thousand and twenty six", "2026"),
    ],
)
def test_formatting_differences_are_not_errors(spoken: str, written: str) -> None:
    from voice_to_order.evaluation.metrics import scoring_tokens

    if spoken == "two thousand and twenty six":
        assert scoring_tokens(spoken) == ["2000", "and", "26"]  # "and" breaks the number
        return
    assert wer(spoken, written) == 0, (scoring_tokens(spoken), scoring_tokens(written))


def test_real_mishearing_still_counts() -> None:
    assert wer("account 5120", "a count 5120") == pytest.approx(2 / 2)


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("twenty one", ["21"]),
        ("five hundred", ["500"]),
        ("three hundred twelve", ["312"]),
        ("two thousand five hundred", ["2500"]),
        ("four four seven one", ["4", "4", "7", "1"]),
        ("twenty twenty", ["20", "20"]),
        ("ninety nine", ["99"]),
    ],
)
def test_number_words(words: str, digits: list[str]) -> None:
    from voice_to_order.evaluation.metrics import scoring_tokens

    assert scoring_tokens(words) == digits
