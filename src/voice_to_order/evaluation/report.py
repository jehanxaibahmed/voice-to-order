"""Compare providers, ROVER consensus and the delivery date vote against ground truth."""

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel

from voice_to_order.consensus import RoverReconciler
from voice_to_order.evaluation.dataset import Sample, TranscriptSource
from voice_to_order.evaluation.metrics import cer, corpus_errors
from voice_to_order.extraction import parse_delivery_date, vote_delivery_date

ROVER = "rover-consensus"
VOTE = "date-vote"


class ProviderScore(BaseModel):
    name: str
    samples: int
    wer: float
    mean_cer: float
    date_accuracy: float


class SampleDetail(BaseModel):
    id: str
    wer: dict[str, float]
    dates_correct: dict[str, bool]


class EvaluationReport(BaseModel):
    scores: list[ProviderScore]
    date_vote_accuracy: float
    details: list[SampleDetail]

    def to_markdown(self) -> str:
        lines = [
            "| Source | Samples | WER | Mean CER | Delivery date accuracy |",
            "|---|---:|---:|---:|---:|",
        ]
        for s in self.scores:
            lines.append(
                f"| {s.name} | {s.samples} | {s.wer:.1%} | {s.mean_cer:.1%} "
                f"| {s.date_accuracy:.0%} |"
            )
        lines.append(
            f"| **{VOTE}** (majority) | {len(self.details)} | n/a | n/a "
            f"| **{self.date_vote_accuracy:.0%}** |"
        )
        return "\n".join(lines)


def evaluate(samples: Sequence[Sample], source: TranscriptSource = "synthetic") -> EvaluationReport:
    samples = [s for s in samples if s.transcripts_for(source)]
    if not samples:
        raise ValueError(f"no samples have {source} transcripts")
    errors: dict[str, int] = defaultdict(int)
    words: dict[str, int] = defaultdict(int)
    cers: dict[str, list[float]] = defaultdict(list)
    dates: dict[str, list[bool]] = defaultdict(list)
    vote_hits: list[bool] = []
    details: list[SampleDetail] = []

    for sample in samples:
        reference_day = sample.received_at.date()
        expected_date = sample.expected.delivery_date
        result = sample.transcription_result(source)
        rover = RoverReconciler.vote(result.transcripts)
        hypotheses = dict(sample.transcripts_for(source))
        if len(hypotheses) > 1:
            hypotheses[ROVER] = rover.text

        detail = SampleDetail(id=sample.id, wer={}, dates_correct={})
        for name, text in hypotheses.items():
            err, n = corpus_errors(sample.reference, text)
            errors[name] += err
            words[name] += n
            cers[name].append(cer(sample.reference, text))
            correct = parse_delivery_date(text, reference_day) == expected_date
            dates[name].append(correct)
            detail.wer[name] = round(err / n, 4)
            detail.dates_correct[name] = correct

        decision = vote_delivery_date(result, rover, reference_day)
        vote_hits.append(decision.date == expected_date)
        detail.dates_correct[VOTE] = vote_hits[-1]
        details.append(detail)

    scores = [
        ProviderScore(
            name=name,
            samples=len(cers[name]),
            wer=round(errors[name] / words[name], 4),
            mean_cer=round(sum(cers[name]) / len(cers[name]), 4),
            date_accuracy=round(sum(dates[name]) / len(dates[name]), 4),
        )
        for name in errors
    ]
    scores.sort(key=lambda s: (s.name == ROVER, s.wer))
    return EvaluationReport(
        scores=scores,
        date_vote_accuracy=round(sum(vote_hits) / len(vote_hits), 4),
        details=details,
    )
