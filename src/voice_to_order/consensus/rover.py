"""ROVER-style word voting (Fiscus, 1997), used offline and as the LLM fallback.

Pick the medoid transcript as the pivot, align every other transcript to it, and take the
majority word at each pivot position. Positions without a strict majority are reported as
uncertain.
"""

from collections import Counter
from collections.abc import Sequence

from voice_to_order.domain import ConsensusTranscript, Transcript, TranscriptionResult
from voice_to_order.text import Op, align, normalise_tokens, word_edit_distance


class RoverReconciler:
    async def reconcile(
        self, result: TranscriptionResult, vocabulary: Sequence[str] = ()
    ) -> ConsensusTranscript:
        return self.vote(result.transcripts)

    @staticmethod
    def vote(transcripts: Sequence[Transcript]) -> ConsensusTranscript:
        sources = [t.provider for t in transcripts]
        tokens = [normalise_tokens(t.text) for t in transcripts]
        if len(transcripts) == 1:
            return ConsensusTranscript(text=" ".join(tokens[0]), method="single", sources=sources)

        pivot = min(
            range(len(tokens)),
            key=lambda i: sum(word_edit_distance(tokens[i], other) for other in tokens),
        )
        pivot_tokens = tokens[pivot]
        # slot_votes[k] holds every transcript's word for pivot position k (None = omitted).
        slot_votes: list[list[str | None]] = [[word] for word in pivot_tokens]
        # insertions[k] holds words other transcripts placed before pivot position k.
        insertions: list[Counter[str]] = [Counter() for _ in range(len(pivot_tokens) + 1)]

        for index, other in enumerate(tokens):
            if index == pivot:
                continue
            position = 0
            for pair in align(pivot_tokens, other):
                if pair.op is Op.INSERT:
                    assert pair.hyp is not None
                    insertions[position][pair.hyp] += 1
                    continue
                slot_votes[position].append(pair.hyp)
                position += 1

        total = len(transcripts)
        words: list[str] = []
        uncertain: list[str] = []
        for position in range(len(pivot_tokens) + 1):
            for word, count in insertions[position].items():
                if count > total / 2:
                    words.append(word)
            if position == len(pivot_tokens):
                break
            ranked = Counter(slot_votes[position]).most_common()
            best, best_count = ranked[0]
            if len(ranked) > 1 and ranked[1][1] == best_count:
                best = pivot_tokens[position]  # tie: trust the medoid
            if best_count <= total / 2:
                options = sorted({w or "<none>" for w in slot_votes[position]})
                uncertain.append("/".join(options))
            if best is not None:
                words.append(best)

        return ConsensusTranscript(
            text=" ".join(words), method="rover", sources=sources, uncertain_spans=uncertain
        )
