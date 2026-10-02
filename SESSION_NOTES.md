# Session notes: Voice to Order

_Last updated: 2 October 2026_

## Where things stand

- `main` has 13 merged PRs (#1–#13). Each one was squash-merged after CI passed.
- 126 tests pass. ruff and mypy (strict) are clean. CI runs lint, type checks and tests, and builds the Docker image and checks that it starts.
- Every item on the README roadmap is built. The README describes the working prototype.
- There's no uncommitted work. This file isn't committed on purpose.

## Done

| PR | What it adds |
|---|---|
| #1 | Project scaffold: `src/` layout, settings, domain models, CI, `docs/architecture.md` |
| #2 | Audio: ffmpeg/ffprobe wrapper; ingestion checks the file and converts it to 16 kHz mono |
| #3 | Deepgram and Whisper providers; parallel runner that keeps partial results |
| #4 | Claude client (structured outputs, refusal fallback); LLM consensus with a ROVER fallback |
| #5 | Order extraction with catalogue SKU matching and review flags |
| #6 | Delivery date parser and majority vote across transcripts |
| #7 | Evaluation: WER/CER, 8 synthetic labelled voicemails, `evaluate` command |
| #8 | AssemblyAI as a third provider, so voting has a real majority |
| #9 | Pipeline, background worker, FastAPI app, `process` command |
| #10 | Short provider errors (no request URLs); short catalogue keyterms for recognition |
| #11 | Phone-quality sample audio (`scripts/synthesize_samples.sh`), `record` command that saves **real** provider output, number-normalised WER, standard key names (`OPENAI_API_KEY` …) |
| #12 | Order-level accuracy (`evaluate --orders`); a missing Anthropic key now gives a clear error |
| #13 | README rewrite, Dockerfile with ffmpeg, Docker build and smoke test in CI |

## Results

**Real Whisper output** on the 8 text-to-speech phone-quality samples:

| Metric | Value |
|---|---:|
| WER (number-normalised) | 5.5% |
| WER before normalisation | 17.3% |
| Mean CER | 2.1% |
| Delivery date accuracy | 100% |

Real errors that are left: "Haber kitchen" (should be Harbour), "a count 5120" (account), "Marco ad Piazza".

**Hand-written transcripts** show how the voting works, not real accuracy. With three engines, ROVER matches the best single engine's WER (2.3%). The date vote fixes both single-engine date errors (100%, against 88% for Deepgram and Whisper on their own).

## What's left (needs keys I didn't have)

Only `OPENAI_API_KEY` was available. To finish the real-world numbers:

1. Add `ANTHROPIC_API_KEY`, `DEEPGRAM_API_KEY` and `ASSEMBLYAI_API_KEY` to `.env`.
2. `voice-to-order record`: adds Deepgram and AssemblyAI output to `samples/voicemails/*.json`. It's cheap, about a minute of audio per provider.
3. `voice-to-order evaluate --transcripts recorded --orders`: the first real run of Claude consensus and extraction. It makes 16 Claude calls.
4. Put the real numbers in the README evaluation section and commit the updated samples.

## Later ideas

- Evaluate on real voicemails (with consent, anonymised). Text-to-speech audio is optimistic.
- A persistent `JobStore` (Redis or Postgres) so that more than one API replica can run.
- Webhook or callback when a job finishes.
- Feed low-confidence words that the order depends on into the review flags.

## Decisions you may want to revisit

- **"next Monday"** means Monday of next week (UK usage). See `_resolve` in `extraction/delivery_date.py`.
- **Numeric dates** are read as day/month: `5/10` is 5 October.
- **`fallbacks: "default"`** uses a beta (`server-side-fallback-2026-07-01`) and is easy to remove in `llm/claude.py`.
- **LLM effort** defaults to `medium`. Change it with `VTO_LLM_EFFORT`.
- **WER normalisation** treats "four" and "4" as the same and maps `kg`/`g` to `kilo`/`gram`. See `evaluation/metrics.py`.
- **PRs were squash-merged without waiting for your review**, as you asked. You can still review each one on GitHub.
