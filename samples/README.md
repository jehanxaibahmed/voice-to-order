# Sample data

Everything here is synthetic. The businesses, people, account numbers and phone numbers are
invented, and the phone numbers use Ofcom's reserved drama range (`07700 900xxx`).

- `catalog.json`: the product catalogue used for SKU matching and recognition hints.
- `audio/*.wav`: each reference transcript read by macOS text-to-speech in several accents,
  then narrowed to the telephone band (300–3400 Hz), resampled to 8 kHz and mixed with light
  background noise. Regenerate with `scripts/synthesize_samples.sh` (macOS only).
- `voicemails/*.json`: one file per voicemail:
  - `reference`: the ground-truth transcript
  - `expected`: the order a human would key in (account, delivery date, `[sku, quantity]` lines)
  - `transcripts`: **hand-written** per-provider transcripts with typical engine errors. They
    exercise consensus and voting offline, but they are not real provider output.
  - `recorded_transcripts`: **real** provider output for `audio/<id>.wav`, saved by
    `voice-to-order record`. Only providers that had an API key at recording time appear here.

Compare them with `voice-to-order evaluate --transcripts synthetic|recorded`.

Text-to-speech audio is cleaner than a real voicemail (one clear voice, scripted pacing, no
crosstalk), so recorded results are an optimistic baseline. Use real voicemails to measure
real-world accuracy.
