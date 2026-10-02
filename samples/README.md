# Sample data

Everything here is synthetic. The businesses, people, account numbers and phone numbers are
invented, and the phone numbers use Ofcom's reserved drama range (`07700 900xxx`).

- `catalog.json` — product catalogue used for SKU matching and vocabulary hints.
- `voicemails/*.json` — one file per voicemail:
  - `reference` — the ground-truth transcript
  - `expected` — the order a human would key in (account, delivery date, `[sku, quantity]` lines)
  - `transcripts` — per-provider transcripts with typical engine errors (misheard names,
    digits split apart, "Tuesday"/"Thursday", "thirteenth"/"thirtieth"). These are written to
    exercise the pipeline offline. They are hand-written, not recorded from the live APIs,
    so the evaluation numbers they produce demonstrate the harness, not real provider accuracy.
