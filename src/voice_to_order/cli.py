"""Command-line entry point: ``voice-to-order <command>``."""

import argparse
import asyncio
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from voice_to_order.config import get_settings
from voice_to_order.domain import Voicemail
from voice_to_order.evaluation import evaluate, load_samples


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate(load_samples(args.samples), args.transcripts)
    print(report.to_markdown())
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2))
        print(f"\nDetailed results written to {args.json}")
    return 0


async def _process_async(args: argparse.Namespace) -> int:
    from voice_to_order.bootstrap import build_pipeline

    settings = get_settings()
    async with httpx.AsyncClient(timeout=settings.transcription_timeout_seconds) as client:
        pipeline = build_pipeline(settings, client, catalog_path=args.catalog)
        voicemail = Voicemail(
            id=args.audio.stem,
            received_at=dt.datetime.now(dt.UTC),
            caller_number=args.caller,
            source_path=args.audio,
        )
        result = await pipeline.process(voicemail)
    print(result.model_dump_json(indent=2))
    return 0


def _process(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_process_async(args))


async def _record_async(args: argparse.Namespace) -> int:
    from voice_to_order.evaluation.recorder import record_transcripts
    from voice_to_order.extraction import Catalog
    from voice_to_order.transcription import TranscriptionRunner, build_providers

    settings = get_settings()
    catalog = Catalog.load(args.catalog)
    async with httpx.AsyncClient(timeout=settings.transcription_timeout_seconds) as client:
        providers = build_providers(settings, client, vocabulary=catalog.keyterms())
        if not providers:
            print("No transcription provider has an API key; nothing to record.", file=sys.stderr)
            return 1
        print(f"Recording with: {', '.join(p.name for p in providers)}")
        outcomes = await record_transcripts(
            load_samples(args.samples),
            audio_dir=args.audio,
            samples_dir=args.samples,
            runner=TranscriptionRunner(
                providers, timeout_seconds=settings.transcription_timeout_seconds
            ),
        )
    for outcome in outcomes:
        failed = "".join(f", {name} failed: {err}" for name, err in outcome.failed.items())
        print(f"{outcome.sample_id}: recorded {', '.join(outcome.recorded) or 'nothing'}{failed}")
    return 0 if outcomes and all(o.recorded for o in outcomes) else 1


def _record(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_record_async(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-to-order")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_cmd = commands.add_parser("evaluate", help="compare provider accuracy on samples")
    evaluate_cmd.add_argument("--samples", type=Path, default=Path("samples/voicemails"))
    evaluate_cmd.add_argument("--json", type=Path, help="also write detailed results here")
    evaluate_cmd.add_argument(
        "--transcripts",
        choices=["synthetic", "recorded"],
        default="synthetic",
        help="hand-written transcripts, or real provider output saved by `record`",
    )
    evaluate_cmd.set_defaults(handler=_evaluate)

    record_cmd = commands.add_parser(
        "record", help="transcribe sample audio with the configured providers and save it"
    )
    record_cmd.add_argument("--samples", type=Path, default=Path("samples/voicemails"))
    record_cmd.add_argument("--audio", type=Path, default=Path("samples/audio"))
    record_cmd.add_argument("--catalog", type=Path, default=Path("samples/catalog.json"))
    record_cmd.set_defaults(handler=_record)

    process_cmd = commands.add_parser("process", help="turn one voicemail file into an order")
    process_cmd.add_argument("audio", type=Path)
    process_cmd.add_argument("--caller", help="caller id, used if the caller gives no number")
    process_cmd.add_argument("--catalog", type=Path, default=Path("samples/catalog.json"))
    process_cmd.set_defaults(handler=_process)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
