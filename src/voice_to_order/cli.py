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
    report = evaluate(load_samples(args.samples))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-to-order")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_cmd = commands.add_parser("evaluate", help="compare provider accuracy on samples")
    evaluate_cmd.add_argument("--samples", type=Path, default=Path("samples/voicemails"))
    evaluate_cmd.add_argument("--json", type=Path, help="also write detailed results here")
    evaluate_cmd.set_defaults(handler=_evaluate)

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
