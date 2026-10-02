"""Command-line entry point: ``voice-to-order <command>``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from voice_to_order.evaluation import evaluate, load_samples


def _evaluate(args: argparse.Namespace) -> int:
    report = evaluate(load_samples(args.samples))
    print(report.to_markdown())
    if args.json:
        args.json.write_text(report.model_dump_json(indent=2))
        print(f"\nDetailed results written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-to-order")
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_cmd = commands.add_parser("evaluate", help="compare provider accuracy on samples")
    evaluate_cmd.add_argument("--samples", type=Path, default=Path("samples/voicemails"))
    evaluate_cmd.add_argument("--json", type=Path, help="also write detailed results here")
    evaluate_cmd.set_defaults(handler=_evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
