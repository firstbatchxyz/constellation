"""Command-line helpers for converting and scoring JSONL traces."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from constellation.filtering import passes_basic_filters
from constellation.io import iter_jsonl, write_jsonl
from constellation.parsers import parse_agenttrove_row, parse_hermes_row
from constellation.schema import CanonicalSample
from constellation.scoring import with_quality_score

ParserFn = Callable[[dict[str, Any]], CanonicalSample]

PARSERS: dict[str, ParserFn] = {
    "agenttrove": parse_agenttrove_row,
    "hermes": parse_hermes_row,
}


def convert(args: argparse.Namespace) -> int:
    parser = PARSERS[args.parser]

    def rows() -> Any:
        for row in iter_jsonl(args.input):
            sample = with_quality_score(parser(row))
            yield sample.to_dict()

    count = write_jsonl(args.output, rows())
    print(f"wrote {count} canonical samples to {args.output}")
    return 0


def filter_rows(args: argparse.Namespace) -> int:
    def rows() -> Any:
        for row in iter_jsonl(args.input):
            sample = CanonicalSample.from_dict(row)
            if passes_basic_filters(
                sample,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                require_success=args.require_success,
            ):
                yield sample.to_dict()

    count = write_jsonl(args.output, rows())
    print(f"wrote {count} filtered samples to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="constellation")
    subcommands = root.add_subparsers(dest="command", required=True)

    convert_cmd = subcommands.add_parser("convert", help="convert raw JSONL to canonical JSONL")
    convert_cmd.add_argument("--parser", choices=sorted(PARSERS), required=True)
    convert_cmd.add_argument("--input", type=Path, required=True)
    convert_cmd.add_argument("--output", type=Path, required=True)
    convert_cmd.set_defaults(func=convert)

    filter_cmd = subcommands.add_parser("filter", help="filter canonical JSONL")
    filter_cmd.add_argument("--input", type=Path, required=True)
    filter_cmd.add_argument("--output", type=Path, required=True)
    filter_cmd.add_argument("--min-tokens", type=int, default=64)
    filter_cmd.add_argument("--max-tokens", type=int, default=32768)
    filter_cmd.add_argument("--require-success", action="store_true")
    filter_cmd.set_defaults(func=filter_rows)

    return root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
