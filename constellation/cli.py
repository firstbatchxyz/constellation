"""Command-line helpers for Constellation pilot data, training, and eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from constellation.config import artifact_path
from constellation.eval import run_eval_from_config
from constellation.filtering import passes_basic_filters
from constellation.io import iter_jsonl, write_jsonl
from constellation.parsers import parse_agenttrove_row, parse_hermes_row
from constellation.schema import CanonicalSample
from constellation.scoring import with_quality_score
from constellation.sft import train_sft_from_config
from constellation.streaming import DATASET_SOURCES, resolve_source, stream_convert
from constellation.subsets import build_debugging_pilot_subsets

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


def stream_convert_rows(args: argparse.Namespace) -> int:
    source = resolve_source(
        args.source,
        dataset_path=args.dataset_path,
        config_name=args.dataset_config,
        split=args.split,
        parser=args.parser,
    )
    stats = stream_convert(
        source=source,
        output=artifact_path(args.output),
        max_rows=args.max_rows,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        min_quality=args.min_quality,
        require_success=args.require_success,
        skip_errors=args.skip_errors,
        max_error_examples=args.max_error_examples,
    )
    print(json.dumps(stats, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    if args.hard_exit:
        os._exit(0)
    return 0


def build_subsets(args: argparse.Namespace) -> int:
    manifest = build_debugging_pilot_subsets(
        inputs=[artifact_path(path) for path in args.input],
        output_dir=artifact_path(args.output_dir),
        target_capability=args.target_capability,
        max_train_tokens=args.max_train_tokens,
        specialist_target_ratio=args.specialist_target_ratio,
        eval_fraction=args.eval_fraction,
        eval_max_samples=args.eval_max_samples,
        min_quality=args.min_quality,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def train_sft(args: argparse.Namespace) -> int:
    manifest = train_sft_from_config(args.config)
    print(json.dumps(manifest, indent=2))
    return 0


def eval_models(args: argparse.Namespace) -> int:
    summary = run_eval_from_config(args.config)
    print(json.dumps(summary, indent=2))
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

    stream_cmd = subcommands.add_parser(
        "stream-convert",
        help="stream a small remote dataset slice into canonical curated JSONL",
    )
    stream_cmd.add_argument("--source", choices=sorted(DATASET_SOURCES), default="agenttrove")
    stream_cmd.add_argument("--dataset-path", help="custom Hugging Face dataset path")
    stream_cmd.add_argument("--dataset-config", help="custom Hugging Face dataset config")
    stream_cmd.add_argument("--split", default=None)
    stream_cmd.add_argument("--parser", choices=sorted(PARSERS), default=None)
    stream_cmd.add_argument("--output", type=Path, required=True)
    stream_cmd.add_argument("--max-rows", type=int, default=10)
    stream_cmd.add_argument("--min-tokens", type=int, default=64)
    stream_cmd.add_argument("--max-tokens", type=int, default=32768)
    stream_cmd.add_argument("--min-quality", type=float, default=0.0)
    stream_cmd.add_argument("--require-success", action="store_true")
    stream_cmd.add_argument("--skip-errors", action="store_true")
    stream_cmd.add_argument("--max-error-examples", type=int, default=3)
    stream_cmd.add_argument(
        "--no-hard-exit",
        action="store_false",
        dest="hard_exit",
        help="return normally instead of forcing process exit after streaming completes",
    )
    stream_cmd.set_defaults(hard_exit=True)
    stream_cmd.set_defaults(func=stream_convert_rows)

    subsets_cmd = subcommands.add_parser(
        "build-subsets",
        help="build matched debugging specialist/control/eval subsets",
    )
    subsets_cmd.add_argument("--input", type=Path, nargs="+", required=True)
    subsets_cmd.add_argument("--output-dir", type=Path, required=True)
    subsets_cmd.add_argument("--target-capability", default="DEBUGGING")
    subsets_cmd.add_argument("--max-train-tokens", type=int, default=2_000_000)
    subsets_cmd.add_argument("--specialist-target-ratio", type=float, default=0.8)
    subsets_cmd.add_argument("--eval-fraction", type=float, default=0.1)
    subsets_cmd.add_argument("--eval-max-samples", type=int, default=200)
    subsets_cmd.add_argument("--min-quality", type=float, default=0.45)
    subsets_cmd.add_argument("--min-tokens", type=int, default=64)
    subsets_cmd.add_argument("--max-tokens", type=int, default=32768)
    subsets_cmd.add_argument("--seed", default="constellation-v1")
    subsets_cmd.set_defaults(func=build_subsets)

    train_cmd = subcommands.add_parser("train-sft", help="launch masked full-SFT from JSON config")
    train_cmd.add_argument("--config", type=Path, required=True)
    train_cmd.set_defaults(func=train_sft)

    eval_cmd = subcommands.add_parser("eval", help="run debugging generation eval from JSON config")
    eval_cmd.add_argument("--config", type=Path, required=True)
    eval_cmd.set_defaults(func=eval_models)

    return root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
