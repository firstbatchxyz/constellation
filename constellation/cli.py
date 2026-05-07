"""Command-line helpers for Constellation pilot data, training, and eval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from constellation.categorization import (
    export_classifier_jsonl,
    export_labeling_prompts_jsonl,
    relabel_jsonl,
    write_taxonomy_markdown,
)
from constellation.config import artifact_path
from constellation.eval import run_eval_from_config
from constellation.filtering import passes_basic_filters
from constellation.io import iter_jsonl, write_jsonl
from constellation.llm_labeling import DEFAULT_LLM_LABEL_MODEL, llm_label_jsonl
from constellation.model_labeling import DEFAULT_ZERO_SHOT_MODEL, model_label_jsonl
from constellation.parsers import parse_agenttrove_row, parse_hermes_row
from constellation.reporting import label_report, write_label_report
from constellation.schema import CanonicalSample
from constellation.scoring import with_quality_score
from constellation.sft import train_sft_from_config
from constellation.specialists import (
    DEFAULT_SPECIALIST_TARGETS_PATH,
    load_specialist_targets,
    targets_to_dicts,
)
from constellation.streaming import DATASET_SOURCES, resolve_source, stream_convert
from constellation.subsets import build_debugging_pilot_subsets

ParserFn = Callable[[dict[str, Any]], CanonicalSample]

PARSERS: dict[str, ParserFn] = {
    "agenttrove": parse_agenttrove_row,
    "hermes": parse_hermes_row,
}

DEFAULT_CAPABILITY_TAXONOMY = Path("configs/capability_taxonomy.json")
DEFAULT_DOMAIN_TAXONOMY = Path("configs/domain_taxonomy.json")


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
    target_capabilities = list(args.target_capability or ["DEBUGGING"])
    target_domains = list(args.target_domain or [])
    output_prefix = args.output_prefix
    if args.target_id:
        targets = {target.id: target for target in load_specialist_targets(args.specialist_targets)}
        if args.target_id not in targets:
            raise ValueError(f"unknown specialist target id: {args.target_id}")
        target = targets[args.target_id]
        target_capabilities = list(target.target_capabilities)
        target_domains = list(target.target_domains)
        output_prefix = output_prefix or target.id

    manifest = build_debugging_pilot_subsets(
        inputs=[artifact_path(path) for path in args.input],
        output_dir=artifact_path(args.output_dir),
        target_capability=target_capabilities[0] if len(target_capabilities) == 1 else "",
        target_capabilities=target_capabilities,
        target_domains=target_domains,
        output_prefix=output_prefix,
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


def relabel_capabilities(args: argparse.Namespace) -> int:
    summary = relabel_jsonl(
        input_path=artifact_path(args.input),
        output_path=artifact_path(args.output),
        taxonomy_path=args.taxonomy,
        domain_taxonomy_path=args.domain_taxonomy,
        min_score=args.min_score,
        max_chars=args.max_chars,
    )
    print(json.dumps(summary, indent=2))
    return 0


def model_label(args: argparse.Namespace) -> int:
    summary = model_label_jsonl(
        input_path=artifact_path(args.input),
        output_path=artifact_path(args.output),
        taxonomy_path=args.taxonomy,
        domain_taxonomy_path=args.domain_taxonomy,
        model_name=args.model,
        capability_threshold=args.capability_threshold,
        domain_threshold=args.domain_threshold,
        max_capabilities=args.max_capabilities,
        max_domains=args.max_domains,
        max_chars=args.max_chars,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def llm_label(args: argparse.Namespace) -> int:
    summary = llm_label_jsonl(
        input_path=artifact_path(args.input),
        output_path=artifact_path(args.output),
        taxonomy_path=args.taxonomy,
        domain_taxonomy_path=args.domain_taxonomy,
        model_name=args.model,
        max_capabilities=args.max_capabilities,
        max_domains=args.max_domains,
        max_chars=args.max_chars,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


def report_labels(args: argparse.Namespace) -> int:
    if args.output:
        report = write_label_report(
            artifact_path(args.input),
            artifact_path(args.output),
            top_examples=args.top_examples,
        )
    else:
        report = label_report(artifact_path(args.input), top_examples=args.top_examples)
    print(json.dumps(report, indent=2))
    return 0


def export_classifier(args: argparse.Namespace) -> int:
    summary = export_classifier_jsonl(
        input_path=artifact_path(args.input),
        output_path=artifact_path(args.output),
        taxonomy_path=args.taxonomy,
        domain_taxonomy_path=args.domain_taxonomy,
        min_score=args.min_score,
        max_chars=args.max_chars,
        include_unlabeled=args.include_unlabeled,
    )
    print(json.dumps(summary, indent=2))
    return 0


def export_labeling_prompts(args: argparse.Namespace) -> int:
    summary = export_labeling_prompts_jsonl(
        input_path=artifact_path(args.input),
        output_path=artifact_path(args.output),
        taxonomy_path=args.taxonomy,
        domain_taxonomy_path=args.domain_taxonomy,
        examples_path=artifact_path(args.examples) if args.examples else None,
        max_examples_per_label=args.max_examples_per_label,
        max_chars=args.max_chars,
        allow_weak_examples=args.allow_weak_examples,
    )
    print(json.dumps(summary, indent=2))
    return 0


def taxonomy_docs(args: argparse.Namespace) -> int:
    summary = write_taxonomy_markdown(args.taxonomy, artifact_path(args.output))
    print(json.dumps(summary, indent=2))
    return 0


def list_specialist_targets(args: argparse.Namespace) -> int:
    targets = load_specialist_targets(args.specialist_targets)
    print(
        json.dumps(
            {
                "specialist_targets": targets_to_dicts(targets),
                "count": len(targets),
            },
            indent=2,
        )
    )
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
    subsets_cmd.add_argument("--target-id")
    subsets_cmd.add_argument("--specialist-targets", type=Path, default=DEFAULT_SPECIALIST_TARGETS_PATH)
    subsets_cmd.add_argument("--target-capability", action="append")
    subsets_cmd.add_argument("--target-domain", action="append", default=[])
    subsets_cmd.add_argument("--output-prefix")
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

    relabel_cmd = subcommands.add_parser(
        "relabel-capabilities",
        help="rewrite canonical JSONL with normalized capability/domain labels and evidence",
    )
    relabel_cmd.add_argument("--input", type=Path, required=True)
    relabel_cmd.add_argument("--output", type=Path, required=True)
    relabel_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    relabel_cmd.add_argument("--domain-taxonomy", type=Path, default=DEFAULT_DOMAIN_TAXONOMY)
    relabel_cmd.add_argument("--min-score", type=float, default=0.65)
    relabel_cmd.add_argument("--max-chars", type=int, default=24000)
    relabel_cmd.set_defaults(func=relabel_capabilities)

    model_label_cmd = subcommands.add_parser(
        "model-label",
        help="label canonical JSONL with a lightweight zero-shot classifier model",
    )
    model_label_cmd.add_argument("--input", type=Path, required=True)
    model_label_cmd.add_argument("--output", type=Path, required=True)
    model_label_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    model_label_cmd.add_argument("--domain-taxonomy", type=Path, default=DEFAULT_DOMAIN_TAXONOMY)
    model_label_cmd.add_argument("--model", default=DEFAULT_ZERO_SHOT_MODEL)
    model_label_cmd.add_argument("--capability-threshold", type=float, default=0.65)
    model_label_cmd.add_argument("--domain-threshold", type=float, default=0.65)
    model_label_cmd.add_argument("--max-capabilities", type=int, default=4)
    model_label_cmd.add_argument("--max-domains", type=int, default=2)
    model_label_cmd.add_argument("--max-chars", type=int, default=12000)
    model_label_cmd.add_argument("--batch-size", type=int, default=16)
    model_label_cmd.add_argument("--device", type=int)
    model_label_cmd.add_argument("--limit", type=int)
    model_label_cmd.set_defaults(func=model_label)

    llm_label_cmd = subcommands.add_parser(
        "llm-label",
        help="label canonical JSONL with a small generative instruction model",
    )
    llm_label_cmd.add_argument("--input", type=Path, required=True)
    llm_label_cmd.add_argument("--output", type=Path, required=True)
    llm_label_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    llm_label_cmd.add_argument("--domain-taxonomy", type=Path, default=DEFAULT_DOMAIN_TAXONOMY)
    llm_label_cmd.add_argument("--model", default=DEFAULT_LLM_LABEL_MODEL)
    llm_label_cmd.add_argument("--max-capabilities", type=int, default=4)
    llm_label_cmd.add_argument("--max-domains", type=int, default=2)
    llm_label_cmd.add_argument("--max-chars", type=int, default=12000)
    llm_label_cmd.add_argument("--max-input-tokens", type=int, default=8192)
    llm_label_cmd.add_argument("--max-new-tokens", type=int, default=384)
    llm_label_cmd.add_argument("--device", type=int)
    llm_label_cmd.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    llm_label_cmd.add_argument("--trust-remote-code", action="store_true")
    llm_label_cmd.add_argument("--limit", type=int)
    llm_label_cmd.set_defaults(func=llm_label)

    report_cmd = subcommands.add_parser(
        "label-report",
        help="summarize capability/domain labels in a canonical JSONL file",
    )
    report_cmd.add_argument("--input", type=Path, required=True)
    report_cmd.add_argument("--output", type=Path)
    report_cmd.add_argument("--top-examples", type=int, default=0)
    report_cmd.set_defaults(func=report_labels)

    export_cmd = subcommands.add_parser(
        "export-classifier-data",
        help="export text/label-vector JSONL for encoder scoring or optional supervised checks",
    )
    export_cmd.add_argument("--input", type=Path, required=True)
    export_cmd.add_argument("--output", type=Path, required=True)
    export_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    export_cmd.add_argument("--domain-taxonomy", type=Path, default=DEFAULT_DOMAIN_TAXONOMY)
    export_cmd.add_argument("--min-score", type=float, default=0.65)
    export_cmd.add_argument("--max-chars", type=int, default=24000)
    export_cmd.add_argument("--include-unlabeled", action="store_true")
    export_cmd.set_defaults(func=export_classifier)

    prompt_cmd = subcommands.add_parser(
        "export-labeling-prompts",
        help="export prompt/ICL JSONL for generative rollout capability/domain labeling",
    )
    prompt_cmd.add_argument("--input", type=Path, required=True)
    prompt_cmd.add_argument("--output", type=Path, required=True)
    prompt_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    prompt_cmd.add_argument("--domain-taxonomy", type=Path, default=DEFAULT_DOMAIN_TAXONOMY)
    prompt_cmd.add_argument("--examples", type=Path)
    prompt_cmd.add_argument("--max-examples-per-label", type=int, default=2)
    prompt_cmd.add_argument("--max-chars", type=int, default=12000)
    prompt_cmd.add_argument(
        "--allow-weak-examples",
        action="store_true",
        help="allow weak heuristic labels to be used as ICL examples",
    )
    prompt_cmd.set_defaults(func=export_labeling_prompts)

    taxonomy_cmd = subcommands.add_parser(
        "taxonomy-docs",
        help="render a taxonomy JSON file to Markdown for review",
    )
    taxonomy_cmd.add_argument("--taxonomy", type=Path, default=DEFAULT_CAPABILITY_TAXONOMY)
    taxonomy_cmd.add_argument("--output", type=Path, required=True)
    taxonomy_cmd.set_defaults(func=taxonomy_docs)

    targets_cmd = subcommands.add_parser(
        "list-specialist-targets",
        help="list configured distillation targets",
    )
    targets_cmd.add_argument(
        "--specialist-targets",
        type=Path,
        default=DEFAULT_SPECIALIST_TARGETS_PATH,
    )
    targets_cmd.set_defaults(func=list_specialist_targets)

    return root


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
