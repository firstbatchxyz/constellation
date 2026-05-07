"""Streaming dataset ingestion for curated pilot shards."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from constellation.filtering import passes_basic_filters
from constellation.io import write_jsonl
from constellation.parsers import parse_agenttrove_row, parse_hermes_row
from constellation.schema import CanonicalSample
from constellation.scoring import with_quality_score


@dataclass(frozen=True)
class DatasetSource:
    dataset_path: str
    parser: str
    split: str = "train"
    config_name: str | None = None
    source_dataset: str | None = None


DATASET_SOURCES = {
    "agenttrove": DatasetSource(
        dataset_path="open-thoughts/AgentTrove",
        parser="agenttrove",
        split="train",
        source_dataset="open-thoughts/AgentTrove",
    ),
    "hermes-kimi": DatasetSource(
        dataset_path="lambda/hermes-agent-reasoning-traces",
        config_name="kimi",
        parser="hermes",
        split="train",
        source_dataset="lambda/hermes-agent-reasoning-traces:kimi",
    ),
    "hermes-glm": DatasetSource(
        dataset_path="lambda/hermes-agent-reasoning-traces",
        config_name="glm-5.1",
        parser="hermes",
        split="train",
        source_dataset="lambda/hermes-agent-reasoning-traces:glm-5.1",
    ),
}


PARSER_BY_NAME = {
    "agenttrove": parse_agenttrove_row,
    "hermes": parse_hermes_row,
}


def resolve_source(
    source: str,
    *,
    dataset_path: str | None = None,
    config_name: str | None = None,
    split: str | None = None,
    parser: str | None = None,
) -> DatasetSource:
    base = DATASET_SOURCES.get(source)
    if base is None and dataset_path is None:
        raise ValueError(f"unknown source {source!r}; pass --dataset-path for custom sources")
    if base is None:
        base = DatasetSource(dataset_path=dataset_path or "", parser=parser or "")

    resolved = DatasetSource(
        dataset_path=dataset_path or base.dataset_path,
        config_name=config_name if config_name is not None else base.config_name,
        split=split or base.split,
        parser=parser or base.parser,
        source_dataset=base.source_dataset,
    )
    if resolved.parser not in PARSER_BY_NAME:
        raise ValueError(f"unknown parser {resolved.parser!r}")
    return resolved


def iter_hf_rows(source: DatasetSource) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "stream-convert requires the optional Hugging Face datasets package. "
            "Install it on the GPU machine with: uv pip install datasets"
        ) from exc

    args: list[str] = [source.dataset_path]
    if source.config_name:
        args.append(source.config_name)
    return iter(load_dataset(*args, split=source.split, streaming=True))


def parse_streamed_row(row: dict[str, Any], source: DatasetSource) -> CanonicalSample:
    parser = PARSER_BY_NAME[source.parser]
    if source.parser == "agenttrove":
        return parser(row, source_dataset=source.source_dataset or source.dataset_path)
    if source.parser == "hermes":
        return parser(row, source_dataset=source.source_dataset or source.dataset_path)
    raise ValueError(f"unknown parser {source.parser!r}")


def iter_limited_hf_rows(source: DatasetSource, max_rows: int | None) -> Any:
    rows = iter_hf_rows(source)
    if max_rows is None or max_rows <= 0:
        return rows
    return islice(rows, max_rows)


def write_sharded_jsonl(
    output_dir: str | Path,
    rows: Any,
    *,
    shard_prefix: str,
    shard_size: int,
) -> tuple[int, list[str]]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    count = 0
    shard_count = 0
    rows_in_shard = 0
    paths: list[str] = []
    handle: Any = None
    try:
        for row in rows:
            if handle is None or rows_in_shard >= shard_size:
                if handle is not None:
                    handle.close()
                shard_path = output_root / f"{shard_prefix}-{shard_count:05d}.jsonl"
                handle = shard_path.open("w", encoding="utf-8")
                paths.append(str(shard_path))
                shard_count += 1
                rows_in_shard = 0
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
            rows_in_shard += 1
    finally:
        if handle is not None:
            handle.close()
    return count, paths


def stream_convert(
    *,
    source: DatasetSource,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    shard_prefix: str | None = None,
    shard_size: int = 50_000,
    max_rows: int,
    min_tokens: int,
    max_tokens: int,
    min_quality: float,
    require_success: bool,
    skip_errors: bool,
    max_error_examples: int = 3,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "seen": 0,
        "parsed": 0,
        "written": 0,
        "skipped": 0,
        "errors": 0,
        "error_types": {},
        "error_examples": [],
        "output_paths": [],
    }
    error_types: Counter[str] = Counter()

    def rows() -> Any:
        for raw_row in iter_limited_hf_rows(source, max_rows):
            stats["seen"] += 1
            try:
                sample = with_quality_score(parse_streamed_row(raw_row, source))
                stats["parsed"] += 1
            except Exception as exc:
                stats["errors"] += 1
                error_name = type(exc).__name__
                error_types[error_name] += 1
                stats["error_types"] = dict(error_types)
                if len(stats["error_examples"]) < max_error_examples:
                    stats["error_examples"].append(
                        {
                            "row_number": stats["seen"],
                            "error_type": error_name,
                            "error": str(exc),
                            "row_keys": sorted(str(key) for key in raw_row.keys()),
                        }
                    )
                if skip_errors:
                    continue
                raise

            if sample.quality_score < min_quality:
                stats["skipped"] += 1
                continue
            if not passes_basic_filters(
                sample,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                require_success=require_success,
            ):
                stats["skipped"] += 1
                continue
            stats["written"] += 1
            yield sample.to_dict()

    if (output is None) == (output_dir is None):
        raise ValueError("provide exactly one of output or output_dir")
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output, rows())
        stats["output_paths"] = [str(output)]
    else:
        prefix = shard_prefix or (source.source_dataset or source.dataset_path).replace("/", "_").replace(":", "_")
        _, paths = write_sharded_jsonl(
            output_dir or "",
            rows(),
            shard_prefix=prefix,
            shard_size=shard_size,
        )
        stats["output_paths"] = paths
    return stats
