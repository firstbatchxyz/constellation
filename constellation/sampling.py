"""Deterministic JSONL sampling helpers for curated canonical shards."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from constellation.io import iter_jsonl, write_jsonl


def group_key(row: dict[str, Any], group_by: str) -> str:
    if group_by == "none":
        return "all"
    if group_by == "source_dataset":
        return str(row.get("source_dataset") or "unknown")
    if group_by == "sample_type":
        return str(row.get("sample_type") or "unknown")
    if group_by == "source_dataset+sample_type":
        return f"{row.get('source_dataset') or 'unknown'}::{row.get('sample_type') or 'unknown'}"
    raise ValueError(f"unsupported group_by: {group_by}")


def sample_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    group_by: str = "source_dataset",
    max_per_group: int = 10,
    limit: int | None = None,
    seed: str = "constellation-v1",
) -> dict[str, Any]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {}
    seen_by_group: dict[str, int] = {}

    for row in iter_jsonl(input_path):
        key = group_key(row, group_by)
        seen_by_group[key] = seen_by_group.get(key, 0) + 1
        buckets.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    selected_by_group: dict[str, int] = {}
    for key in sorted(buckets):
        rows = list(buckets[key])
        rng.shuffle(rows)
        if max_per_group > 0:
            rows = rows[:max_per_group]
        selected.extend(rows)
        selected_by_group[key] = len(rows)

    rng.shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
        selected_by_group = {}
        for row in selected:
            key = group_key(row, group_by)
            selected_by_group[key] = selected_by_group.get(key, 0) + 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(output_path, selected)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "group_by": group_by,
        "seed": seed,
        "seen": sum(seen_by_group.values()),
        "written": written,
        "max_per_group": max_per_group,
        "limit": limit,
        "seen_by_group": dict(sorted(seen_by_group.items())),
        "selected_by_group": dict(sorted(selected_by_group.items())),
    }
