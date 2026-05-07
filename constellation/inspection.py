"""Inspection helpers for labeled canonical JSONL rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from constellation.categorization import task_focused_text
from constellation.io import iter_jsonl
from constellation.schema import CanonicalSample


def row_matches(
    row: dict[str, Any],
    *,
    sample_id: str | None,
    label: str | None,
    axis: str,
) -> bool:
    if sample_id and row.get("id") != sample_id:
        return False
    if label:
        values = row.get(axis) or []
        if label not in values:
            return False
    return True


def inspect_samples(
    input_path: str | Path,
    *,
    sample_id: str | None = None,
    label: str | None = None,
    axis: str = "domains",
    limit: int = 5,
    max_chars: int = 4000,
) -> dict[str, Any]:
    if axis not in {"capabilities", "domains"}:
        raise ValueError("axis must be capabilities or domains")
    if sample_id is None and label is None:
        raise ValueError("provide at least one of sample_id or label")

    matches: list[dict[str, Any]] = []
    for row in iter_jsonl(input_path):
        if not row_matches(row, sample_id=sample_id, label=label, axis=axis):
            continue
        sample = CanonicalSample.from_dict(row)
        metadata = dict(sample.metadata)
        matches.append(
            {
                "id": sample.id,
                "source_dataset": sample.source_dataset,
                "sample_type": sample.sample_type,
                "capabilities": sample.capabilities,
                "domains": sample.domains,
                "success": sample.success,
                "quality_score": sample.quality_score,
                "metadata": {
                    "capability_labeling": metadata.get("capability_labeling"),
                    "domain_labeling": metadata.get("domain_labeling"),
                    "label_guardrails": metadata.get("label_guardrails"),
                    "original_id": metadata.get("original_id"),
                    "teacher_model": metadata.get("teacher_model"),
                },
                "text": task_focused_text(sample, max_chars=max_chars),
            }
        )
        if len(matches) >= limit:
            break

    return {
        "input": str(input_path),
        "filters": {
            "id": sample_id,
            "label": label,
            "axis": axis,
            "limit": limit,
            "max_chars": max_chars,
        },
        "count": len(matches),
        "samples": matches,
    }
