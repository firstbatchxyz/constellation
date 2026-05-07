"""Reports for labeled canonical datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from constellation.io import iter_jsonl


def label_report(input_path: str | Path, *, top_examples: int = 0) -> dict[str, Any]:
    capability_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    sample_type_counts: Counter[str] = Counter()
    capability_by_source: dict[str, Counter[str]] = {}
    domain_by_source: dict[str, Counter[str]] = {}
    capability_by_sample_type: dict[str, Counter[str]] = {}
    domain_by_sample_type: dict[str, Counter[str]] = {}
    capability_examples: dict[str, list[str]] = {}
    domain_examples: dict[str, list[str]] = {}
    empty = 0
    rows = 0

    for row in iter_jsonl(input_path):
        rows += 1
        capabilities = list(row.get("capabilities") or [])
        domains = list(row.get("domains") or [])
        source = str(row.get("source_dataset") or "unknown")
        sample_type = str(row.get("sample_type") or "unknown")
        source_counts.update([source])
        sample_type_counts.update([sample_type])
        capability_counts.update(capabilities)
        domain_counts.update(domains)
        capability_by_source.setdefault(source, Counter()).update(capabilities)
        domain_by_source.setdefault(source, Counter()).update(domains)
        capability_by_sample_type.setdefault(sample_type, Counter()).update(capabilities)
        domain_by_sample_type.setdefault(sample_type, Counter()).update(domains)
        if not capabilities and not domains:
            empty += 1

        if top_examples:
            sample_id = str(row.get("id", ""))
            for label in capabilities:
                capability_examples.setdefault(label, [])
                if len(capability_examples[label]) < top_examples:
                    capability_examples[label].append(sample_id)
            for label in domains:
                domain_examples.setdefault(label, [])
                if len(domain_examples[label]) < top_examples:
                    domain_examples[label].append(sample_id)

    report: dict[str, Any] = {
        "input": str(input_path),
        "rows": rows,
        "empty": empty,
        "empty_rate": round(empty / rows, 4) if rows else 0.0,
        "sources": dict(source_counts.most_common()),
        "sample_types": dict(sample_type_counts.most_common()),
        "capabilities": dict(capability_counts.most_common()),
        "domains": dict(domain_counts.most_common()),
        "breakdowns": {
            "capabilities_by_source": {
                key: dict(counter.most_common())
                for key, counter in sorted(capability_by_source.items())
            },
            "domains_by_source": {
                key: dict(counter.most_common())
                for key, counter in sorted(domain_by_source.items())
            },
            "capabilities_by_sample_type": {
                key: dict(counter.most_common())
                for key, counter in sorted(capability_by_sample_type.items())
            },
            "domains_by_sample_type": {
                key: dict(counter.most_common())
                for key, counter in sorted(domain_by_sample_type.items())
            },
        },
    }
    if top_examples:
        report["examples"] = {
            "capabilities": capability_examples,
            "domains": domain_examples,
        }
    return report


def write_label_report(input_path: str | Path, output_path: str | Path, *, top_examples: int = 0) -> dict[str, Any]:
    report = label_report(input_path, top_examples=top_examples)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
