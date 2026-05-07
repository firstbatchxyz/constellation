"""Dataset-wide capability relabeling and classifier export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constellation.io import iter_jsonl, write_jsonl
from constellation.labeling import label_capability_evidence
from constellation.schema import CanonicalSample
from constellation.taxonomy import CapabilityTaxonomy


def classifier_text(sample: CanonicalSample, *, max_chars: int = 24000) -> str:
    """Render a bounded trajectory text for encoder classification."""
    parts: list[str] = []
    for turn in sample.messages:
        parts.append(f"{turn.role}:{turn.type}\n{turn.content.strip()}")
    text = "\n\n".join(parts)
    if len(text) <= max_chars:
        return text

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return text[:head_chars] + "\n\n[...truncated...]\n\n" + text[-tail_chars:]


def relabel_sample(
    sample: CanonicalSample,
    *,
    taxonomy: CapabilityTaxonomy,
    min_score: float,
    max_chars: int,
) -> CanonicalSample:
    text = classifier_text(sample, max_chars=max_chars)
    row = sample.to_dict()
    evidence = label_capability_evidence(row=row, text=text, taxonomy=taxonomy)
    labels = taxonomy.validate_labels(
        [item.label for item in evidence if item.score >= min_score]
    )

    metadata = dict(sample.metadata)
    metadata["capability_labeling"] = {
        "taxonomy_version": taxonomy.version,
        "method": "weak_heuristic_v1",
        "min_score": min_score,
        "evidence": [item.to_dict() for item in evidence],
        "previous_capabilities": sample.capabilities,
    }

    sample.capabilities = labels
    sample.metadata = metadata
    return sample


def relabel_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    min_score: float = 0.45,
    max_chars: int = 24000,
) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    label_counts = {label: 0 for label in taxonomy.names}
    total = 0

    def rows() -> Any:
        nonlocal total
        for row in iter_jsonl(input_path):
            sample = relabel_sample(
                CanonicalSample.from_dict(row),
                taxonomy=taxonomy,
                min_score=min_score,
                max_chars=max_chars,
            )
            total += 1
            for label in sample.capabilities:
                label_counts[label] = label_counts.get(label, 0) + 1
            yield sample.to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "taxonomy_version": taxonomy.version,
        "written": written,
        "label_counts": {key: value for key, value in label_counts.items() if value},
    }


def export_classifier_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    min_score: float = 0.45,
    max_chars: int = 24000,
    include_unlabeled: bool = False,
) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    written = 0
    skipped = 0

    def rows() -> Any:
        nonlocal written, skipped
        for row in iter_jsonl(input_path):
            sample = CanonicalSample.from_dict(row)
            text = classifier_text(sample, max_chars=max_chars)
            labels = taxonomy.validate_labels(sample.capabilities)
            if not labels:
                evidence = label_capability_evidence(row=sample.to_dict(), text=text, taxonomy=taxonomy)
                labels = taxonomy.validate_labels(
                    [item.label for item in evidence if item.score >= min_score]
                )
            if not labels and not include_unlabeled:
                skipped += 1
                continue
            written += 1
            yield {
                "id": sample.id,
                "text": text,
                "labels": labels,
                "label_vector": [1 if label in labels else 0 for label in taxonomy.names],
                "taxonomy_version": taxonomy.version,
                "source_dataset": sample.source_dataset,
                "metadata": {
                    "quality_score": sample.quality_score,
                    "success": sample.success,
                    "capabilities": sample.capabilities,
                },
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "taxonomy_version": taxonomy.version,
        "labels": list(taxonomy.names),
        "written": written,
        "skipped": skipped,
    }


def write_taxonomy_markdown(taxonomy_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    lines = [f"# {taxonomy.version}", ""]
    for capability in taxonomy.capabilities:
        lines.append(f"## {capability.name}")
        lines.append("")
        lines.append(capability.description or "No description.")
        if capability.positive_cues:
            lines.append("")
            lines.append("Cues: " + ", ".join(f"`{cue}`" for cue in capability.positive_cues))
        if capability.source_aliases:
            lines.append("")
            lines.append("Aliases: " + ", ".join(f"`{alias}`" for alias in capability.source_aliases))
        lines.append("")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    return {"output": str(output_path), "capability_count": len(taxonomy.names)}

