"""Dataset-wide capability relabeling and classifier export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constellation.io import iter_jsonl, write_jsonl
from constellation.labeling import label_capability_evidence, label_domain_evidence
from constellation.schema import CanonicalSample
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy


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


def task_focused_text(sample: CanonicalSample, *, max_chars: int = 12000) -> str:
    """Extract text useful for weak labeling without rollout harness boilerplate."""
    user_texts = [turn.content for turn in sample.messages if turn.role == "user"]
    text = "\n\n".join(user_texts) if user_texts else sample.joined_text()

    start_markers = (
        "Task Description:",
        "## Problem Description",
        "Problem Description",
        "Task:",
    )
    stop_markers = (
        "Current terminal state:",
        "Current Terminal Screen:",
        "New Terminal Output:",
        "\nassistant:",
    )
    lower_text = text.lower()
    start = 0
    for marker in start_markers:
        index = lower_text.find(marker.lower())
        if index >= 0:
            start = index
            break
    text = text[start:]
    lower_text = text.lower()
    stop_indexes = [
        lower_text.find(marker.lower())
        for marker in stop_markers
        if lower_text.find(marker.lower()) >= 0
    ]
    if stop_indexes:
        text = text[: min(stop_indexes)]

    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def taxonomy_prompt(taxonomy: CapabilityTaxonomy, *, label_axis: str = "Capability") -> str:
    lines = [f"{label_axis} taxonomy. Use only these exact labels:"]
    for capability in taxonomy.capabilities:
        lines.append(f"- {capability.name}: {capability.description}")
    return "\n".join(lines)


def load_icl_examples(
    *,
    examples_path: str | Path | None,
    taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    max_examples_per_label: int,
    max_chars: int,
    allow_weak_examples: bool = False,
) -> list[dict[str, Any]]:
    if examples_path is None:
        return []

    counts = {label: 0 for label in taxonomy.names}
    examples: list[dict[str, Any]] = []
    for row in iter_jsonl(examples_path):
        sample = CanonicalSample.from_dict(row)
        capability_method = (sample.metadata.get("capability_labeling") or {}).get("method")
        domain_method = (sample.metadata.get("domain_labeling") or {}).get("method")
        is_reviewed = capability_method in {"manual_review_v1", "prompt_icl_v1"} or domain_method in {
            "manual_review_v1",
            "prompt_icl_v1",
        }
        if not allow_weak_examples and not is_reviewed:
            continue
        labels = taxonomy.validate_labels(sample.capabilities)
        domains = domain_taxonomy.validate_labels(sample.domains)
        if not labels and not domains:
            continue
        if labels and all(counts[label] >= max_examples_per_label for label in labels):
            continue
        for label in labels:
            counts[label] += 1
        examples.append(
            {
                "id": sample.id,
                "text": classifier_text(sample, max_chars=max_chars),
                "capabilities": labels,
                "domains": domains,
            }
        )
        if all(count >= max_examples_per_label for count in counts.values()):
            break
    return examples


def labeling_prompt(
    sample: CanonicalSample,
    *,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    examples: list[dict[str, Any]],
    max_chars: int,
) -> str:
    lines = [
        "You are labeling rollout trajectories for specialist distillation.",
        taxonomy_prompt(capability_taxonomy, label_axis="Capability"),
        "",
        taxonomy_prompt(domain_taxonomy, label_axis="Domain"),
        "",
        "Return strict JSON only with this schema:",
        '{"capabilities":["STRUCTURED_REASONING"],"domains":["SCIENCE"],"confidence":0.0,"rationale":"short evidence summary"}',
        "",
        "Rules:",
        "- This is multi-label classification on two axes.",
        "- Capabilities describe behavior; domains describe subject matter.",
        "- Use exact labels from the taxonomies only.",
        "- Use [] when no label clearly applies on an axis.",
        "- Scientific skill is usually a domain+capability intersection, for example SCIENCE plus STRUCTURED_REASONING.",
    ]

    if examples:
        lines.extend(["", "Examples:"])
        for index, example in enumerate(examples, start=1):
            lines.append(f"\nExample {index} trajectory:\n{example['text']}")
            lines.append("Example output:")
            lines.append(
                json.dumps(
                    {
                        "capabilities": example["capabilities"],
                        "domains": example["domains"],
                        "confidence": 0.9,
                    }
                )
            )

    lines.extend(
        [
            "",
            "Trajectory to label:",
            classifier_text(sample, max_chars=max_chars),
            "",
            "JSON output:",
        ]
    )
    return "\n".join(lines)


def relabel_sample(
    sample: CanonicalSample,
    *,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    min_score: float,
    max_chars: int,
) -> CanonicalSample:
    text = task_focused_text(sample, max_chars=max_chars)
    row = sample.to_dict()
    capability_evidence = label_capability_evidence(
        row=row,
        text=text,
        taxonomy=capability_taxonomy,
    )
    domain_evidence = label_domain_evidence(row=row, text=text, taxonomy=domain_taxonomy)
    labels = capability_taxonomy.validate_labels(
        [item.label for item in capability_evidence if item.score >= min_score]
    )
    domains = domain_taxonomy.validate_labels(
        [item.label for item in domain_evidence if item.score >= min_score]
    )

    metadata = dict(sample.metadata)
    metadata["capability_labeling"] = {
        "taxonomy_version": capability_taxonomy.version,
        "method": "weak_heuristic_v1",
        "min_score": min_score,
        "evidence": [item.to_dict() for item in capability_evidence],
        "previous_capabilities": sample.capabilities,
    }
    metadata["domain_labeling"] = {
        "taxonomy_version": domain_taxonomy.version,
        "method": "weak_heuristic_v1",
        "min_score": min_score,
        "evidence": [item.to_dict() for item in domain_evidence],
        "previous_domains": sample.domains,
    }

    sample.capabilities = labels
    sample.domains = domains
    sample.metadata = metadata
    return sample


def relabel_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    domain_taxonomy_path: str | Path,
    min_score: float = 0.65,
    max_chars: int = 24000,
) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    label_counts = {label: 0 for label in taxonomy.names}
    domain_counts = {label: 0 for label in domain_taxonomy.names}
    total = 0

    def rows() -> Any:
        nonlocal total
        for row in iter_jsonl(input_path):
            sample = relabel_sample(
                CanonicalSample.from_dict(row),
                capability_taxonomy=taxonomy,
                domain_taxonomy=domain_taxonomy,
                min_score=min_score,
                max_chars=max_chars,
            )
            total += 1
            for label in sample.capabilities:
                label_counts[label] = label_counts.get(label, 0) + 1
            for label in sample.domains:
                domain_counts[label] = domain_counts.get(label, 0) + 1
            yield sample.to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "taxonomy_version": taxonomy.version,
        "domain_taxonomy_version": domain_taxonomy.version,
        "written": written,
        "label_counts": {key: value for key, value in label_counts.items() if value},
        "domain_counts": {key: value for key, value in domain_counts.items() if value},
    }


def export_classifier_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    domain_taxonomy_path: str | Path,
    min_score: float = 0.65,
    max_chars: int = 24000,
    include_unlabeled: bool = False,
) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    written = 0
    skipped = 0

    def rows() -> Any:
        nonlocal written, skipped
        for row in iter_jsonl(input_path):
            sample = CanonicalSample.from_dict(row)
            text = classifier_text(sample, max_chars=max_chars)
            labels = taxonomy.validate_labels(sample.capabilities)
            domains = domain_taxonomy.validate_labels(sample.domains)
            if not labels:
                evidence = label_capability_evidence(row=sample.to_dict(), text=text, taxonomy=taxonomy)
                labels = taxonomy.validate_labels(
                    [item.label for item in evidence if item.score >= min_score]
                )
            if not domains:
                domain_evidence = label_domain_evidence(
                    row=sample.to_dict(),
                    text=text,
                    taxonomy=domain_taxonomy,
                )
                domains = domain_taxonomy.validate_labels(
                    [item.label for item in domain_evidence if item.score >= min_score]
                )
            if not labels and not domains and not include_unlabeled:
                skipped += 1
                continue
            written += 1
            yield {
                "id": sample.id,
                "text": text,
                "labels": labels,
                "domains": domains,
                "label_vector": [1 if label in labels else 0 for label in taxonomy.names],
                "domain_vector": [1 if label in domains else 0 for label in domain_taxonomy.names],
                "taxonomy_version": taxonomy.version,
                "domain_taxonomy_version": domain_taxonomy.version,
                "source_dataset": sample.source_dataset,
                "metadata": {
                    "quality_score": sample.quality_score,
                    "success": sample.success,
                    "capabilities": sample.capabilities,
                    "domains": sample.domains,
                },
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "taxonomy_version": taxonomy.version,
        "domain_taxonomy_version": domain_taxonomy.version,
        "labels": list(taxonomy.names),
        "domains": list(domain_taxonomy.names),
        "written": written,
        "skipped": skipped,
    }


def export_labeling_prompts_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    domain_taxonomy_path: str | Path,
    examples_path: str | Path | None = None,
    max_examples_per_label: int = 2,
    max_chars: int = 12000,
    allow_weak_examples: bool = False,
) -> dict[str, Any]:
    taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    examples = load_icl_examples(
        examples_path=examples_path,
        taxonomy=taxonomy,
        domain_taxonomy=domain_taxonomy,
        max_examples_per_label=max_examples_per_label,
        max_chars=max_chars,
        allow_weak_examples=allow_weak_examples,
    )
    written = 0

    def rows() -> Any:
        nonlocal written
        for row in iter_jsonl(input_path):
            sample = CanonicalSample.from_dict(row)
            written += 1
            yield {
                "id": sample.id,
                "source_dataset": sample.source_dataset,
                "taxonomy_version": taxonomy.version,
                "domain_taxonomy_version": domain_taxonomy.version,
                "prompt": labeling_prompt(
                    sample,
                    capability_taxonomy=taxonomy,
                    domain_taxonomy=domain_taxonomy,
                    examples=examples,
                    max_chars=max_chars,
                ),
                "metadata": {
                    "quality_score": sample.quality_score,
                    "success": sample.success,
                    "existing_capabilities": sample.capabilities,
                    "existing_domains": sample.domains,
                },
            }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "taxonomy_version": taxonomy.version,
        "domain_taxonomy_version": domain_taxonomy.version,
        "written": written,
        "example_count": len(examples),
        "max_examples_per_label": max_examples_per_label,
        "allow_weak_examples": allow_weak_examples,
    }


def write_taxonomy_markdown(taxonomy_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    with Path(taxonomy_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    taxonomy = (
        DomainTaxonomy.from_dict(payload)
        if isinstance(payload.get("domains"), dict)
        else CapabilityTaxonomy.from_dict(payload)
    )
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
    return {"output": str(output_path), "label_count": len(taxonomy.names)}
