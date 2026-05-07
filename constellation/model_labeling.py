"""Model-based lightweight labeling for canonical rollout datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from constellation.categorization import task_focused_text
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample
from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy

DEFAULT_ZERO_SHOT_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"


class OptionalLabelingDependencyError(RuntimeError):
    """Raised when model-label dependencies are not installed."""


def taxonomy_candidates(taxonomy: CapabilityTaxonomy) -> dict[str, str]:
    return {
        capability.name: capability.classifier_label
        for capability in taxonomy.capabilities
    }


def select_labels(
    scores: dict[str, float],
    *,
    threshold: float,
    max_labels: int,
) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    selected = [label for label, score in ranked if score >= threshold]
    if max_labels > 0:
        selected = selected[:max_labels]
    return selected


def require_zero_shot_pipeline(model_name: str, device: int | None) -> Any:
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise OptionalLabelingDependencyError(
            "model-label requires lightweight labeling dependencies. Install on the GPU machine with:\n"
            "uv pip install -r requirements/labeling.txt"
        ) from exc

    if device is None:
        resolved_device = 0 if torch.cuda.is_available() else -1
    else:
        resolved_device = device
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=resolved_device,
    )


def score_axis(
    classifier: Any,
    *,
    text: str,
    candidates: dict[str, str],
    hypothesis_template: str,
    batch_size: int,
) -> dict[str, float]:
    reverse = {value: key for key, value in candidates.items()}
    result = classifier(
        text,
        list(candidates.values()),
        multi_label=True,
        hypothesis_template=hypothesis_template,
        batch_size=batch_size,
    )
    return {
        reverse[label_text]: float(score)
        for label_text, score in zip(result["labels"], result["scores"])
        if label_text in reverse
    }


def label_sample_with_model(
    sample: CanonicalSample,
    *,
    classifier: Any,
    capability_taxonomy: CapabilityTaxonomy,
    domain_taxonomy: DomainTaxonomy,
    model_name: str,
    capability_threshold: float,
    domain_threshold: float,
    max_capabilities: int,
    max_domains: int,
    max_chars: int,
    batch_size: int,
) -> CanonicalSample:
    text = task_focused_text(sample, max_chars=max_chars)
    capability_scores = score_axis(
        classifier,
        text=text,
        candidates=taxonomy_candidates(capability_taxonomy),
        hypothesis_template="This trajectory teaches {}.",
        batch_size=batch_size,
    )
    domain_scores = score_axis(
        classifier,
        text=text,
        candidates=taxonomy_candidates(domain_taxonomy),
        hypothesis_template="The subject domain of this trajectory is {}.",
        batch_size=batch_size,
    )

    capabilities = select_labels(
        capability_scores,
        threshold=capability_threshold,
        max_labels=max_capabilities,
    )
    domains = select_labels(
        domain_scores,
        threshold=domain_threshold,
        max_labels=max_domains,
    )

    metadata = dict(sample.metadata)
    metadata["capability_labeling"] = {
        "taxonomy_version": capability_taxonomy.version,
        "method": "zero_shot_nli_v1",
        "model": model_name,
        "threshold": capability_threshold,
        "max_labels": max_capabilities,
        "scores": capability_scores,
        "previous_capabilities": sample.capabilities,
    }
    metadata["domain_labeling"] = {
        "taxonomy_version": domain_taxonomy.version,
        "method": "zero_shot_nli_v1",
        "model": model_name,
        "threshold": domain_threshold,
        "max_labels": max_domains,
        "scores": domain_scores,
        "previous_domains": sample.domains,
    }

    sample.capabilities = capabilities
    sample.domains = domains
    sample.metadata = metadata
    return sample


def model_label_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    taxonomy_path: str | Path,
    domain_taxonomy_path: str | Path,
    model_name: str = DEFAULT_ZERO_SHOT_MODEL,
    capability_threshold: float = 0.65,
    domain_threshold: float = 0.65,
    max_capabilities: int = 4,
    max_domains: int = 2,
    max_chars: int = 12000,
    batch_size: int = 16,
    device: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    capability_taxonomy = CapabilityTaxonomy.load(taxonomy_path)
    domain_taxonomy = DomainTaxonomy.load(domain_taxonomy_path)
    classifier = require_zero_shot_pipeline(model_name, device)
    capability_counts = {label: 0 for label in capability_taxonomy.names}
    domain_counts = {label: 0 for label in domain_taxonomy.names}
    written = 0

    def rows() -> Any:
        nonlocal written
        for row in iter_jsonl(input_path):
            if limit is not None and written >= limit:
                break
            sample = label_sample_with_model(
                CanonicalSample.from_dict(row),
                classifier=classifier,
                capability_taxonomy=capability_taxonomy,
                domain_taxonomy=domain_taxonomy,
                model_name=model_name,
                capability_threshold=capability_threshold,
                domain_threshold=domain_threshold,
                max_capabilities=max_capabilities,
                max_domains=max_domains,
                max_chars=max_chars,
                batch_size=batch_size,
            )
            for label in sample.capabilities:
                capability_counts[label] = capability_counts.get(label, 0) + 1
            for label in sample.domains:
                domain_counts[label] = domain_counts.get(label, 0) + 1
            written += 1
            yield sample.to_dict()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows())
    return {
        "input": str(input_path),
        "output": str(output_path),
        "model": model_name,
        "taxonomy_version": capability_taxonomy.version,
        "domain_taxonomy_version": domain_taxonomy.version,
        "written": written,
        "thresholds": {
            "capability": capability_threshold,
            "domain": domain_threshold,
        },
        "max_labels": {
            "capabilities": max_capabilities,
            "domains": max_domains,
        },
        "capability_counts": {
            key: value for key, value in capability_counts.items() if value
        },
        "domain_counts": {key: value for key, value in domain_counts.items() if value},
    }
