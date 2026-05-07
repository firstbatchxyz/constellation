"""Deterministic pilot subset construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from constellation.filtering import passes_basic_filters, token_count_estimate
from constellation.io import iter_jsonl, write_jsonl
from constellation.schema import CanonicalSample


@dataclass(frozen=True)
class Selection:
    samples: list[CanonicalSample]
    tokens: int


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_score(value: str) -> float:
    return int(stable_digest(value)[:16], 16) / float(16**16)


def first_user_text(sample: CanonicalSample) -> str:
    for turn in sample.messages:
        if turn.role == "user":
            return turn.content
    return sample.joined_text()[:1000]


def group_key(sample: CanonicalSample) -> str:
    metadata = sample.metadata
    original_id = metadata.get("original_id") or metadata.get("task_id")
    repo = metadata.get("repo") or metadata.get("repository")
    prompt_hash = stable_digest(first_user_text(sample))[:16]
    return "|".join(
        [
            sample.source_dataset,
            str(repo or ""),
            str(original_id or prompt_hash),
        ]
    )


def load_samples(paths: list[Path]) -> list[CanonicalSample]:
    samples: list[CanonicalSample] = []
    for path in paths:
        for row in iter_jsonl(path):
            samples.append(CanonicalSample.from_dict(row))
    return samples


def deterministic_order(sample: CanonicalSample, seed: str) -> str:
    return stable_digest(f"{seed}|{group_key(sample)}|{sample.id}")


def select_by_token_budget(
    samples: list[CanonicalSample],
    *,
    token_budget: int,
    seed: str,
) -> Selection:
    selected: list[CanonicalSample] = []
    tokens = 0
    for sample in sorted(samples, key=lambda item: deterministic_order(item, seed)):
        sample_tokens = token_count_estimate(sample)
        if tokens + sample_tokens > token_budget and selected:
            continue
        if tokens >= token_budget:
            break
        selected.append(sample)
        tokens += sample_tokens
    return Selection(selected, tokens)


def write_samples(path: Path, samples: list[CanonicalSample]) -> int:
    return write_jsonl(path, (sample.to_dict() for sample in samples))


def safe_name(value: str) -> str:
    return value.lower().replace("/", "_").replace(" ", "_")


def default_output_prefix(target_capabilities: list[str], target_domains: list[str]) -> str:
    if target_capabilities == ["DEBUGGING"] and not target_domains:
        return "debugging"
    parts = [safe_name(item) for item in target_domains + target_capabilities]
    return "_".join(parts) if parts else "general_agent"


def sample_matches_target(
    sample: CanonicalSample,
    *,
    target_capabilities: list[str],
    target_domains: list[str],
) -> bool:
    capability_match = (
        all(capability in sample.capabilities for capability in target_capabilities)
        if target_capabilities
        else True
    )
    domain_match = (
        any(domain in sample.domains for domain in target_domains)
        if target_domains
        else True
    )
    return capability_match and domain_match


def sample_matches_eval_filters(
    sample: CanonicalSample,
    *,
    eval_sample_types: list[str],
    eval_required_cues: list[str],
    eval_excluded_cues: list[str],
) -> bool:
    if eval_sample_types and sample.sample_type not in eval_sample_types:
        return False
    text = sample.joined_text().lower()
    required_cues = [cue.lower() for cue in eval_required_cues if cue]
    excluded_cues = [cue.lower() for cue in eval_excluded_cues if cue]
    if required_cues and not any(cue in text for cue in required_cues):
        return False
    if excluded_cues and any(cue in text for cue in excluded_cues):
        return False
    return True


def build_debugging_pilot_subsets(
    *,
    inputs: list[Path],
    output_dir: Path,
    target_capability: str = "DEBUGGING",
    target_capabilities: list[str] | None = None,
    target_domains: list[str] | None = None,
    output_prefix: str | None = None,
    max_train_tokens: int = 2_000_000,
    specialist_target_ratio: float = 0.8,
    eval_fraction: float = 0.1,
    eval_max_samples: int = 200,
    min_quality: float = 0.45,
    min_tokens: int = 64,
    max_tokens: int = 32768,
    eval_sample_types: list[str] | None = None,
    eval_required_cues: list[str] | None = None,
    eval_excluded_cues: list[str] | None = None,
    seed: str = "constellation-v1",
) -> dict[str, Any]:
    if not 0.0 < specialist_target_ratio <= 1.0:
        raise ValueError("specialist_target_ratio must be in (0, 1]")
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")

    resolved_capabilities = target_capabilities or ([target_capability] if target_capability else [])
    resolved_domains = target_domains or []
    resolved_eval_sample_types = eval_sample_types or []
    resolved_eval_required_cues = eval_required_cues or []
    resolved_eval_excluded_cues = eval_excluded_cues or []
    prefix = output_prefix or default_output_prefix(resolved_capabilities, resolved_domains)

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        sample
        for sample in load_samples(inputs)
        if sample.quality_score >= min_quality
        and passes_basic_filters(sample, min_tokens=min_tokens, max_tokens=max_tokens)
    ]

    target_samples = [
        sample
        for sample in samples
        if sample_matches_target(
            sample,
            target_capabilities=resolved_capabilities,
            target_domains=resolved_domains,
        )
    ]
    eval_candidate_samples = [
        sample
        for sample in target_samples
        if sample_matches_eval_filters(
            sample,
            eval_sample_types=resolved_eval_sample_types,
            eval_required_cues=resolved_eval_required_cues,
            eval_excluded_cues=resolved_eval_excluded_cues,
        )
    ]
    eval_groups = {
        group_key(sample)
        for sample in eval_candidate_samples
        if stable_score(f"{seed}|eval|{group_key(sample)}") < eval_fraction
    }
    eval_samples = [
        sample for sample in eval_candidate_samples if group_key(sample) in eval_groups
    ]
    eval_samples = sorted(eval_samples, key=lambda item: deterministic_order(item, f"{seed}|eval"))[
        :eval_max_samples
    ]
    if not eval_samples and eval_candidate_samples and eval_max_samples > 0:
        eval_samples = sorted(
            eval_candidate_samples,
            key=lambda item: deterministic_order(item, f"{seed}|eval-fallback"),
        )[:1]

    eval_group_keys = {group_key(sample) for sample in eval_samples}
    train_pool = [sample for sample in samples if group_key(sample) not in eval_group_keys]
    target_pool = [
        sample
        for sample in train_pool
        if sample_matches_target(
            sample,
            target_capabilities=resolved_capabilities,
            target_domains=resolved_domains,
        )
    ]
    anchor_pool = [
        sample
        for sample in train_pool
        if not sample_matches_target(
            sample,
            target_capabilities=resolved_capabilities,
            target_domains=resolved_domains,
        )
    ]

    target_budget = int(max_train_tokens * specialist_target_ratio)
    anchor_budget = max_train_tokens - target_budget
    target_selection = select_by_token_budget(
        target_pool,
        token_budget=target_budget,
        seed=f"{seed}|specialist-target",
    )
    anchor_selection = select_by_token_budget(
        anchor_pool,
        token_budget=anchor_budget,
        seed=f"{seed}|specialist-anchor",
    )
    specialist_samples = target_selection.samples + anchor_selection.samples
    specialist_tokens = target_selection.tokens + anchor_selection.tokens

    general_selection = select_by_token_budget(
        train_pool,
        token_budget=specialist_tokens,
        seed=f"{seed}|general",
    )

    paths = {
        "specialist_train": output_dir / f"{prefix}.train.jsonl",
        "general_agentic_mix_train": output_dir / f"general_agentic_mix.{prefix}.train.jsonl",
        "target_eval": output_dir / f"{prefix}.eval.jsonl",
        "manifest": output_dir / f"{prefix}.manifest.json",
    }
    if prefix == "debugging" and resolved_capabilities == ["DEBUGGING"] and not resolved_domains:
        paths = {
            "specialist_train": output_dir / "debugging_specialist.train.jsonl",
            "general_agentic_mix_train": output_dir / "general_agentic_mix.train.jsonl",
            "target_eval": output_dir / "debugging.eval.jsonl",
            "manifest": output_dir / "manifest.json",
        }

    counts = {
        "specialist_train": write_samples(paths["specialist_train"], specialist_samples),
        "general_agentic_mix_train": write_samples(
            paths["general_agentic_mix_train"], general_selection.samples
        ),
        "target_eval": write_samples(paths["target_eval"], eval_samples),
    }

    manifest = {
        "target_capability": target_capability,
        "target_capabilities": resolved_capabilities,
        "target_domains": resolved_domains,
        "output_prefix": prefix,
        "seed": seed,
        "inputs": [str(path) for path in inputs],
        "paths": {key: str(value) for key, value in paths.items()},
        "counts": counts,
        "tokens": {
            "specialist_train": specialist_tokens,
            "general_agentic_mix_train": general_selection.tokens,
            "target_eval": sum(token_count_estimate(sample) for sample in eval_samples),
        },
        "selection": {
            "max_train_tokens": max_train_tokens,
            "specialist_target_ratio": specialist_target_ratio,
            "eval_fraction": eval_fraction,
            "eval_max_samples": eval_max_samples,
            "min_quality": min_quality,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "eval_sample_types": resolved_eval_sample_types,
            "eval_required_cues": resolved_eval_required_cues,
            "eval_excluded_cues": resolved_eval_excluded_cues,
            "eval_candidate_samples": len(eval_candidate_samples),
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
