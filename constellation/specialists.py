"""Specialist target registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPECIALIST_TARGETS_PATH = Path("configs/specialist_targets.json")


@dataclass(frozen=True)
class SpecialistTarget:
    id: str
    model_name: str
    target_capabilities: tuple[str, ...]
    target_domains: tuple[str, ...]
    description: str


def load_specialist_targets(path: str | Path = DEFAULT_SPECIALIST_TARGETS_PATH) -> list[SpecialistTarget]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    max_distillations = int(payload.get("max_distillations", 20))
    raw_targets = payload.get("targets") or []
    if len(raw_targets) > max_distillations:
        raise ValueError(f"{path} defines {len(raw_targets)} targets, max is {max_distillations}")
    targets = [
        SpecialistTarget(
            id=str(row["id"]),
            model_name=str(row["model_name"]),
            target_capabilities=tuple(row.get("target_capabilities") or ()),
            target_domains=tuple(row.get("target_domains") or ()),
            description=str(row.get("description", "")),
        )
        for row in raw_targets
    ]
    ids = [target.id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate specialist target ids")
    return targets


def targets_to_dicts(targets: list[SpecialistTarget]) -> list[dict[str, Any]]:
    return [
        {
            "id": target.id,
            "model_name": target.model_name,
            "target_capabilities": list(target.target_capabilities),
            "target_domains": list(target.target_domains),
            "description": target.description,
        }
        for target in targets
    ]
