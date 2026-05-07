"""Capability taxonomy loading and normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CAPABILITY_TAXONOMY_PATH = Path("configs/capability_taxonomy.json")
DEFAULT_DOMAIN_TAXONOMY_PATH = Path("configs/domain_taxonomy.json")


def normalize_label_text(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


@dataclass(frozen=True)
class CapabilityDefinition:
    name: str
    description: str
    positive_cues: tuple[str, ...]
    source_aliases: tuple[str, ...]
    classifier_label: str = ""


class CapabilityTaxonomy:
    def __init__(self, *, version: str, capabilities: list[CapabilityDefinition]) -> None:
        self.version = version
        self.capabilities = capabilities
        self.names = tuple(capability.name for capability in capabilities)
        aliases: dict[str, str] = {}
        for capability in capabilities:
            aliases[normalize_label_text(capability.name)] = capability.name
            if capability.classifier_label:
                aliases[normalize_label_text(capability.classifier_label)] = capability.name
            for alias in capability.source_aliases:
                aliases[normalize_label_text(alias)] = capability.name
        self.aliases = aliases

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CapabilityTaxonomy":
        raw_capabilities = value.get("capabilities")
        if isinstance(raw_capabilities, list):
            capabilities = [
                CapabilityDefinition(
                    name=str(name),
                    description="",
                    positive_cues=(),
                    source_aliases=(),
                    classifier_label=str(name).replace("_", " ").lower(),
                )
                for name in raw_capabilities
            ]
        elif isinstance(raw_capabilities, dict):
            capabilities = [
                CapabilityDefinition(
                    name=str(name),
                    description=str(definition.get("description", "")),
                    positive_cues=tuple(definition.get("positive_cues") or ()),
                    source_aliases=tuple(definition.get("source_aliases") or ()),
                    classifier_label=str(
                        definition.get("classifier_label")
                        or str(name).replace("_", " ").lower()
                    ),
                )
                for name, definition in raw_capabilities.items()
            ]
        else:
            raise ValueError("taxonomy capabilities must be a list or object")
        return cls(version=str(value.get("version", "capability-taxonomy-v1")), capabilities=capabilities)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CAPABILITY_TAXONOMY_PATH) -> "CapabilityTaxonomy":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def normalize_label(self, value: str) -> str | None:
        return self.aliases.get(normalize_label_text(value))

    def validate_labels(self, labels: list[str]) -> list[str]:
        valid = set(self.names)
        return sorted({label for label in labels if label in valid})

    def cues_by_capability(self) -> dict[str, tuple[str, ...]]:
        return {capability.name: capability.positive_cues for capability in self.capabilities}


class DomainTaxonomy(CapabilityTaxonomy):
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DomainTaxonomy":
        raw_domains = value.get("domains")
        if not isinstance(raw_domains, dict):
            raise ValueError("domain taxonomy must contain a domains object")
        capabilities = [
            CapabilityDefinition(
                name=str(name),
                description=str(definition.get("description", "")),
                positive_cues=tuple(definition.get("positive_cues") or ()),
                source_aliases=tuple(definition.get("source_aliases") or ()),
                classifier_label=str(
                    definition.get("classifier_label")
                    or str(name).replace("_", " ").lower()
                ),
            )
            for name, definition in raw_domains.items()
        ]
        return cls(version=str(value.get("version", "domain-taxonomy-v1")), capabilities=capabilities)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DOMAIN_TAXONOMY_PATH) -> "DomainTaxonomy":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
