"""Capability labeling for pilot dataset curation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from constellation.taxonomy import CapabilityTaxonomy, DomainTaxonomy

CAPABILITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "DEBUGGING": (
        "debug",
        "bug",
        "traceback",
        "stack trace",
        "exception",
        "failing test",
        "failure",
        "fix",
        "regression",
    ),
    "ERROR_RECOVERY": (
        "retry",
        "recover",
        "failed command",
        "error",
        "fallback",
        "try again",
        "diagnose",
    ),
    "TERMINAL_WORKFLOW": (
        "shell",
        "bash",
        "terminal",
        "command",
        "grep",
        "rg ",
        "pytest",
        "npm ",
        "cargo ",
    ),
    "TOOL_USE": (
        "<tool_call>",
        "tool call",
        "function call",
        "tool_response",
        "observation",
    ),
    "CODEBASE_NAVIGATION": (
        "repository",
        "repo",
        "codebase",
        "find file",
        "search",
        "inspect",
        "read file",
    ),
    "MULTI_FILE_EDITING": (
        "multi-file",
        "multiple files",
        "refactor",
        "across files",
        "apply patch",
        "unified diff",
    ),
    "TEST_WRITING": (
        "unit test",
        "tests",
        "test case",
        "coverage",
        "pytest",
        "jest",
        "vitest",
    ),
    "PLANNING": (
        "plan",
        "roadmap",
        "milestone",
        "decompose",
        "strategy",
        "steps",
    ),
    "RETRIEVAL_SEARCH": (
        "search",
        "retrieve",
        "lookup",
        "documentation",
        "reference",
        "query",
    ),
    "CODE_EDITING": (
        "edit",
        "patch",
        "diff",
        "implementation",
        "modify",
        "change code",
    ),
    "COMPOSITION": (
        "write",
        "compose",
        "draft",
        "essay",
        "story",
        "paragraph",
        "argument",
        "narrative",
    ),
    "REVISION": (
        "revise",
        "rewrite",
        "edit the text",
        "improve clarity",
        "tone",
        "style",
        "proofread",
    ),
    "STRUCTURED_REASONING": (
        "<think>",
        "<thinking>",
        "reasoning",
        "prove",
        "derive",
        "analyze",
    ),
}

SOURCE_CATEGORY_FIELDS = (
    "category",
    "subcategory",
    "task",
    "task_type",
    "source_category",
    "original_source",
)


@dataclass
class LabelEvidence:
    label: str
    score: float
    sources: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "sources": sorted(set(self.sources)),
            "cues": sorted(set(self.cues)),
        }


def source_category_values(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in SOURCE_CATEGORY_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for field in SOURCE_CATEGORY_FIELDS:
            value = metadata.get(field)
            if isinstance(value, str) and value.strip():
                values.append(value)
    return values


def label_capability_evidence(
    *,
    row: dict[str, Any],
    text: str,
    taxonomy: CapabilityTaxonomy | None = None,
) -> list[LabelEvidence]:
    taxonomy = taxonomy or CapabilityTaxonomy.load()
    cues_by_label = taxonomy.cues_by_capability()
    evidence: dict[str, LabelEvidence] = {}

    def add(label: str, score: float, source: str, cue: str) -> None:
        entry = evidence.setdefault(label, LabelEvidence(label=label, score=0.0))
        entry.score = min(1.0, max(entry.score, score))
        entry.sources.append(source)
        entry.cues.append(cue)

    haystack_parts = [
        text,
        *source_category_values(row),
    ]
    haystack = "\n".join(haystack_parts).lower()

    for label, cues in cues_by_label.items():
        if not cues:
            cues = CAPABILITY_KEYWORDS.get(label, ())
        matched = [cue for cue in cues if cue.lower() in haystack]
        if matched:
            add(label, min(0.95, 0.45 + 0.10 * len(matched)), "heuristic_keyword", matched[0])

    for raw_value in source_category_values(row):
        normalized = taxonomy.normalize_label(raw_value)
        if normalized is not None:
            add(normalized, 0.75, "source_category_alias", raw_value)

    if "TOOL_USE" not in evidence and any(marker in haystack for marker in ("<tool", "tool ")):
        add("TOOL_USE", 0.55, "heuristic_marker", "tool")

    return sorted(evidence.values(), key=lambda item: (-item.score, item.label))


def label_capabilities(*, row: dict[str, Any], text: str) -> list[str]:
    return sorted({evidence.label for evidence in label_capability_evidence(row=row, text=text)})


def label_domain_evidence(
    *,
    row: dict[str, Any],
    text: str,
    taxonomy: DomainTaxonomy | None = None,
) -> list[LabelEvidence]:
    taxonomy = taxonomy or DomainTaxonomy.load()
    evidence: dict[str, LabelEvidence] = {}

    def add(label: str, score: float, source: str, cue: str) -> None:
        entry = evidence.setdefault(label, LabelEvidence(label=label, score=0.0))
        entry.score = min(1.0, max(entry.score, score))
        entry.sources.append(source)
        entry.cues.append(cue)

    haystack = "\n".join([text, *source_category_values(row)]).lower()
    for label, cues in taxonomy.cues_by_capability().items():
        matched = [cue for cue in cues if cue.lower() in haystack]
        if matched:
            add(label, min(0.95, 0.45 + 0.10 * len(matched)), "heuristic_keyword", matched[0])

    for raw_value in source_category_values(row):
        normalized = taxonomy.normalize_label(raw_value)
        if normalized is not None:
            add(normalized, 0.75, "source_category_alias", raw_value)

    return sorted(evidence.values(), key=lambda item: (-item.score, item.label))


def label_domains(*, row: dict[str, Any], text: str) -> list[str]:
    return sorted({evidence.label for evidence in label_domain_evidence(row=row, text=text)})


def sample_type_for_row(*, row: dict[str, Any], text: str) -> str:
    haystack = " ".join(
        [
            str(row.get("category") or ""),
            str(row.get("subcategory") or ""),
            str(row.get("original_source") or ""),
            text[:2000],
        ]
    ).lower()
    if any(token in haystack for token in ("repo", "coding", "code", "python", "javascript")):
        return "coding"
    if any(token in haystack for token in ("tool", "terminal", "shell", "browser")):
        return "agent"
    return "reasoning"
