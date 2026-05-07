"""Heuristic capability labeling for pilot curation."""

from __future__ import annotations

from typing import Any

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
    "STRUCTURED_REASONING": (
        "<think>",
        "<thinking>",
        "reasoning",
        "prove",
        "derive",
        "analyze",
    ),
}


def label_capabilities(*, row: dict[str, Any], text: str) -> list[str]:
    haystack_parts = [
        text,
        str(row.get("category") or ""),
        str(row.get("subcategory") or ""),
        str(row.get("task") or ""),
        str(row.get("original_source") or ""),
    ]
    haystack = "\n".join(haystack_parts).lower()

    labels = [
        capability
        for capability, keywords in CAPABILITY_KEYWORDS.items()
        if any(keyword.lower() in haystack for keyword in keywords)
    ]
    if "TOOL_USE" not in labels and any(marker in haystack for marker in ("<tool", "tool ")):
        labels.append("TOOL_USE")
    return sorted(set(labels))


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
