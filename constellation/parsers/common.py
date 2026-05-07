"""Shared parser helpers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from constellation.schema import CanonicalTurn

ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "observation": "tool",
}

TAG_TO_TYPE = {
    "think": "reasoning",
    "thinking": "reasoning",
    "reasoning": "reasoning",
    "tool_call": "tool_call",
    "tool_response": "observation",
}

TAG_PATTERN = re.compile(
    r"<(?P<tag>think|thinking|reasoning|tool_call|tool_response)>\s*"
    r"(?P<body>.*?)\s*</(?P=tag)>",
    flags=re.IGNORECASE | re.DOTALL,
)


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def normalize_role(raw_role: str | None) -> str:
    role = (raw_role or "").strip().lower()
    if role not in ROLE_MAP:
        raise ValueError(f"unknown role: {raw_role!r}")
    return ROLE_MAP[role]


def split_assistant_content(content: str) -> list[CanonicalTurn]:
    """Split tagged assistant content while preserving order."""
    turns: list[CanonicalTurn] = []
    cursor = 0

    for match in TAG_PATTERN.finditer(content):
        prefix = content[cursor : match.start()].strip()
        if prefix:
            turns.append(
                CanonicalTurn(role="assistant", type="final", content=prefix, trainable=True)
            )

        tag = match.group("tag").lower()
        body = match.group("body").strip()
        turn_type = TAG_TO_TYPE[tag]
        if body:
            if turn_type == "observation":
                turns.append(
                    CanonicalTurn(role="tool", type="observation", content=body, trainable=False)
                )
            else:
                turns.append(
                    CanonicalTurn(
                        role="assistant",
                        type=turn_type,
                        content=body,
                        trainable=True,
                    )
                )
        cursor = match.end()

    suffix = content[cursor:].strip()
    if suffix:
        turns.append(CanonicalTurn(role="assistant", type="final", content=suffix, trainable=True))

    if not turns and content.strip():
        turns.append(
            CanonicalTurn(role="assistant", type="final", content=content.strip(), trainable=True)
        )

    return turns


def parse_success(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "success", "passed", "pass"}:
            return True
        if lowered in {"0", "false", "failure", "failed", "fail"}:
            return False
    return None
