"""Parser for Hermes agent reasoning trace rows."""

from __future__ import annotations

from typing import Any

from constellation.labeling import label_capabilities
from constellation.parsers.common import (
    normalize_role,
    split_assistant_content,
    stable_id,
)
from constellation.schema import CanonicalSample, CanonicalTurn


def parse_hermes_row(
    row: dict[str, Any],
    *,
    source_dataset: str = "lambda/hermes-agent-reasoning-traces",
) -> CanonicalSample:
    raw_messages = row.get("conversations") or row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("Hermes row is missing conversations/messages list")

    turns: list[CanonicalTurn] = []
    for raw_message in raw_messages:
        role = normalize_role(raw_message.get("from") or raw_message.get("role"))
        content = str(raw_message.get("value") or raw_message.get("content") or "").strip()
        if not content:
            continue

        if role == "assistant":
            turns.extend(split_assistant_content(content))
        elif role == "tool":
            turns.append(CanonicalTurn(role="tool", type="observation", content=content))
        else:
            turns.append(CanonicalTurn(role=role, type="message", content=content))

    sample_id = str(row.get("id") or stable_id(source_dataset, raw_messages))
    metadata = {
        "original_id": row.get("id"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "task": row.get("task"),
        "tools_present": bool(row.get("tools")),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    text = "\n".join(turn.content for turn in turns)

    return CanonicalSample(
        id=sample_id,
        source_dataset=source_dataset,
        sample_type="agent",
        messages=turns,
        capabilities=label_capabilities(row=row, text=text),
        success=None,
        metadata=metadata,
    )
