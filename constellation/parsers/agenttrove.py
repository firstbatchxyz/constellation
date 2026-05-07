"""Parser for AgentTrove-style rows."""

from __future__ import annotations

from typing import Any

from constellation.labeling import label_capabilities, sample_type_for_row
from constellation.parsers.common import (
    normalize_role,
    parse_success,
    split_assistant_content,
    stable_id,
)
from constellation.schema import CanonicalSample, CanonicalTurn


def parse_agenttrove_row(
    row: dict[str, Any],
    *,
    source_dataset: str = "open-thoughts/AgentTrove",
) -> CanonicalSample:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("AgentTrove row is missing messages list")

    turns: list[CanonicalTurn] = []
    for raw_message in raw_messages:
        role = normalize_role(raw_message.get("role") or raw_message.get("from"))
        content = str(raw_message.get("content") or raw_message.get("value") or "").strip()
        if not content:
            continue

        if role == "assistant":
            turns.extend(split_assistant_content(content))
        elif role == "tool":
            turns.append(CanonicalTurn(role="tool", type="observation", content=content))
        else:
            turns.append(CanonicalTurn(role=role, type="message", content=content))

    sample_id = str(
        row.get("trajectory_id")
        or row.get("task_id")
        or row.get("id")
        or stable_id(source_dataset, row.get("original_source"), raw_messages)
    )

    metadata = {
        "original_id": row.get("task_id") or row.get("id"),
        "original_source": row.get("original_source"),
        "teacher_model": row.get("original_teacher") or row.get("model"),
        "reward": row.get("reward"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    text = "\n".join(turn.content for turn in turns)
    capabilities = label_capabilities(row=row, text=text)

    return CanonicalSample(
        id=sample_id,
        source_dataset=source_dataset,
        sample_type=sample_type_for_row(row=row, text=text),
        messages=turns,
        capabilities=capabilities,
        success=parse_success(row.get("reward") or row.get("success")),
        metadata=metadata,
    )
