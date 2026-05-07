"""Parser for AgentTrove-style rows."""

from __future__ import annotations

from typing import Any

from constellation.labeling import label_capabilities, label_domains, sample_type_for_row
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
    raw_messages = row.get("messages") or row.get("conversations")
    if not isinstance(raw_messages, list):
        raise ValueError("AgentTrove row is missing messages/conversations list")

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
        or row.get("path")
        or row.get("trial_name")
        or row.get("episode")
        or stable_id(source_dataset, row.get("original_source"), raw_messages)
    )

    metadata = {
        "original_id": row.get("task_id") or row.get("id") or row.get("path"),
        "original_source": row.get("original_source"),
        "teacher_model": row.get("original_teacher") or row.get("model"),
        "reward": row.get("reward"),
        "result": row.get("result"),
        "task": row.get("task"),
        "episode": row.get("episode"),
        "run_id": row.get("run_id"),
        "trial_name": row.get("trial_name"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    text = "\n".join(turn.content for turn in turns)
    capabilities = label_capabilities(row=row, text=text)
    domains = label_domains(row=row, text=text)

    return CanonicalSample(
        id=sample_id,
        source_dataset=source_dataset,
        sample_type=sample_type_for_row(row=row, text=text),
        messages=turns,
        capabilities=capabilities,
        domains=domains,
        success=parse_success(
            row["reward"]
            if "reward" in row
            else row.get("success", row.get("result", row.get("task_binary")))
        ),
        metadata=metadata,
    )
