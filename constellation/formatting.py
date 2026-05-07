"""Canonical transcript formatting and SFT loss masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from constellation.schema import CanonicalSample, CanonicalTurn

IGNORE_INDEX = -100

TURN_LABELS = {
    ("system", "message"): "system",
    ("user", "message"): "user",
    ("tool", "observation"): "tool_observation",
    ("assistant", "reasoning"): "assistant_reasoning",
    ("assistant", "tool_call"): "assistant_tool_call",
    ("assistant", "final"): "assistant_final",
    ("assistant", "message"): "assistant",
}


@dataclass(frozen=True)
class TextSegment:
    text: str
    trainable: bool


def turn_label(turn: CanonicalTurn) -> str:
    return TURN_LABELS.get((turn.role, turn.type), f"{turn.role}_{turn.type}")


def transcript_segments(sample: CanonicalSample) -> list[TextSegment]:
    segments: list[TextSegment] = []
    for turn in sample.messages:
        header = f"<|{turn_label(turn)}|>\n"
        segments.append(TextSegment(header, trainable=False))
        segments.append(TextSegment(turn.content.strip() + "\n", trainable=turn.trainable))
    return segments


def render_transcript(sample: CanonicalSample, *, include_trainable: bool = True) -> str:
    parts: list[str] = []
    for segment in transcript_segments(sample):
        if include_trainable or not segment.trainable:
            parts.append(segment.text)
    return "".join(parts)


def render_eval_prompt(sample: CanonicalSample, *, mode: str = "initial") -> str:
    if mode not in {"initial", "after_first_observation"}:
        raise ValueError("eval prompt mode must be initial or after_first_observation")

    parts: list[str] = []
    seen_assistant = False
    seen_observation = False

    for turn in sample.messages:
        if mode == "initial" and turn.role == "assistant":
            break
        if mode == "after_first_observation" and seen_observation and turn.role == "assistant":
            break

        include = True
        if mode == "initial" and seen_assistant:
            include = False

        if include:
            parts.append(f"<|{turn_label(turn)}|>\n{turn.content.strip()}\n")

        if turn.role == "assistant":
            seen_assistant = True
        if turn.type == "observation":
            seen_observation = True

    parts.append("<|assistant_final|>\n")
    return "".join(parts)


def tokenize_with_loss_mask(
    tokenizer: Any,
    sample: CanonicalSample,
    *,
    max_length: int,
    add_eos: bool = True,
) -> dict[str, list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []

    for segment in transcript_segments(sample):
        token_ids = tokenizer(segment.text, add_special_tokens=False)["input_ids"]
        input_ids.extend(token_ids)
        if segment.trainable:
            labels.extend(token_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(token_ids))

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos_id is not None:
        input_ids.append(int(eos_id))
        labels.append(int(eos_id) if labels and labels[-1] != IGNORE_INDEX else IGNORE_INDEX)

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
