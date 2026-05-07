"""Canonical schema objects for specialist distillation data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ROLES = {"system", "user", "assistant", "tool"}
TURN_TYPES = {"message", "reasoning", "tool_call", "observation", "final"}
SAMPLE_TYPES = {"reasoning", "agent", "coding"}


class SchemaError(ValueError):
    """Raised when a canonical sample is malformed."""


@dataclass
class CanonicalTurn:
    role: str
    type: str
    content: str
    trainable: bool = False
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SchemaError(f"invalid role: {self.role!r}")
        if self.type not in TURN_TYPES:
            raise SchemaError(f"invalid turn type: {self.type!r}")
        if not isinstance(self.content, str):
            raise SchemaError("turn content must be a string")
        if self.trainable and self.role != "assistant":
            raise SchemaError("only assistant turns may be trainable")
        if self.role == "tool" and self.type != "observation":
            raise SchemaError("tool role must use observation turn type")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role,
            "type": self.type,
            "content": self.content,
            "trainable": self.trainable,
        }
        if self.name is not None:
            result["name"] = self.name
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalTurn":
        return cls(
            role=value["role"],
            type=value["type"],
            content=value.get("content", ""),
            trainable=bool(value.get("trainable", False)),
            name=value.get("name"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class CanonicalSample:
    id: str
    source_dataset: str
    sample_type: str
    messages: list[CanonicalTurn]
    capabilities: list[str] = field(default_factory=list)
    success: bool | None = None
    quality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("sample id is required")
        if self.sample_type not in SAMPLE_TYPES:
            raise SchemaError(f"invalid sample type: {self.sample_type!r}")
        if not self.messages:
            raise SchemaError("sample must contain at least one message")
        if not 0.0 <= self.quality_score <= 1.0:
            raise SchemaError("quality_score must be between 0.0 and 1.0")
        if self.success is not None and not isinstance(self.success, bool):
            raise SchemaError("success must be a bool or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_dataset": self.source_dataset,
            "sample_type": self.sample_type,
            "messages": [turn.to_dict() for turn in self.messages],
            "capabilities": self.capabilities,
            "success": self.success,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalSample":
        return cls(
            id=value["id"],
            source_dataset=value["source_dataset"],
            sample_type=value["sample_type"],
            messages=[CanonicalTurn.from_dict(item) for item in value.get("messages", [])],
            capabilities=list(value.get("capabilities") or []),
            success=value.get("success"),
            quality_score=float(value.get("quality_score", 0.0)),
            metadata=dict(value.get("metadata") or {}),
        )

    def trainable_turn_count(self) -> int:
        return sum(1 for turn in self.messages if turn.trainable)

    def joined_text(self) -> str:
        return "\n".join(turn.content for turn in self.messages)
