from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ComponentToken:
    """AstrBot-independent representation of a message component."""

    kind: str
    payload: Any


@dataclass(frozen=True)
class TextCandidate:
    original: str
    cleaned: str
    llm_output: str | None
    accepted: bool
    rejection_reason: str | None
    layout_text: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason_code: str
    first_mismatch_position: int | None
    baseline_non_layout_chars: int
    candidate_non_layout_chars: int
    normalized_candidate: str | None = None


@dataclass
class PlannedSegment:
    components: list[ComponentToken]
    plain_text_length: int = 0
    component_kinds: tuple[str, ...] = field(default_factory=tuple)
    auto_reply_component: ComponentToken | None = None

    @classmethod
    def from_components(
        cls,
        components: list[ComponentToken],
        *,
        auto_reply_component: ComponentToken | None = None,
    ) -> "PlannedSegment":
        copied = list(components)
        return cls(
            components=copied,
            plain_text_length=sum(
                len(str(component.payload))
                for component in copied
                if component.kind == "Plain"
            ),
            component_kinds=tuple(component.kind for component in copied),
            auto_reply_component=auto_reply_component,
        )


@dataclass
class DeliveryPlan:
    active_segments: list[PlannedSegment]
    respond_segments: list[PlannedSegment]
    mode: Literal["segmented", "respond_only"]
    reason: str

    @property
    def all_segments(self) -> list[PlannedSegment]:
        return [*self.active_segments, *self.respond_segments]

