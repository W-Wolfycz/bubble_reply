from __future__ import annotations

from collections.abc import Iterable

from ..config import ComponentPolicy, MediaPolicyConfig
from .models import ComponentToken, DeliveryPlan, PlannedSegment
from .text_rules import strip_tail_chars


ACTIVE_SAFE_KINDS = frozenset({"Plain", "Reply", "At", "Image", "Face"})
HEADER_ONLY_KINDS = frozenset({"Reply", "At"})


def _flush(buffer: list[ComponentToken], segments: list[list[ComponentToken]]) -> None:
    if buffer:
        segments.append(list(buffer))
        buffer.clear()


def _segment_tokens(
    tokens: Iterable[ComponentToken],
    media: MediaPolicyConfig,
    *,
    apply_media_policy: bool,
) -> list[list[ComponentToken]]:
    segments: list[list[ComponentToken]] = []
    buffer: list[ComponentToken] = []

    for token in tokens:
        if token.kind == "Plain":
            parts = str(token.payload).split("\n")
            for index, part in enumerate(parts):
                if part.strip():
                    buffer.append(ComponentToken("Plain", part))
                if index < len(parts) - 1:
                    _flush(buffer, segments)
            continue

        policy = media.for_kind(token.kind) if apply_media_policy else ComponentPolicy.EMBED
        if policy == ComponentPolicy.SEPARATE:
            _flush(buffer, segments)
            segments.append([token])
        elif policy == ComponentPolicy.FOLLOW_PREVIOUS:
            if buffer:
                buffer.append(token)
            elif segments:
                segments[-1].append(token)
            else:
                segments.append([token])
        elif policy == ComponentPolicy.FOLLOW_NEXT:
            _flush(buffer, segments)
            buffer.append(token)
        else:
            buffer.append(token)

    _flush(buffer, segments)
    return segments


def _clean_segment_boundaries(
    components: list[ComponentToken],
    chars: str,
    strip_whitespace: bool,
) -> list[ComponentToken]:
    cleaned = list(components)

    if strip_whitespace:
        index = 0
        while index < len(cleaned):
            component = cleaned[index]
            if component.kind != "Plain":
                index += 1
                continue
            value = str(component.payload).lstrip()
            if value:
                cleaned[index] = ComponentToken("Plain", value)
                break
            del cleaned[index]

    for index in range(len(cleaned) - 1, -1, -1):
        component = cleaned[index]
        if component.kind != "Plain":
            continue
        value = str(component.payload)
        if strip_whitespace:
            value = value.rstrip()
        value = strip_tail_chars(value, chars)
        if strip_whitespace:
            value = value.rstrip()
        if value.strip():
            cleaned[index] = ComponentToken("Plain", value)
            break
        del cleaned[index]

    return [
        component
        for component in cleaned
        if component.kind != "Plain" or str(component.payload).strip()
    ]


def _planned(
    raw_segments: Iterable[list[ComponentToken]],
    strip_segment_tail_chars: str,
    strip_segment_whitespace: bool,
) -> list[PlannedSegment]:
    result: list[PlannedSegment] = []
    for segment in raw_segments:
        cleaned = _clean_segment_boundaries(
            segment,
            strip_segment_tail_chars,
            strip_segment_whitespace,
        )
        if cleaned:
            result.append(PlannedSegment.from_components(cleaned))
    return result


def _with_auto_reply(
    segments: list[PlannedSegment],
    reply_token: ComponentToken | None,
    should_add_reply: bool,
) -> list[PlannedSegment]:
    copied = [PlannedSegment.from_components(segment.components) for segment in segments]
    if not copied or not should_add_reply or reply_token is None:
        return copied
    if any(
        component.kind == "Reply"
        for segment in copied
        for component in segment.components
    ):
        return copied
    first = copied[0]
    copied[0] = PlannedSegment.from_components(
        [reply_token, *first.components],
        auto_reply_component=reply_token,
    )
    return copied


def _is_header_only(segment: PlannedSegment) -> bool:
    return bool(segment.components) and all(
        component.kind in HEADER_ONLY_KINDS for component in segment.components
    )


def plan_delivery(
    tokens: list[ComponentToken],
    *,
    media: MediaPolicyConfig,
    strip_segment_tail_chars: str = "",
    strip_segment_whitespace: bool = True,
    should_add_reply: bool = False,
    reply_token: ComponentToken | None = None,
) -> DeliveryPlan:
    natural = _planned(
        _segment_tokens(tokens, media, apply_media_policy=False),
        strip_segment_tail_chars,
        strip_segment_whitespace,
    )
    policy_segments = _planned(
        _segment_tokens(tokens, media, apply_media_policy=True),
        strip_segment_tail_chars,
        strip_segment_whitespace,
    )

    if not policy_segments:
        return DeliveryPlan([], [], "respond_only", "empty")

    candidate_active = (
        policy_segments
        if _is_header_only(policy_segments[-1])
        else policy_segments[:-1]
    )
    if any(
        component.kind not in ACTIVE_SAFE_KINDS
        for segment in candidate_active
        for component in segment.components
    ):
        respond = _with_auto_reply(natural, reply_token, should_add_reply)
        return DeliveryPlan([], respond, "respond_only", "unsafe_active_component")

    selected = _with_auto_reply(policy_segments, reply_token, should_add_reply)
    if _is_header_only(selected[-1]):
        return DeliveryPlan(selected, [], "segmented", "active_header_only_tail")
    return DeliveryPlan(
        active_segments=selected[:-1],
        respond_segments=selected[-1:],
        mode="segmented",
        reason="safe_prefix",
    )


def apply_segment_limit(
    candidate_plan: DeliveryPlan,
    baseline_plan: DeliveryPlan,
    maximum_segments: int,
) -> DeliveryPlan:
    """限制最终气泡数:智能候选超限时先回退普通分段,再整条发送。"""
    if maximum_segments <= 0:
        return candidate_plan
    if len(candidate_plan.all_segments) <= maximum_segments:
        return candidate_plan
    if len(baseline_plan.all_segments) <= maximum_segments:
        return DeliveryPlan(
            active_segments=baseline_plan.active_segments,
            respond_segments=baseline_plan.respond_segments,
            mode=baseline_plan.mode,
            reason="segment_limit_fallback",
        )
    return DeliveryPlan(
        active_segments=[],
        respond_segments=baseline_plan.all_segments,
        mode="respond_only",
        reason="segment_limit",
    )
