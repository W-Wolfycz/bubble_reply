from __future__ import annotations


def expects_visible_streaming(
    *,
    action_type: str,
    streaming_enabled: bool | None,
    platform_supports_streaming: bool | None,
    unsupported_strategy: str | None,
) -> bool:
    """Predict whether AstrBot will deliver this request as a visible stream."""

    if action_type.lower() == "live":
        return True
    if streaming_enabled is None:
        return True
    if not streaming_enabled:
        return False
    return not (
        platform_supports_streaming is False
        and unsupported_strategy == "turn_off"
    )
