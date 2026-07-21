from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Protocol

from ..config import DelayConfig
from ..domain.models import DeliveryPlan, PlannedSegment


class DeliveryOutcome(str, Enum):
    RESPOND_ONLY = "respond_only"
    COMPLETED = "completed"
    AMBIGUOUS_FAILURE = "ambiguous_failure"


class DeliveryGateway(Protocol):
    def replace_result_segments(self, segments: list[PlannedSegment]) -> None: ...

    async def send_active(self, segment: PlannedSegment) -> None: ...

    def mark_partial_failure(self, reason: str) -> None: ...


class ConfiguredDelayPolicy:
    def __init__(self, config: DelayConfig) -> None:
        self._config = config

    def seconds_for(self, plain_text_length: int) -> float:
        del plain_text_length
        return self._config.seconds


def _temporary_reply_fallback(
    remaining: list[PlannedSegment],
    current: PlannedSegment,
) -> list[PlannedSegment]:
    reply = current.auto_reply_component
    if reply is None or not remaining:
        return remaining
    first = remaining[0]
    if any(component.kind == "Reply" for component in first.components):
        return remaining
    replacement = PlannedSegment.from_components([reply, *first.components])
    return [replacement, *remaining[1:]]


class DeliveryOrchestrator:
    def __init__(
        self,
        delay_policy: ConfiguredDelayPolicy,
        logger,
        *,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._delay_policy = delay_policy
        self._logger = logger
        self._sleep = sleep_func

    async def execute(
        self,
        plan: DeliveryPlan,
        gateway: DeliveryGateway,
        *,
        trace_id: str,
        log_prefix: str = "[BubbleReply]",
    ) -> DeliveryOutcome:
        if plan.mode == "respond_only" or not plan.active_segments:
            gateway.replace_result_segments(plan.respond_segments)
            return DeliveryOutcome.RESPOND_ONLY

        total = len(plan.active_segments) + len(plan.respond_segments)
        for index, current in enumerate(plan.active_segments):
            remaining = [
                *plan.active_segments[index + 1 :],
                *plan.respond_segments,
            ]
            gateway.replace_result_segments(
                _temporary_reply_fallback(remaining, current),
            )
            self._logger.info(
                "%s send segment=%s/%s text_len=%s components=%s trace=%s",
                log_prefix,
                index + 1,
                total,
                current.plain_text_length,
                ",".join(current.component_kinds),
                trace_id,
            )
            try:
                await gateway.send_active(current)
            except asyncio.CancelledError:
                gateway.mark_partial_failure("cancelled_ambiguous")
                raise
            except Exception as exc:
                gateway.mark_partial_failure("ambiguous_failure")
                self._logger.warning(
                    "%s send_failed segment=%s error_type=%s trace=%s",
                    log_prefix,
                    index + 1,
                    type(exc).__name__,
                    trace_id,
                )
                return DeliveryOutcome.AMBIGUOUS_FAILURE

            # There must be no await between a successful send and this normal fallback.
            gateway.replace_result_segments(remaining)
            if remaining:
                seconds = self._delay_policy.seconds_for(current.plain_text_length)
                if seconds > 0:
                    try:
                        await self._sleep(seconds)
                    except asyncio.CancelledError:
                        gateway.mark_partial_failure("delay_cancelled")
                        raise

        return DeliveryOutcome.COMPLETED
