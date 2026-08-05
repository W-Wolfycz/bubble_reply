from __future__ import annotations

import asyncio
import logging
import unittest

from bubble_reply.config import DelayConfig, DelayMode
from bubble_reply.domain.models import ComponentToken, DeliveryPlan, PlannedSegment
from bubble_reply.services.delivery_orchestrator import (
    ConfiguredDelayPolicy,
    DeliveryOrchestrator,
    DeliveryOutcome,
)


def segment(text: str, *, reply: bool = False) -> PlannedSegment:
    components = [ComponentToken("Plain", text)]
    reply_token = None
    if reply:
        reply_token = ComponentToken("Reply", object())
        components.insert(0, reply_token)
    return PlannedSegment.from_components(
        components,
        auto_reply_component=reply_token,
    )


def delay_config(seconds: float = 0.0) -> DelayConfig:
    return DelayConfig(seconds=seconds)


class FakeGateway:
    def __init__(self, failure_at: int | None = None, cancel_at: int | None = None):
        self.failure_at = failure_at
        self.cancel_at = cancel_at
        self.send_count = 0
        self.replacements: list[list[tuple[str, ...]]] = []
        self.sent: list[str] = []
        self.partial_reason: str | None = None

    def replace_result_segments(self, segments: list[PlannedSegment]) -> None:
        self.replacements.append([segment.component_kinds for segment in segments])

    async def send_active(self, current: PlannedSegment) -> None:
        self.send_count += 1
        self.sent.append(str(current.components[-1].payload))
        if self.cancel_at == self.send_count:
            raise asyncio.CancelledError
        if self.failure_at == self.send_count:
            raise RuntimeError("simulated")

    def mark_partial_failure(self, reason: str) -> None:
        self.partial_reason = reason


class DeliveryOrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def orchestrator(self, seconds: float = 0.0, sleep=asyncio.sleep):
        return DeliveryOrchestrator(
            ConfiguredDelayPolicy(delay_config(seconds)),
            logging.getLogger("bubble-reply-test"),
            sleep_func=sleep,
        )

    async def test_exception_never_restores_current_segment(self) -> None:
        plan = DeliveryPlan(
            [segment("a"), segment("b")],
            [segment("c")],
            "segmented",
            "test",
        )
        gateway = FakeGateway(failure_at=1)
        outcome = await self.orchestrator().execute(plan, gateway, trace_id="demo")
        self.assertEqual(outcome, DeliveryOutcome.AMBIGUOUS_FAILURE)
        self.assertEqual(gateway.sent, ["a"])
        self.assertEqual(gateway.replacements[0], [("Plain",), ("Plain",)])

    async def test_second_segment_failure_leaves_only_tail(self) -> None:
        plan = DeliveryPlan(
            [segment("a"), segment("b")],
            [segment("c")],
            "segmented",
            "test",
        )
        gateway = FakeGateway(failure_at=2)
        await self.orchestrator().execute(plan, gateway, trace_id="demo")
        self.assertEqual(gateway.sent, ["a", "b"])
        self.assertEqual(gateway.replacements[-1], [("Plain",)])

    async def test_send_cancellation_keeps_committed_fallback(self) -> None:
        plan = DeliveryPlan(
            [segment("a")],
            [segment("b")],
            "segmented",
            "test",
        )
        gateway = FakeGateway(cancel_at=1)
        with self.assertRaises(asyncio.CancelledError):
            await self.orchestrator().execute(plan, gateway, trace_id="demo")
        self.assertEqual(gateway.replacements[-1], [("Plain",)])
        self.assertEqual(gateway.partial_reason, "cancelled_ambiguous")

    async def test_reply_is_temporarily_added_to_failure_fallback(self) -> None:
        plan = DeliveryPlan(
            [segment("a", reply=True)],
            [segment("b")],
            "segmented",
            "test",
        )
        gateway = FakeGateway(failure_at=1)
        await self.orchestrator().execute(plan, gateway, trace_id="demo")
        self.assertEqual(gateway.replacements[-1], [("Reply", "Plain")])

    async def test_delay_cancellation_uses_normal_remaining_fallback(self) -> None:
        async def cancelled_sleep(_seconds: float) -> None:
            raise asyncio.CancelledError

        plan = DeliveryPlan(
            [segment("a")],
            [segment("b")],
            "segmented",
            "test",
        )
        gateway = FakeGateway()
        with self.assertRaises(asyncio.CancelledError):
            await self.orchestrator(1.0, cancelled_sleep).execute(
                plan,
                gateway,
                trace_id="demo",
            )
        self.assertEqual(gateway.replacements[-1], [("Plain",)])
        self.assertEqual(gateway.partial_reason, "delay_cancelled")

    async def test_per_character_delay_uses_sent_segment_length(self) -> None:
        sleeps: list[float] = []

        async def record_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        plan = DeliveryPlan(
            [segment("a" * 40)],
            [segment("b")],
            "segmented",
            "test",
        )
        gateway = FakeGateway()
        orchestrator = DeliveryOrchestrator(
            ConfiguredDelayPolicy(
                DelayConfig(
                    seconds=0.8,
                    mode=DelayMode.PER_CHARACTER,
                    seconds_per_character=0.025,
                    minimum_seconds=0.5,
                    maximum_seconds=2.5,
                )
            ),
            logging.getLogger("bubble-reply-test"),
            sleep_func=record_sleep,
        )
        await orchestrator.execute(plan, gateway, trace_id="demo")
        self.assertEqual(sleeps, [1.0])

    async def test_respond_only_performs_no_active_send(self) -> None:
        plan = DeliveryPlan([], [segment("a")], "respond_only", "test")
        gateway = FakeGateway()
        outcome = await self.orchestrator().execute(plan, gateway, trace_id="demo")
        self.assertEqual(outcome, DeliveryOutcome.RESPOND_ONLY)
        self.assertFalse(gateway.sent)


if __name__ == "__main__":
    unittest.main()
