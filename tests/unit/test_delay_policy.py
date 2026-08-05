from __future__ import annotations

import unittest

from bubble_reply.config import DelayConfig, DelayMode
from bubble_reply.services.delivery_orchestrator import ConfiguredDelayPolicy


class DelayPolicyTests(unittest.TestCase):
    def test_fixed_interval_ignores_text_length(self) -> None:
        policy = ConfiguredDelayPolicy(DelayConfig(seconds=0.8))
        self.assertEqual(policy.seconds_for(0), 0.8)
        self.assertEqual(policy.seconds_for(1000), 0.8)

    def test_random_jitter_applies_to_fixed_interval(self) -> None:
        policy = ConfiguredDelayPolicy(
            DelayConfig(seconds=0.8, jitter_seconds=0.2),
            random_uniform=lambda lower, upper: upper,
        )
        self.assertAlmostEqual(policy.seconds_for(20), 1.0)

    def test_fixed_interval_cannot_become_negative(self) -> None:
        policy = ConfiguredDelayPolicy(
            DelayConfig(seconds=0.1, jitter_seconds=0.2),
            random_uniform=lambda lower, upper: lower,
        )
        self.assertEqual(policy.seconds_for(20), 0)

    def test_per_character_interval_uses_minimum_and_maximum(self) -> None:
        policy = ConfiguredDelayPolicy(
            DelayConfig(
                seconds=0.8,
                mode=DelayMode.PER_CHARACTER,
                seconds_per_character=0.025,
                minimum_seconds=0.5,
                maximum_seconds=2.5,
            )
        )
        self.assertEqual(policy.seconds_for(0), 0.5)
        self.assertEqual(policy.seconds_for(10), 0.5)
        self.assertEqual(policy.seconds_for(40), 1.0)
        self.assertEqual(policy.seconds_for(200), 2.5)

    def test_random_jitter_applies_before_per_character_clamp(self) -> None:
        policy = ConfiguredDelayPolicy(
            DelayConfig(
                seconds=0.8,
                mode=DelayMode.PER_CHARACTER,
                seconds_per_character=0.025,
                minimum_seconds=0.5,
                maximum_seconds=1.1,
                jitter_seconds=0.2,
            ),
            random_uniform=lambda lower, upper: upper,
        )
        self.assertEqual(policy.seconds_for(40), 1.1)


if __name__ == "__main__":
    unittest.main()
