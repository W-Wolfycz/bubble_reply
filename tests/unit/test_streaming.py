from __future__ import annotations

import unittest

from bubble_reply.domain.streaming import expects_visible_streaming


class StreamingDecisionTests(unittest.TestCase):
    def test_live_mode_is_always_streaming(self) -> None:
        self.assertTrue(
            expects_visible_streaming(
                action_type="live",
                streaming_enabled=False,
                platform_supports_streaming=False,
                unsupported_strategy="turn_off",
            )
        )

    def test_disabled_streaming_is_not_visible(self) -> None:
        self.assertFalse(
            expects_visible_streaming(
                action_type="",
                streaming_enabled=False,
                platform_supports_streaming=True,
                unsupported_strategy="realtime_segmenting",
            )
        )

    def test_supported_platform_keeps_streaming(self) -> None:
        self.assertTrue(
            expects_visible_streaming(
                action_type="",
                streaming_enabled=True,
                platform_supports_streaming=True,
                unsupported_strategy="turn_off",
            )
        )

    def test_unsupported_platform_can_fall_back_to_non_streaming(self) -> None:
        self.assertFalse(
            expects_visible_streaming(
                action_type="",
                streaming_enabled=True,
                platform_supports_streaming=False,
                unsupported_strategy="turn_off",
            )
        )

    def test_realtime_segmenting_remains_streaming_on_unsupported_platform(self) -> None:
        self.assertTrue(
            expects_visible_streaming(
                action_type="",
                streaming_enabled=True,
                platform_supports_streaming=False,
                unsupported_strategy="realtime_segmenting",
            )
        )

    def test_unknown_configuration_is_treated_as_streaming(self) -> None:
        self.assertTrue(
            expects_visible_streaming(
                action_type="",
                streaming_enabled=None,
                platform_supports_streaming=None,
                unsupported_strategy=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
