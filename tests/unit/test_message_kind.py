from __future__ import annotations

import unittest

from bubble_reply.domain.message_kind import is_noise_message


class IsNoiseMessageTests(unittest.TestCase):
    def test_other_message_is_noise(self) -> None:
        self.assertTrue(
            is_noise_message(
                message_type_value="OtherMessage",
                post_type="",
                has_poke=False,
            )
        )

    def test_notice_and_request_are_noise(self) -> None:
        for post_type in ("notice", "request"):
            self.assertTrue(
                is_noise_message(
                    message_type_value="GroupMessage",
                    post_type=post_type,
                    has_poke=False,
                )
            )

    def test_poke_component_is_noise(self) -> None:
        self.assertTrue(
            is_noise_message(
                message_type_value="GroupMessage",
                post_type="message",
                has_poke=True,
            )
        )

    def test_plain_chat_message_is_not_noise(self) -> None:
        self.assertFalse(
            is_noise_message(
                message_type_value="GroupMessage",
                post_type="message",
                has_poke=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
