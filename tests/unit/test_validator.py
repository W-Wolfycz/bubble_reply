from __future__ import annotations

import unittest

from bubble_reply.domain.validator import validate_layout_only_change


class ValidatorTests(unittest.TestCase):
    def test_accepts_real_newline_layout(self) -> None:
        result = validate_layout_only_change("早上好。今天有雨。", "早上好。\n今天有雨。")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_candidate, "早上好。\n今天有雨。")

    def test_accepts_added_literal_newline(self) -> None:
        result = validate_layout_only_change("abc", "a\\nbc")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_candidate, "a\nbc")

    def test_preserves_original_literal_escape(self) -> None:
        result = validate_layout_only_change("a\\nb", "a\\n\\nb")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_candidate, "a\\n\nb")

    def test_real_newlines_may_move(self) -> None:
        result = validate_layout_only_change("a\n\nb\nc", "ab\n\nc")
        self.assertTrue(result.accepted)

    def test_rejects_changed_punctuation(self) -> None:
        result = validate_layout_only_change("你好。", "你好！")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "changed_character")

    def test_rejects_added_space(self) -> None:
        result = validate_layout_only_change("ab", "a b")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "added_character")

    def test_rejects_removed_text(self) -> None:
        result = validate_layout_only_change("abc", "ac")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "removed_character")

    def test_rejects_wrapper(self) -> None:
        result = validate_layout_only_change("abc", "<output>abc</output>")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "unexpected_wrapper")

    def test_rejects_unclosed_emoticon_tag(self) -> None:
        result = validate_layout_only_change(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "<bubble-reply-emoticon>(^_^)",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "unclosed_emoticon_tag")


if __name__ == "__main__":
    unittest.main()
