from __future__ import annotations

import unittest

from bubble_reply.domain.validator import (
    validate_layout_only_change,
    validate_relaxed_candidate,
)


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

    def test_relaxed_mode_accepts_text_changes_and_normalizes_newlines(self) -> None:
        result = validate_relaxed_candidate("你好。", "你好！\r\n补充一句。")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "accepted_relaxed")
        self.assertEqual(result.normalized_candidate, "你好！\n补充一句。")

    def test_relaxed_mode_rejects_blank_output(self) -> None:
        result = validate_relaxed_candidate("你好。", " \n ")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "empty_candidate")

    def test_relaxed_mode_rejects_nonempty_output_for_empty_baseline(self) -> None:
        result = validate_relaxed_candidate("", "凭空生成")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "empty_baseline")

    def test_relaxed_mode_rejects_added_literal_newline(self) -> None:
        result = validate_relaxed_candidate("原文", "原\\n补充")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "added_literal_newline")

    def test_relaxed_mode_keeps_structural_guards(self) -> None:
        wrapper = validate_relaxed_candidate("abc", "<output>abd</output>")
        fence = validate_relaxed_candidate("abc", "```text\nabd\n```")
        emoticon = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "<bubble-reply-emoticon>(^_^)",
        )
        malformed_emoticon = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "<bubble-reply-emoticonx>(^_^)</bubble-reply-emoticonx>",
        )
        self.assertEqual(wrapper.reason_code, "unexpected_wrapper")
        self.assertEqual(fence.reason_code, "unexpected_wrapper")
        self.assertEqual(emoticon.reason_code, "unclosed_emoticon_tag")
        self.assertEqual(
            malformed_emoticon.reason_code,
            "unclosed_emoticon_tag",
        )

    def test_protocol_name_in_plain_text_is_not_a_malformed_tag(self) -> None:
        result = validate_relaxed_candidate(
            "请提到 bubble-reply-emoticon 这个协议名。",
            "请提到 bubble-reply-emoticon 这个协议名。",
        )
        self.assertTrue(result.accepted)

    def test_relaxed_mode_rejects_excessive_length_changes(self) -> None:
        expanded = validate_relaxed_candidate("a" * 100, "b" * 151)
        contracted = validate_relaxed_candidate("a" * 100, "b" * 49)
        self.assertEqual(expanded.reason_code, "excessive_length_change")
        self.assertEqual(contracted.reason_code, "excessive_length_change")


if __name__ == "__main__":
    unittest.main()
