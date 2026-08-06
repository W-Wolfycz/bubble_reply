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
        # 新增字面量原样输出(不再折叠),由 prepare_text_for_planning 统一折叠。
        self.assertEqual(result.normalized_candidate, "a\\nbc")

    def test_preserves_original_literal_escape(self) -> None:
        result = validate_layout_only_change("a\\nb", "a\\n\\nb")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_candidate, "a\\n\\nb")

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
        result = validate_relaxed_candidate("你好。原句。", "你好！\r\n原句。")
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason_code, "accepted_relaxed")
        self.assertEqual(result.normalized_candidate, "你好！\n原句。")

    def test_relaxed_mode_rejects_blank_output(self) -> None:
        result = validate_relaxed_candidate("你好。", " \n ")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "empty_candidate")

    def test_relaxed_mode_rejects_nonempty_output_for_empty_baseline(self) -> None:
        result = validate_relaxed_candidate("", "凭空生成")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "empty_baseline")

    def test_relaxed_mode_accepts_added_literal_newline(self) -> None:
        # 字面量转义为占位符后,候选中的字面量必为分段 LLM 新增,
        # relaxed 不再拒绝,由 prepare 统一折叠为换行。
        result = validate_relaxed_candidate("原文", "原\\n文")
        self.assertTrue(result.accepted)
        self.assertEqual(result.normalized_candidate, "原\\n文")

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
        expanded = validate_relaxed_candidate("a" * 100, "b" * 111)
        contracted = validate_relaxed_candidate("a" * 100, "b" * 89)
        self.assertEqual(expanded.reason_code, "excessive_length_change")
        self.assertEqual(contracted.reason_code, "excessive_length_change")

    def test_relaxed_mode_accepts_ten_percent_length_change(self) -> None:
        expanded = validate_relaxed_candidate("a" * 100, "b" * 110)
        contracted = validate_relaxed_candidate("a" * 100, "b" * 90)
        self.assertTrue(expanded.accepted)
        self.assertTrue(contracted.accepted)

    def test_relaxed_mode_has_no_minimum_character_allowance(self) -> None:
        result = validate_relaxed_candidate("a", "bb")
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "excessive_length_change")

    def test_placeholder_removed_rejected_in_strict(self) -> None:
        result = validate_layout_only_change(
            "a<bubble-lf/>b",
            "ab",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "placeholder_modified")

    def test_placeholder_removed_rejected_in_relaxed(self) -> None:
        result = validate_relaxed_candidate(
            "a<bubble-lf/>b",
            "ab",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "placeholder_modified")

    def test_placeholder_added_rejected(self) -> None:
        result = validate_relaxed_candidate(
            "a<bubble-lf/>b",
            "a<bubble-lf/>b<bubble-lf/>",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "placeholder_modified")

    def test_placeholder_kept_and_newline_added_ok(self) -> None:
        result = validate_layout_only_change(
            "a<bubble-lf/>b",
            "a<bubble-lf/>\nb",
        )
        self.assertTrue(result.accepted)

    def test_placeholder_type_change_is_rejected(self) -> None:
        result = validate_relaxed_candidate(
            "a<bubble-cr/>b",
            "a<bubble-lf/>b",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "placeholder_modified")

    def test_relaxed_mode_allows_placeholder_movement(self) -> None:
        result = validate_relaxed_candidate(
            "a<bubble-lf/>b",
            "ab<bubble-lf/>",
        )
        self.assertTrue(result.accepted)

    def test_emoticon_tag_split_by_newline_rejected(self) -> None:
        result = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "<bubble-reply-emoticon>\n(^_^)\n</bubble-reply-emoticon>",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "emoticon_tag_split")

    def test_emoticon_tag_split_rejected_in_strict(self) -> None:
        result = validate_layout_only_change(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "<bubble-reply-emoticon>\n(^_^)</bubble-reply-emoticon>",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "emoticon_tag_split")

    def test_emoticon_open_tag_removed_rejected(self) -> None:
        # 单删开标签:balanced 先抓到(闭标签无配对)。
        result = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "(^_^)</bubble-reply-emoticon>",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "unclosed_emoticon_tag")

    def test_emoticon_tag_group_removed_rejected(self) -> None:
        # 两组中删一组:开/闭计数各 2→1,计数检查拦截。
        result = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon> 你好 "
            "<bubble-reply-emoticon>(>_<)</bubble-reply-emoticon>",
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon> 你好",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "emoticon_tag_count")

    def test_emoticon_tag_pair_removed_rejected(self) -> None:
        # 整体删除一组:开、闭分别计数变化,同样被拦截。
        result = validate_relaxed_candidate(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            "(^_^)",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "emoticon_tag_count")

    def test_emoticon_tag_added_rejected(self) -> None:
        result = validate_relaxed_candidate(
            "(^_^)",
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason_code, "emoticon_tag_count")


if __name__ == "__main__":
    unittest.main()
