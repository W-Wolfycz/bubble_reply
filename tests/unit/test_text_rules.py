from __future__ import annotations

import unittest

from bubble_reply.domain.text_rules import (
    escape_literal_newlines,
    prepare_text_for_planning,
    restore_placeholder_newlines,
    strip_bubble_reply_xml_tags,
)


class TextRuleTests(unittest.TestCase):
    def test_normalizes_crlf_and_cr(self) -> None:
        value = prepare_text_for_planning("a\r\nb\rc")
        self.assertEqual(value, "a\nb\nc")

    def test_added_literal_newlines_folded_unconditionally(self) -> None:
        # 输入合同:已 escape 的文本。出现字面量必为分段 LLM 新增 → 一律折叠。
        self.assertEqual(prepare_text_for_planning("a\\nb"), "a\nb")

    def test_placeholder_newlines_restore_per_config(self) -> None:
        # 占位符 = 主 LLM 原文的字面量，按配置还原为字面量或真实换行。
        value = (
            "a<bubble-cr/>b<bubble-lf/>c"
            "<bubble-crlf/>d<bubble-lfcr/>e"
        )
        self.assertEqual(
            prepare_text_for_planning(value),
            "a\\rb\\nc\\r\\nd\\n\\re",
        )
        self.assertEqual(
            prepare_text_for_planning(value, interpret_literals=True),
            "a\nb\nc\nd\ne",
        )

    def test_escape_literal_newlines_ordered(self) -> None:
        value = escape_literal_newlines("a\\r\\nb\\n\\rc\\nd\\re")
        self.assertEqual(
            value,
            "a<bubble-crlf/>b<bubble-lfcr/>c<bubble-lf/>d<bubble-cr/>e",
        )

    def test_literal_newline_variants_round_trip_when_disabled(self) -> None:
        original = "a\\rb\\nc\\r\\nd\\n\\re"
        escaped = escape_literal_newlines(original)
        self.assertEqual(prepare_text_for_planning(escaped), original)

    def test_escape_preserves_real_newlines(self) -> None:
        self.assertEqual(escape_literal_newlines("a\nb\\nc"), "a\nb<bubble-lf/>c")

    def test_restore_placeholder_newlines(self) -> None:
        self.assertEqual(
            restore_placeholder_newlines(
                "a<bubble-cr/>b<bubble-lf/>c<bubble-crlf/>d<bubble-lfcr/>e",
                False,
            ),
            "a\\rb\\nc\\r\\nd\\n\\re",
        )
        self.assertEqual(
            restore_placeholder_newlines(
                "a<bubble-cr/>b<bubble-lf/>c<bubble-crlf/>d<bubble-lfcr/>e",
                True,
            ),
            "a\nb\nc\nd\ne",
        )

    def test_extra_split_points(self) -> None:
        value = prepare_text_for_planning("a<cut>b", extra_split_points=("<cut>",))
        self.assertEqual(value, "a\nb")

    def test_emoticon_tags_are_removed(self) -> None:
        value = prepare_text_for_planning(
            "a<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>b"
        )
        self.assertEqual(value, "a(^_^)b")

    def test_bubble_reply_xml_fallback_removes_tags_only(self) -> None:
        value = strip_bubble_reply_xml_tags(
            "a<bubble-reply-x>yy</bubble-reply-x>b"
        )
        self.assertEqual(value, "ayyb")

    def test_bubble_reply_xml_fallback_removes_misspelled_tag(self) -> None:
        value = strip_bubble_reply_xml_tags(
            "///</bubble-reply-emat>被夸得不知道怎么办。"
        )
        self.assertEqual(value, "///被夸得不知道怎么办。")

    def test_bubble_reply_xml_fallback_handles_attributes_and_self_closing(self) -> None:
        value = strip_bubble_reply_xml_tags(
            'a<bubble-reply-x mode="demo">b<bubble-reply-y/>c'
        )
        self.assertEqual(value, "abc")

    def test_bubble_reply_xml_fallback_respects_namespace_boundary(self) -> None:
        value = strip_bubble_reply_xml_tags(
            "<bubble-replying>保留</bubble-replying>"
        )
        self.assertEqual(value, "<bubble-replying>保留</bubble-replying>")

    def test_markdown_fence_is_preserved(self) -> None:
        code = "```python\nprint('x')\n```"
        self.assertEqual(prepare_text_for_planning(code), code)


if __name__ == "__main__":
    unittest.main()
