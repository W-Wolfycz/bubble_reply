from __future__ import annotations

import unittest

from bubble_reply.domain.text_rules import (
    prepare_text_for_planning,
    split_nonempty_lines,
)


class TextRuleTests(unittest.TestCase):
    def test_normalizes_crlf_and_cr(self) -> None:
        value = prepare_text_for_planning("a\r\nb\rc")
        self.assertEqual(value, "a\nb\nc")

    def test_literal_newlines_are_opt_in(self) -> None:
        self.assertEqual(prepare_text_for_planning("a\\nb"), "a\\nb")
        self.assertEqual(
            prepare_text_for_planning("a\\nb", interpret_literals=True),
            "a\nb",
        )

    def test_extra_split_points(self) -> None:
        value = prepare_text_for_planning("a<cut>b", extra_split_points=("<cut>",))
        self.assertEqual(value, "a\nb")

    def test_emoticon_tags_are_removed(self) -> None:
        value = prepare_text_for_planning(
            "a<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>b"
        )
        self.assertEqual(value, "a(^_^)b")

    def test_markdown_fence_is_preserved(self) -> None:
        code = "```python\nprint('x')\n```"
        self.assertEqual(prepare_text_for_planning(code), code)

    def test_continuous_newlines_do_not_create_empty_messages(self) -> None:
        self.assertEqual(split_nonempty_lines("a\n\n\n b\n"), ["a", " b"])


if __name__ == "__main__":
    unittest.main()
