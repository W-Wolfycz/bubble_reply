from __future__ import annotations

import logging
import unittest

from bubble_reply.config import LlmSegmenterConfig
from bubble_reply.services.llm_segmenter import LlmSegmenter, SegmentContext


class _Gateway:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls = 0

    async def get_chat_provider_id(self) -> str:
        return "provider_demo"

    async def llm_generate(self, provider_id: str, prompt: str) -> str:
        del provider_id, prompt
        self.calls += 1
        return self.output


def segmenter_config(*, allow_text_changes: bool = False) -> LlmSegmenterConfig:
    return LlmSegmenterConfig(
        enabled=True,
        provider_id="provider_demo",
        runtime_rule="请分段：\n{text}",
        remove_before_split_regex=(),
        sanitize_llm_output_regex=(),
        allow_text_changes=allow_text_changes,
    )


class LlmSegmenterPromptTests(unittest.TestCase):
    def test_text_placeholder_is_replaced(self) -> None:
        segmenter = LlmSegmenter(
            segmenter_config(),
            logging.getLogger("bubble-reply-test"),
        )
        prompt = segmenter._prompt("第一句。第二句。")
        self.assertIn("请分段：\n第一句。第二句。", prompt)
        self.assertNotIn("{text}", prompt)

    def test_relaxed_prompt_requires_real_newlines(self) -> None:
        segmenter = LlmSegmenter(
            segmenter_config(allow_text_changes=True),
            logging.getLogger("bubble-reply-test"),
        )
        prompt = segmenter._prompt("第一句。第二句。")
        self.assertIn("不要输出字面量 \\n", prompt)

    def test_prompt_declares_protocol_markers(self) -> None:
        segmenter = LlmSegmenter(
            segmenter_config(),
            logging.getLogger("bubble-reply-test"),
        )
        prompt = segmenter._prompt("第一句。第二句。")
        self.assertIn("内部颜文字保护标记", prompt)
        self.assertIn("换行只能添加在整体之前或之后", prompt)
        self.assertIn("<bubble-cr/>", prompt)
        self.assertIn("<bubble-lf/>", prompt)
        self.assertIn("<bubble-crlf/>", prompt)
        self.assertIn("<bubble-lfcr/>", prompt)
        self.assertIn("是内部转义占位符", prompt)
        self.assertIn("不得删除、修改、替换或移动", prompt)


class LlmSegmenterValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_mode_rejects_text_changes(self) -> None:
        segmenter = LlmSegmenter(
            segmenter_config(),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "你好。",
            SegmentContext(_Gateway("你好！"), "trace_demo"),
        )
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.layout_text, "你好。")
        self.assertEqual(candidate.rejection_reason, "changed_character")

    async def test_relaxed_mode_accepts_text_changes(self) -> None:
        segmenter = LlmSegmenter(
            segmenter_config(allow_text_changes=True),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "你好。原句。",
            SegmentContext(_Gateway("你好！\n原句。"), "trace_demo"),
        )
        self.assertTrue(candidate.accepted)
        self.assertEqual(candidate.layout_text, "你好！\n原句。")
        self.assertIsNone(candidate.rejection_reason)

    async def test_relaxed_mode_accepts_added_literal_newline(self) -> None:
        # 字面量转义后,候选中的字面量必为分段 LLM 新增,relaxed 不再拒绝。
        gateway = _Gateway("你好。\\n补充一句。")
        segmenter = LlmSegmenter(
            segmenter_config(allow_text_changes=True),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "你好。补充一句。",
            SegmentContext(gateway, "trace_demo"),
        )
        self.assertTrue(candidate.accepted)
        self.assertEqual(candidate.layout_text, "你好。\\n补充一句。")

    async def test_placeholder_passes_through_validation(self) -> None:
        gateway = _Gateway("你好。<bubble-lf/>补充一句。")
        segmenter = LlmSegmenter(
            segmenter_config(),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "你好。\\n补充一句。",
            SegmentContext(gateway, "trace_demo"),
        )
        self.assertTrue(candidate.accepted)
        self.assertEqual(candidate.layout_text, "你好。<bubble-lf/>补充一句。")

    async def test_placeholder_removal_rejected(self) -> None:
        gateway = _Gateway("你好。补充一句。")
        segmenter = LlmSegmenter(
            segmenter_config(),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "你好。\\n补充一句。",
            SegmentContext(gateway, "trace_demo"),
        )
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.rejection_reason, "placeholder_modified")
        self.assertEqual(candidate.layout_text, "你好。<bubble-lf/>补充一句。")

    async def test_emoticon_tag_split_rejected_by_segmenter(self) -> None:
        gateway = _Gateway(
            "<bubble-reply-emoticon>\n(^_^)</bubble-reply-emoticon>"
        )
        segmenter = LlmSegmenter(
            segmenter_config(allow_text_changes=True),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "<bubble-reply-emoticon>(^_^)</bubble-reply-emoticon>",
            SegmentContext(gateway, "trace_demo"),
        )
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.rejection_reason, "emoticon_tag_split")

    async def test_empty_baseline_does_not_call_provider(self) -> None:
        gateway = _Gateway("凭空生成")
        segmenter = LlmSegmenter(
            segmenter_config(allow_text_changes=True),
            logging.getLogger("bubble-reply-test"),
        )
        candidate = await segmenter.segment(
            "   ",
            SegmentContext(gateway, "trace_demo"),
        )
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.rejection_reason, "empty_baseline")
        self.assertEqual(candidate.layout_text, "   ")
        self.assertEqual(gateway.calls, 0)


if __name__ == "__main__":
    unittest.main()
