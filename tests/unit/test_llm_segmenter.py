from __future__ import annotations

import logging
import unittest

from bubble_reply.config import LlmSegmenterConfig
from bubble_reply.services.llm_segmenter import LlmSegmenter


class LlmSegmenterPromptTests(unittest.TestCase):
    def test_text_placeholder_is_replaced(self) -> None:
        segmenter = LlmSegmenter(
            LlmSegmenterConfig(
                enabled=True,
                provider_id="provider_demo",
                runtime_rule="请分段：\n{text}",
                remove_before_split_regex=(),
                sanitize_llm_output_regex=(),
            ),
            logging.getLogger("bubble-reply-test"),
        )
        prompt = segmenter._prompt("第一句。第二句。")
        self.assertIn("请分段：\n第一句。第二句。", prompt)
        self.assertNotIn("{text}", prompt)


if __name__ == "__main__":
    unittest.main()
