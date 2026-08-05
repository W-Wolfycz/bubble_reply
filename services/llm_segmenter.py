from __future__ import annotations

import re
import secrets
from dataclasses import dataclass
from typing import Protocol

from ..config import LlmSegmenterConfig
from ..domain.models import TextCandidate
from ..domain.text_rules import apply_regex_removals
from ..domain.validator import validate_layout_only_change, validate_relaxed_candidate


class SegmentGateway(Protocol):
    async def get_chat_provider_id(self) -> str: ...

    async def llm_generate(self, provider_id: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class SegmentContext:
    gateway: SegmentGateway
    trace_id: str
    log_prefix: str = "[BubbleReply]"


class LlmSegmenter:
    def __init__(self, config: LlmSegmenterConfig, logger) -> None:
        self._config = config
        self._logger = logger
        self._before = tuple(
            re.compile(pattern, re.MULTILINE) for pattern in config.remove_before_split_regex
        )
        self._after = tuple(
            re.compile(pattern, re.MULTILINE) for pattern in config.sanitize_llm_output_regex
        )

    def _prompt(self, baseline: str) -> str:
        nonce = secrets.token_hex(8)
        runtime_rule = self._config.runtime_rule.replace("{text}", baseline)
        newline_constraint = (
            "只能新增、删除或移动真实换行；不要输出字面量 \\n、\\r 或 \\r\\n 作为分段标记。"
            if self._config.allow_text_changes
            else "只能新增、删除或移动真实换行，或新增字面量\\n、\\r、\\r\\n。"
        )
        return (
            "<role>你是文本布局器，只调整分段，不修改正文。</role>\n"
            f"<constraints>{newline_constraint}"
            "禁止修改文字、空格、标点、标签及其内部内容；禁止解释、总结或添加包装。</constraints>\n"
            f'<task nonce="{nonce}">\n{runtime_rule}\n</task nonce="{nonce}">'
        )

    async def segment(self, text: str, context: SegmentContext) -> TextCandidate:
        baseline = apply_regex_removals(text, self._before)
        if not baseline.strip():
            return TextCandidate(
                original=text,
                cleaned=baseline,
                llm_output=None,
                accepted=False,
                rejection_reason="empty_baseline",
                layout_text=baseline,
            )
        if not self._config.enabled:
            return TextCandidate(text, baseline, None, False, None, baseline)

        try:
            provider_id = self._config.provider_id or await context.gateway.get_chat_provider_id()
            if not provider_id:
                raise RuntimeError("provider unavailable")
            output = await context.gateway.llm_generate(provider_id, self._prompt(baseline))
        except Exception as exc:
            self._logger.warning(
                "%s llm_candidate accepted=false reason=provider_error error_type=%s trace=%s",
                context.log_prefix,
                type(exc).__name__,
                context.trace_id,
            )
            return TextCandidate(
                text,
                baseline,
                None,
                False,
                "provider_error",
                baseline,
            )

        sanitized = apply_regex_removals(output or "", self._after)
        validation_mode = (
            "relaxed" if self._config.allow_text_changes else "strict"
        )
        validation = (
            validate_relaxed_candidate(baseline, sanitized)
            if self._config.allow_text_changes
            else validate_layout_only_change(baseline, sanitized)
        )
        self._logger.info(
            "%s llm_candidate accepted=%s validation=%s reason=%s position=%s trace=%s",
            context.log_prefix,
            str(validation.accepted).lower(),
            validation_mode,
            validation.reason_code,
            validation.first_mismatch_position,
            context.trace_id,
        )
        return TextCandidate(
            original=text,
            cleaned=baseline,
            llm_output=sanitized,
            accepted=validation.accepted,
            rejection_reason=None if validation.accepted else validation.reason_code,
            layout_text=(
                validation.normalized_candidate if validation.accepted else baseline
            ),
        )
