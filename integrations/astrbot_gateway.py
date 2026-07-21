from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api.event import AstrMessageEvent, ResultContentType
from astrbot.api.star import Context

from ..domain.models import ComponentToken, PlannedSegment
from ..domain.streaming import expects_visible_streaming
from ..domain.text_rules import strip_emoticon_tags


@dataclass(frozen=True)
class ResultSnapshot:
    chain: tuple[Any, ...]
    content_type: ResultContentType | None
    is_llm_result: bool
    plain_text_length: int


class AstrBotGateway:
    """The only production module that mutates AstrBot event/result objects."""

    def __init__(self, context: Context, event: AstrMessageEvent) -> None:
        self._context = context
        self._event = event
        self._result = event.get_result()

    def read_result(self) -> ResultSnapshot | None:
        result = self._result
        if result is None:
            return None
        chain = tuple(result.chain)
        return ResultSnapshot(
            chain=chain,
            content_type=result.result_content_type,
            is_llm_result=result.is_llm_result(),
            plain_text_length=sum(
                len(component.text)
                for component in chain
                if isinstance(component, Comp.Plain)
            ),
        )

    def to_token(self, component: Any) -> ComponentToken:
        if isinstance(component, Comp.Plain):
            return ComponentToken("Plain", component.text)
        component_type = getattr(component, "type", None)
        kind = getattr(component_type, "value", None) or type(component).__name__
        return ComponentToken(str(kind), component)

    def make_reply_token(self) -> ComponentToken | None:
        message_obj = getattr(self._event, "message_obj", None)
        message_id = getattr(message_obj, "message_id", None)
        if message_id in (None, ""):
            return None
        sender_id = self._event.get_sender_id()
        return ComponentToken(
            "Reply",
            Comp.Reply(id=message_id, sender_id=sender_id),
        )

    @staticmethod
    def _materialize_segment(segment: PlannedSegment) -> list[Any]:
        result: list[Any] = []
        for token in segment.components:
            if token.kind == "Plain":
                result.append(Comp.Plain(str(token.payload)))
            else:
                result.append(token.payload)
        return result

    @classmethod
    def _materialize_segments(cls, segments: list[PlannedSegment]) -> list[Any]:
        chain: list[Any] = []
        for index, segment in enumerate(segments):
            materialized = cls._materialize_segment(segment)
            if index:
                # RespondStage removes whitespace-only Plain components.  Fold
                # the boundary into a non-empty adjacent Plain so fallback
                # segments cannot be concatenated after core cleanup.
                if (
                    chain
                    and materialized
                    and isinstance(chain[-1], Comp.Plain)
                    and isinstance(materialized[0], Comp.Plain)
                ):
                    chain[-1].text += "\n" + materialized[0].text
                    materialized = materialized[1:]
                    chain.extend(materialized)
                    continue
                next_plain = next(
                    (
                        component
                        for component in materialized
                        if isinstance(component, Comp.Plain)
                    ),
                    None,
                )
                if next_plain is not None:
                    next_plain.text = "\n" + next_plain.text
                else:
                    previous_plain = next(
                        (
                            component
                            for component in reversed(chain)
                            if isinstance(component, Comp.Plain)
                        ),
                        None,
                    )
                    if previous_plain is not None:
                        previous_plain.text += "\n"
            chain.extend(materialized)
        return chain

    def replace_result_segments(self, segments: list[PlannedSegment]) -> None:
        if self._result is not None:
            self._result.chain = self._materialize_segments(segments)

    async def send_active(self, segment: PlannedSegment) -> None:
        if self._result is None:
            return
        chain = self._result.derive(self._materialize_segment(segment))
        await self._event.send(chain)

    async def get_chat_provider_id(self) -> str:
        return await self._context.get_current_chat_provider_id(self.umo)

    async def llm_generate(self, provider_id: str, prompt: str) -> str:
        response = await self._context.llm_generate(
            chat_provider_id=provider_id,
            prompt=prompt,
        )
        return response.completion_text or ""

    def expects_visible_streaming(self) -> bool:
        override = self.get_extra("enable_streaming", None)
        settings = None
        try:
            config = self._context.get_config(self.umo)
            candidate = config.get("provider_settings", {})
            if isinstance(candidate, dict):
                settings = candidate
        except Exception:
            settings = None

        if override is not None:
            streaming_enabled: bool | None = bool(override)
        elif settings is not None and "streaming_response" in settings:
            streaming_enabled = bool(settings["streaming_response"])
        else:
            streaming_enabled = None

        platform_meta = getattr(self._event, "platform_meta", None)
        platform_support = getattr(
            platform_meta,
            "support_streaming_message",
            None,
        )
        if not isinstance(platform_support, bool):
            platform_support = None

        strategy = (
            settings.get("unsupported_streaming_strategy")
            if settings is not None
            else None
        )
        if not isinstance(strategy, str):
            strategy = None

        return expects_visible_streaming(
            action_type=str(self.get_extra("action_type", "") or ""),
            streaming_enabled=streaming_enabled,
            platform_supports_streaming=platform_support,
            unsupported_strategy=strategy,
        )

    def mark_partial_failure(self, reason: str) -> None:
        self._event.set_extra("_bubble_reply_partial_delivery", reason)

    def strip_internal_emoticon_tags(self) -> None:
        if self._result is None:
            return
        replacement: list[Any] = []
        for component in self._result.chain:
            if isinstance(component, Comp.Plain):
                replacement.append(Comp.Plain(strip_emoticon_tags(component.text)))
            else:
                replacement.append(component)
        self._result.chain = replacement

    @property
    def umo(self) -> str:
        return str(self._event.unified_msg_origin)

    @property
    def session_id(self) -> str:
        return str(self._event.session_id)

    @property
    def sender_id(self) -> str:
        return str(self._event.get_sender_id() or "")

    @property
    def group_id(self) -> str:
        return str(self._event.get_group_id() or "")

    @property
    def platform_name(self) -> str:
        return str(self._event.get_platform_name() or "")

    @property
    def self_id(self) -> str:
        return str(self._event.get_self_id() or "")

    @property
    def is_group(self) -> bool:
        message_type = self._event.get_message_type()
        return bool(self._event.get_group_id()) or (
            getattr(message_type, "name", "") == "GROUP_MESSAGE"
        )

    @property
    def message_id(self) -> str:
        message_obj = getattr(self._event, "message_obj", None)
        return str(getattr(message_obj, "message_id", "") or "")

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self._event.get_extra(key, default)

    def set_extra(self, key: str, value: Any) -> None:
        self._event.set_extra(key, value)
