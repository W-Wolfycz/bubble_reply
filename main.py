from __future__ import annotations

import secrets
from typing import Any

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, ResultContentType, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star

from .config import BubbleReplyConfig, QuoteMode, SplitScope, load_runtime_config
from .domain.models import ComponentToken, PlannedSegment
from .domain.planner import apply_segment_limit, plan_delivery
from .domain.text_rules import prepare_text_for_planning
from .integrations.astrbot_gateway import AstrBotGateway, ResultSnapshot
from .services.delivery_orchestrator import (
    ConfiguredDelayPolicy,
    DeliveryOrchestrator,
)
from .services.llm_segmenter import LlmSegmenter, SegmentContext
from .services.smart_reply_tracker import RequestMark, SmartReplyTracker


DECORATE_PRIORITY = -100_000
INCOMING_PRIORITY = 100_000
REQUEST_MARK_KEY = "_bubble_reply_request_mark"
EMOTICON_PROTOCOL_MARKER = "[bubble-reply-emoticon-protocol-v1]"
EMOTICON_PROTOCOL = f"""{EMOTICON_PROTOCOL_MARKER}
当回复中出现颜文字时，请逐字使用
<bubble-reply-emoticon>颜文字原文</bubble-reply-emoticon>
包裹该颜文字。标签及内部内容都必须原样保留；如需换行，只能在标签整体之前或之后，
不得在标签内部插入换行；普通 Markdown 三反引号代码围栏不使用此标签替代。"""

_SMART_QUOTE_OBSERVER_ENABLED = False
_SMART_QUOTE_GROUP_BLACKLIST: frozenset[str] = frozenset()
_SMART_QUOTE_FRIEND_BLACKLIST: frozenset[str] = frozenset()


class _SmartQuoteObserverFilter(filter.CustomFilter):
    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        del cfg
        if not _SMART_QUOTE_OBSERVER_ENABLED:
            return False
        # 平台若把 bot 自身消息回送为入站事件，也将其计入打断检测。
        group_id = str(event.get_group_id() or "")
        if group_id:
            return group_id not in _SMART_QUOTE_GROUP_BLACKLIST
        sender_id = str(event.get_sender_id() or "")
        return sender_id not in _SMART_QUOTE_FRIEND_BLACKLIST


class BubbleReplyPlugin(Star):
    def __init__(
        self,
        context: Context,
        config: AstrBotConfig | None = None,
    ) -> None:
        super().__init__(context, config)
        # A plugin config is supplied by StarManager in production.  Keep the
        # no-config path side-effect free for import/contract tests.
        self._raw_config = config if config is not None else {}
        self._tracker = SmartReplyTracker()
        self._runtime: BubbleReplyConfig
        self._segmenter: LlmSegmenter
        self._orchestrator: DeliveryOrchestrator
        self._rebuild_services()

    def _config_warning(self, field_name: str) -> None:
        logger.warning("[BubbleReply] config_warning field=%s", field_name)

    def _rebuild_services(self) -> None:
        global _SMART_QUOTE_FRIEND_BLACKLIST
        global _SMART_QUOTE_GROUP_BLACKLIST, _SMART_QUOTE_OBSERVER_ENABLED
        self._runtime = load_runtime_config(self._raw_config, self._config_warning)
        _SMART_QUOTE_OBSERVER_ENABLED = self._runtime.quote.mode == QuoteMode.SMART
        _SMART_QUOTE_GROUP_BLACKLIST = self._runtime.blacklist.group_ids
        _SMART_QUOTE_FRIEND_BLACKLIST = self._runtime.blacklist.friend_ids
        self._segmenter = LlmSegmenter(self._runtime.llm, logger)
        self._orchestrator = DeliveryOrchestrator(
            ConfiguredDelayPolicy(self._runtime.delay),
            logger,
        )

    async def initialize(self) -> None:
        self._rebuild_services()
        logger.info("[BubbleReply] initialized")

    async def terminate(self) -> None:
        global _SMART_QUOTE_FRIEND_BLACKLIST
        global _SMART_QUOTE_GROUP_BLACKLIST, _SMART_QUOTE_OBSERVER_ENABLED
        _SMART_QUOTE_OBSERVER_ENABLED = False
        _SMART_QUOTE_GROUP_BLACKLIST = frozenset()
        _SMART_QUOTE_FRIEND_BLACKLIST = frozenset()
        self._tracker.clear()

    def _is_blacklisted(self, gateway: AstrBotGateway) -> bool:
        if gateway.is_group:
            return gateway.group_id in self._runtime.blacklist.group_ids
        return gateway.sender_id in self._runtime.blacklist.friend_ids

    def _request_is_eligible(self, gateway: AstrBotGateway) -> bool:
        return not self._is_blacklisted(gateway)

    def _skip_reason(
        self,
        gateway: AstrBotGateway,
        snapshot: ResultSnapshot,
    ) -> str | None:
        if snapshot.content_type in {
            ResultContentType.STREAMING_RESULT,
            ResultContentType.STREAMING_FINISH,
        }:
            return "streaming"
        if self._runtime.scope == SplitScope.LLM_ONLY and not snapshot.is_llm_result:
            return "scope"
        if self._is_blacklisted(gateway):
            return "blacklist"
        limit = self._runtime.max_length_to_disable
        if limit > 0 and snapshot.plain_text_length > limit:
            return "length_limit"
        return None

    def _log_prefix(self, gateway: AstrBotGateway) -> str:
        if self._runtime.logging.log_with_bot_id and gateway.self_id:
            return f"[BubbleReply:{gateway.self_id}]"
        return "[BubbleReply]"

    def _debug(self, message: str, *args) -> None:
        method = logger.info if self._runtime.logging.debug_to_info else logger.debug
        method(message, *args)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=INCOMING_PRIORITY)
    @filter.custom_filter(
        _SmartQuoteObserverFilter,
        False,
        priority=INCOMING_PRIORITY,
    )
    async def observe_incoming_message(self, event: AstrMessageEvent) -> None:
        gateway = AstrBotGateway(self.context, event)
        mark = self._tracker.begin(gateway.umo, gateway.message_id)
        gateway.set_extra(REQUEST_MARK_KEY, mark)

    @filter.on_llm_request()
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        request: ProviderRequest,
    ) -> None:
        gateway = AstrBotGateway(self.context, event)
        if not self._request_is_eligible(gateway):
            return

        if self._runtime.quote.mode == QuoteMode.SMART:
            mark = gateway.get_extra(REQUEST_MARK_KEY)
            if not isinstance(mark, RequestMark):
                mark = self._tracker.begin(gateway.umo, gateway.message_id)
                gateway.set_extra(REQUEST_MARK_KEY, mark)

        if self._runtime.text_rules.emoticon_protection:
            if gateway.expects_visible_streaming():
                self._debug(
                    "%s emoticon_protocol skipped reason=streaming",
                    self._log_prefix(gateway),
                )
                return
            current = request.system_prompt or ""
            if EMOTICON_PROTOCOL_MARKER not in current:
                request.system_prompt = f"{current}\n\n{EMOTICON_PROTOCOL}".strip()

    def _should_add_quote(
        self,
        gateway: AstrBotGateway,
        mark: RequestMark | None,
    ) -> bool:
        policy = self._runtime.quote
        if policy.mode == QuoteMode.NONE:
            return False
        if policy.mode == QuoteMode.ALWAYS:
            return True
        if gateway.platform_name.lower() == "dingtalk":
            return False
        return self._tracker.was_interrupted(gateway.umo, mark)

    def _content_log(self, original: str, final_text: str) -> None:
        if not self._runtime.logging.log_original_text:
            return
        # Never combine message content with an account-bearing log prefix.
        self._debug("[BubbleReply] 原文本: %s", original)
        self._debug("[BubbleReply] 分段后: %s", final_text)

    async def _render_fallback_to_image(
        self,
        gateway: AstrBotGateway,
        plain_parts: list[str],
        reply_component: Any | None,
        log_prefix: str,
    ) -> bool:
        """最终回退时把纯文本渲染成图,结果链替换为 [Reply?, Image]。

        渲染失败或产出为空时返回 False,调用方保持原样发送。
        """
        text = "\n".join(part for part in plain_parts if part)
        if not text.strip():
            return False
        try:
            path = await self.text_to_image(text, return_url=False)
        except Exception as exc:
            logger.warning(
                "%s t2i_fallback failed error_type=%s",
                log_prefix,
                type(exc).__name__,
            )
            return False
        if not path:
            return False
        components: list[Any] = []
        if reply_component is not None:
            components.append(reply_component)
        components.append(Comp.Image(file=str(path)))
        segment = PlannedSegment.from_components(components)
        gateway.replace_result_segments([segment])
        logger.info("%s t2i_fallback rendered text_len=%s", log_prefix, len(text))
        return True

    @filter.on_decorating_result(priority=DECORATE_PRIORITY)
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        gateway = AstrBotGateway(self.context, event)
        snapshot = gateway.read_result()
        if snapshot is None or not snapshot.chain:
            return

        mark = gateway.get_extra(REQUEST_MARK_KEY)
        skip_reason = self._skip_reason(gateway, snapshot)
        if skip_reason is not None:
            log_prefix = self._log_prefix(gateway)
            self._debug(
                "%s skip reason=%s",
                log_prefix,
                skip_reason,
            )
            # 即使本轮跳过分段，也统一清理插件私有 XML 标签。
            gateway.strip_internal_bubble_reply_xml_tags()
            if skip_reason == "length_limit" and self._runtime.render_fallback_to_image:
                plain_parts = [
                    str(component.text)
                    for component in snapshot.chain
                    if isinstance(component, Comp.Plain)
                ]
                media_present = any(
                    not isinstance(component, (Comp.Plain, Comp.Reply))
                    for component in snapshot.chain
                )
                reply_component = next(
                    (
                        component
                        for component in snapshot.chain
                        if isinstance(component, Comp.Reply)
                    ),
                    None,
                )
                if not media_present and await self._render_fallback_to_image(
                    gateway,
                    plain_parts,
                    reply_component,
                    log_prefix,
                ):
                    self._tracker.finish(gateway.umo, mark)
                    return
            self._tracker.finish(gateway.umo, mark)
            return

        trace_id = secrets.token_hex(4)
        log_prefix = self._log_prefix(gateway)
        tokens: list[ComponentToken] = []
        baseline_tokens: list[ComponentToken] = []
        try:
            for component in snapshot.chain:
                token = gateway.to_token(component)
                if token.kind != "Plain":
                    baseline_tokens.append(token)
                    tokens.append(token)
                    continue

                original = str(token.payload)
                candidate = await self._segmenter.segment(
                    original,
                    SegmentContext(
                        gateway=gateway,
                        trace_id=trace_id,
                        log_prefix=log_prefix,
                    ),
                )
                baseline_prepared = prepare_text_for_planning(
                    candidate.cleaned,
                    extra_split_points=self._runtime.text_rules.extra_split_points,
                    interpret_literals=(
                        self._runtime.text_rules.interpret_literal_newlines
                    ),
                    emoticon_protection=self._runtime.text_rules.emoticon_protection,
                )
                chosen = candidate.layout_text or candidate.cleaned
                prepared = prepare_text_for_planning(
                    chosen,
                    extra_split_points=self._runtime.text_rules.extra_split_points,
                    # 统一按配置还原占位符:主 LLM 原文的字面量命运不因分段成败改变;
                    # 分段 LLM 新增的字面量在 prepare 内无条件折叠为换行。
                    interpret_literals=(
                        self._runtime.text_rules.interpret_literal_newlines
                    ),
                    emoticon_protection=self._runtime.text_rules.emoticon_protection,
                )
                self._content_log(original, prepared)
                baseline_tokens.append(ComponentToken("Plain", baseline_prepared))
                tokens.append(ComponentToken("Plain", prepared))

            should_reply = self._should_add_quote(gateway, mark)
            reply_token = gateway.make_reply_token() if should_reply else None
            baseline_plan = plan_delivery(
                baseline_tokens,
                media=self._runtime.media,
                strip_segment_tail_chars=self._runtime.text_rules.strip_segment_tail_chars,
                strip_segment_whitespace=(
                    self._runtime.text_rules.strip_segment_whitespace
                ),
                should_add_reply=should_reply,
                reply_token=reply_token,
            )
            plan = plan_delivery(
                tokens,
                media=self._runtime.media,
                strip_segment_tail_chars=self._runtime.text_rules.strip_segment_tail_chars,
                strip_segment_whitespace=(
                    self._runtime.text_rules.strip_segment_whitespace
                ),
                should_add_reply=should_reply,
                reply_token=reply_token,
            )
            candidate_segments = len(plan.all_segments)
            baseline_segments = len(baseline_plan.all_segments)
            limit = self._runtime.max_segments_to_disable
            plan = apply_segment_limit(plan, baseline_plan, limit)
            logger.info(
                "%s plan mode=%s segments=%s candidate_segments=%s baseline_segments=%s limit=%s text_len=%s reason=%s trace=%s",
                log_prefix,
                plan.mode,
                len(plan.all_segments),
                candidate_segments,
                baseline_segments,
                limit,
                sum(segment.plain_text_length for segment in plan.all_segments),
                plan.reason,
                trace_id,
            )
            if (
                plan.mode == "respond_only"
                and plan.reason == "segment_limit"
                and self._runtime.render_fallback_to_image
            ):
                plain_parts = [
                    str(token.payload)
                    for segment in plan.respond_segments
                    for token in segment.components
                    if token.kind == "Plain"
                ]
                media_present = any(
                    token.kind not in {"Plain", "Reply"}
                    for segment in plan.respond_segments
                    for token in segment.components
                )
                reply_component = next(
                    (
                        token.payload
                        for segment in plan.respond_segments
                        for token in segment.components
                        if token.kind == "Reply"
                    ),
                    None,
                )
                if not media_present and await self._render_fallback_to_image(
                    gateway,
                    plain_parts,
                    reply_component,
                    log_prefix,
                ):
                    return
            await self._orchestrator.execute(
                plan,
                gateway,
                trace_id=trace_id,
                log_prefix=log_prefix,
            )
        finally:
            self._tracker.finish(gateway.umo, mark)
