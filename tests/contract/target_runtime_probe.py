"""Small probe intended for the real AstrBot 4.26.x Python environment."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import astrbot.api.message_components as Comp
from astrbot.api.event import MessageEventResult, ResultContentType
from astrbot.api.provider import ProviderRequest
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.platform.message_type import MessageType
from astrbot.core.star.star_handler import EventType, star_handlers_registry

from bubble_reply.domain.models import PlannedSegment
from bubble_reply.integrations.astrbot_gateway import AstrBotGateway
from bubble_reply.main import BubbleReplyPlugin, DECORATE_PRIORITY


class _Event:
    def __init__(self, result: MessageEventResult) -> None:
        self._result = result
        self._extra: dict[str, object] = {}
        self.sent = []
        self.unified_msg_origin = "platform_demo:FriendMessage:session_demo"
        self.session_id = "session_demo"
        self.message_obj = type("Message", (), {"message_id": "message_demo"})()
        self.platform_meta = type(
            "PlatformMeta",
            (),
            {"support_streaming_message": True},
        )()

    def get_result(self):
        return self._result

    async def send(self, chain) -> None:
        self.sent.append(chain)

    def get_sender_id(self) -> str:
        return "10001"

    def get_self_id(self) -> str:
        return "10002"

    def get_platform_name(self) -> str:
        return "platform_demo"

    def get_group_id(self) -> str:
        return ""

    def get_message_type(self):
        return MessageType.FRIEND_MESSAGE

    def get_extra(self, key, default=None):
        return self._extra.get(key, default)

    def set_extra(self, key, value) -> None:
        self._extra[key] = value


class _Context:
    def __init__(
        self,
        *,
        streaming_response: bool = False,
        unsupported_strategy: str = "realtime_segmenting",
    ) -> None:
        self.streaming_response = streaming_response
        self.unsupported_strategy = unsupported_strategy

    def get_config(self, _umo=None):
        return {
            "provider_settings": {
                "streaming_response": self.streaming_response,
                "unsupported_streaming_strategy": self.unsupported_strategy,
            }
        }


async def _run() -> None:
    schema_path = Path(__file__).resolve().parents[2] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as temp_dir:
        config = AstrBotConfig(
            config_path=str(Path(temp_dir) / "bubble_reply_config.json"),
            schema=schema,
        )
        assert config["basic_settings"]["split_scope"] == "LLM_ONLY"
        assert config["basic_settings"]["max_segments_to_disable"] == 0
        assert config["split_settings"]["strip_segment_whitespace"] is True
        assert set(config["quote_media_settings"]) == {
            "quote_mode",
            "image_strategy",
        }
        assert set(config["delay_settings"]) == {
            "delay_mode",
            "delay_seconds",
            "seconds_per_character",
            "minimum_delay_seconds",
            "maximum_delay_seconds",
            "random_jitter_seconds",
        }
        assert config["delay_settings"]["delay_mode"] == "固定间隔"
        assert config["delay_settings"]["random_jitter_seconds"] == 0
        assert config["llm_split_settings"]["allow_llm_text_changes"] is False
        assert "20–80" in config["llm_split_settings"]["llm_split_prompt"]

    result = MessageEventResult([Comp.Plain("a")])
    result.use_markdown_ = True
    result.set_result_content_type(ResultContentType.LLM_RESULT)
    derived = result.derive([Comp.Plain("b")])
    assert derived.use_markdown_ is True

    event = _Event(result)
    gateway = AstrBotGateway(object(), event)  # type: ignore[arg-type]
    reply = gateway.make_reply_token()
    assert reply is not None and reply.kind == "Reply"
    planned = PlannedSegment.from_components([gateway.to_token(Comp.Plain("x"))])
    planned_tail = PlannedSegment.from_components(
        [gateway.to_token(Comp.Plain("y"))]
    )
    gateway.replace_result_segments([planned, planned_tail])
    assert result.get_plain_text() == "x\ny"
    assert all(
        not isinstance(component, Comp.Plain) or component.text.strip()
        for component in result.chain
    )
    await gateway.send_active(planned)
    assert len(event.sent) == 1

    plugin = BubbleReplyPlugin(
        _Context(),  # type: ignore[arg-type]
        {
            "basic_settings": {"split_scope": "ALL"},
            "delay_settings": {"delay_seconds": 0},
        },
    )
    await plugin.initialize()

    non_stream_request = ProviderRequest(system_prompt="persona")
    non_stream_event = _Event(MessageEventResult([Comp.Plain("request")]))
    await plugin.on_llm_request(  # type: ignore[arg-type]
        non_stream_event,
        non_stream_request,
    )
    assert "bubble-reply-emoticon-protocol-v1" in non_stream_request.system_prompt

    stream_plugin = BubbleReplyPlugin(
        _Context(streaming_response=True),  # type: ignore[arg-type]
        {"basic_settings": {"split_scope": "ALL"}},
    )
    stream_request = ProviderRequest(system_prompt="persona")
    stream_event = _Event(MessageEventResult([Comp.Plain("request")]))
    await stream_plugin.on_llm_request(  # type: ignore[arg-type]
        stream_event,
        stream_request,
    )
    assert "bubble-reply-emoticon-protocol-v1" not in stream_request.system_prompt

    fallback_plugin = BubbleReplyPlugin(
        _Context(
            streaming_response=True,
            unsupported_strategy="turn_off",
        ),  # type: ignore[arg-type]
        {"basic_settings": {"split_scope": "ALL"}},
    )
    fallback_request = ProviderRequest(system_prompt="persona")
    fallback_event = _Event(MessageEventResult([Comp.Plain("request")]))
    fallback_event.platform_meta.support_streaming_message = False
    await fallback_plugin.on_llm_request(  # type: ignore[arg-type]
        fallback_event,
        fallback_request,
    )
    assert "bubble-reply-emoticon-protocol-v1" in fallback_request.system_prompt

    override_request = ProviderRequest(system_prompt="persona")
    override_event = _Event(MessageEventResult([Comp.Plain("request")]))
    override_event.set_extra("enable_streaming", True)
    await plugin.on_llm_request(  # type: ignore[arg-type]
        override_event,
        override_request,
    )
    assert "bubble-reply-emoticon-protocol-v1" not in override_request.system_prompt

    disable_override_request = ProviderRequest(system_prompt="persona")
    disable_override_event = _Event(MessageEventResult([Comp.Plain("request")]))
    disable_override_event.set_extra("enable_streaming", False)
    await stream_plugin.on_llm_request(  # type: ignore[arg-type]
        disable_override_event,
        disable_override_request,
    )
    assert (
        "bubble-reply-emoticon-protocol-v1"
        in disable_override_request.system_prompt
    )

    live_request = ProviderRequest(system_prompt="persona")
    live_event = _Event(MessageEventResult([Comp.Plain("request")]))
    live_event.set_extra("action_type", "live")
    await plugin.on_llm_request(live_event, live_request)  # type: ignore[arg-type]
    assert "bubble-reply-emoticon-protocol-v1" not in live_request.system_prompt

    flow_result = MessageEventResult([Comp.Plain("  a  \n   \n b \n c  ")])
    flow_event = _Event(flow_result)
    await plugin.on_decorating_result(flow_event)  # type: ignore[arg-type]
    assert [chain.get_plain_text() for chain in flow_event.sent] == ["a", "b"]
    assert flow_result.get_plain_text() == "c"

    mixed_result = MessageEventResult(
        [Comp.Plain("  a  "), Comp.At(qq="10001"), Comp.Plain("  b  ")]
    )
    mixed_event = _Event(mixed_result)
    await plugin.on_decorating_result(mixed_event)  # type: ignore[arg-type]
    mixed_plain = [
        component.text
        for component in mixed_result.chain
        if isinstance(component, Comp.Plain)
    ]
    assert mixed_plain == ["a  ", "  b"]

    unsafe_result = MessageEventResult(
        [
            Comp.Plain("a\n"),
            Comp.File(name="demo.txt", file="demo.txt"),
            Comp.Plain("\nb"),
        ]
    )
    unsafe_event = _Event(unsafe_result)
    await plugin.on_decorating_result(unsafe_event)  # type: ignore[arg-type]
    assert not unsafe_event.sent
    assert any(isinstance(component, Comp.File) for component in unsafe_result.chain)

    skipped_plugin = BubbleReplyPlugin(
        _Context(),  # type: ignore[arg-type]
        {
            "basic_settings": {
                "split_scope": "ALL",
                "max_length_to_disable": 1,
            }
        },
    )
    skipped_result = MessageEventResult(
        [Comp.Plain("<bubble-reply-x>正文</bubble-reply-x>")]
    )
    skipped_event = _Event(skipped_result)
    await skipped_plugin.on_decorating_result(  # type: ignore[arg-type]
        skipped_event
    )
    assert skipped_result.get_plain_text() == "正文"

    await plugin.terminate()
    await stream_plugin.terminate()
    await fallback_plugin.terminate()
    await skipped_plugin.terminate()

    smart_plugin = BubbleReplyPlugin(
        _Context(),  # type: ignore[arg-type]
        {
            "basic_settings": {"split_scope": "ALL"},
            "quote_media_settings": {
                "quote_mode": "智能引用",
            },
            "delay_settings": {"delay_seconds": 0},
        },
    )
    await smart_plugin.initialize()
    first_event = _Event(MessageEventResult([Comp.Plain("a\nb")]))
    later_non_llm_event = _Event(MessageEventResult([Comp.Plain("旁路消息")]))
    observer_handler = next(
        handler
        for handler in star_handlers_registry._handlers
        if handler.event_type == EventType.AdapterMessageEvent
        and handler.handler_module_path == "bubble_reply.main"
    )
    assert all(event_filter.filter(first_event, {}) for event_filter in observer_handler.event_filters)
    await smart_plugin.observe_incoming_message(first_event)  # type: ignore[arg-type]
    await smart_plugin.observe_incoming_message(  # type: ignore[arg-type]
        later_non_llm_event
    )
    await smart_plugin.on_decorating_result(first_event)  # type: ignore[arg-type]
    assert isinstance(first_event.sent[0].chain[0], Comp.Reply)
    await smart_plugin.terminate()

    decorate_handlers = [
        handler
        for handler in star_handlers_registry._handlers
        if handler.event_type == EventType.OnDecoratingResultEvent
        and handler.handler_module_path == "bubble_reply.main"
    ]
    assert len(decorate_handlers) == 1
    assert decorate_handlers[0].extras_configs["priority"] == DECORATE_PRIORITY

    incoming_handlers = [
        handler
        for handler in star_handlers_registry._handlers
        if handler.event_type == EventType.AdapterMessageEvent
        and handler.handler_module_path == "bubble_reply.main"
    ]
    assert len(incoming_handlers) == 1


if __name__ == "__main__":
    asyncio.run(_run())
    print("TARGET_RUNTIME_PROBE_OK")
