from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


WarnCallback = Callable[[str], None]

DEFAULT_LLM_SPLIT_PROMPT = (
    "请将下面的文本整理成自然的即时聊天分段。每段表达一个完整意群，通常保留 1–2 句；"
    "短回复保持单段，长回复优先在话题切换、语义转折、动作与台词边界处分段。"
    "避免把称谓、否定结构、数字、URL、代码、JSON 或紧密关联的上下文拆开。"
    "每段建议约 20–80 个中文字符，但自然性与语义完整性优先，不要为了长度机械切割。\n\n"
    "待分段文本：\n{text}"
)


class SplitScope(str, Enum):
    LLM_ONLY = "LLM_ONLY"
    ALL = "ALL"


class ComponentPolicy(str, Enum):
    SEPARATE = "单独发送"
    FOLLOW_PREVIOUS = "跟随上一段"
    FOLLOW_NEXT = "跟随下一段"
    EMBED = "保持原位置"


class QuoteMode(str, Enum):
    NONE = "不引用"
    ALWAYS = "始终引用原消息"
    SMART = "智能引用"


class DelayMode(str, Enum):
    FIXED = "固定间隔"
    PER_CHARACTER = "按字数"


@dataclass(frozen=True)
class TextRuleConfig:
    extra_split_points: tuple[str, ...]
    interpret_literal_newlines: bool
    strip_segment_whitespace: bool
    strip_segment_tail_chars: str
    emoticon_protection: bool


@dataclass(frozen=True)
class LlmSegmenterConfig:
    enabled: bool
    provider_id: str
    runtime_rule: str
    remove_before_split_regex: tuple[str, ...]
    sanitize_llm_output_regex: tuple[str, ...]
    allow_text_changes: bool = False


@dataclass(frozen=True)
class MediaPolicyConfig:
    image: ComponentPolicy
    face: ComponentPolicy
    at: ComponentPolicy

    def for_kind(self, kind: str) -> ComponentPolicy:
        if kind == "Image":
            return self.image
        if kind == "Face":
            return self.face
        if kind == "At":
            return self.at
        return ComponentPolicy.EMBED


@dataclass(frozen=True)
class QuotePolicyConfig:
    mode: QuoteMode


@dataclass(frozen=True)
class DelayConfig:
    seconds: float
    mode: DelayMode = DelayMode.FIXED
    seconds_per_character: float = 0.025
    minimum_seconds: float = 0.5
    maximum_seconds: float = 2.5
    jitter_seconds: float = 0.0


@dataclass(frozen=True)
class LoggingConfig:
    log_original_text: bool
    log_with_bot_id: bool
    debug_to_info: bool


@dataclass(frozen=True)
class BlacklistConfig:
    group_ids: frozenset[str]
    friend_ids: frozenset[str]


@dataclass(frozen=True)
class BubbleReplyConfig:
    scope: SplitScope
    max_length_to_disable: int
    blacklist: BlacklistConfig
    text_rules: TextRuleConfig
    llm: LlmSegmenterConfig
    media: MediaPolicyConfig
    quote: QuotePolicyConfig
    delay: DelayConfig
    logging: LoggingConfig


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _int(value: Any, default: int, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, parsed)


def _float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, parsed)


def _enum(
    enum_type: type[Enum],
    value: Any,
    default: Enum,
    field_name: str,
    warn: WarnCallback,
) -> Enum:
    try:
        return enum_type(str(value))
    except (TypeError, ValueError):
        warn(field_name)
        return default


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return ()
    return tuple(str(item) for item in values if str(item))


def _validated_patterns(
    value: Any,
    field_name: str,
    warn: WarnCallback,
) -> tuple[str, ...]:
    valid: list[str] = []
    for pattern in _string_tuple(value):
        try:
            re.compile(pattern)
        except re.error:
            warn(field_name)
            continue
        valid.append(pattern)
    return tuple(valid)


def _blacklist(value: Any, warn: WarnCallback) -> BlacklistConfig:
    group_ids: set[str] = set()
    friend_ids: set[str] = set()
    for item in _string_tuple(value):
        prefix, separator, identifier = item.strip().partition(":")
        identifier = identifier.strip()
        if separator and identifier and prefix.upper() == "G":
            group_ids.add(identifier)
        elif separator and identifier and prefix.upper() == "F":
            friend_ids.add(identifier)
        else:
            warn("basic_settings.blacklist")
    return BlacklistConfig(frozenset(group_ids), frozenset(friend_ids))


def load_runtime_config(
    raw: Mapping[str, Any] | None,
    warn: WarnCallback | None = None,
) -> BubbleReplyConfig:
    raw = raw if isinstance(raw, Mapping) else {}
    warn = warn or (lambda _field: None)

    basic = _section(raw, "basic_settings")
    split = _section(raw, "split_settings")
    llm = _section(raw, "llm_split_settings")
    quote_media = _section(raw, "quote_media_settings")
    delay = _section(raw, "delay_settings")
    logging = _section(raw, "log_config")

    scope = _enum(
        SplitScope,
        basic.get("split_scope", SplitScope.LLM_ONLY.value),
        SplitScope.LLM_ONLY,
        "basic_settings.split_scope",
        warn,
    )
    image_policy = _enum(
        ComponentPolicy,
        quote_media.get("image_strategy", ComponentPolicy.EMBED.value),
        ComponentPolicy.EMBED,
        "quote_media_settings.image_strategy",
        warn,
    )
    face_policy = _enum(
        ComponentPolicy,
        quote_media.get("face_strategy", ComponentPolicy.EMBED.value),
        ComponentPolicy.EMBED,
        "quote_media_settings.face_strategy",
        warn,
    )
    at_policy = _enum(
        ComponentPolicy,
        quote_media.get("at_strategy", ComponentPolicy.EMBED.value),
        ComponentPolicy.EMBED,
        "quote_media_settings.at_strategy",
        warn,
    )
    quote_mode = _enum(
        QuoteMode,
        quote_media.get("quote_mode", QuoteMode.NONE.value),
        QuoteMode.NONE,
        "quote_media_settings.quote_mode",
        warn,
    )
    delay_mode = _enum(
        DelayMode,
        delay.get("delay_mode", DelayMode.FIXED.value),
        DelayMode.FIXED,
        "delay_settings.delay_mode",
        warn,
    )
    minimum_delay_seconds = _float(
        delay.get("minimum_delay_seconds", 0.5),
        0.5,
    )
    maximum_delay_seconds = _float(
        delay.get("maximum_delay_seconds", 2.5),
        2.5,
    )
    if maximum_delay_seconds < minimum_delay_seconds:
        warn("delay_settings.maximum_delay_seconds")
        maximum_delay_seconds = minimum_delay_seconds
    tail_chars = split.get("strip_segment_tail_chars", "")
    if isinstance(tail_chars, (list, tuple, set, frozenset)):
        tail_chars = "".join(str(item) for item in tail_chars)
    elif not isinstance(tail_chars, str):
        tail_chars = str(tail_chars or "")

    runtime_rule = str(
        llm.get("llm_split_prompt", DEFAULT_LLM_SPLIT_PROMPT)
        or DEFAULT_LLM_SPLIT_PROMPT
    ).strip()
    if "{text}" not in runtime_rule:
        warn("llm_split_settings.llm_split_prompt")
        runtime_rule = f"{runtime_rule}\n\n{{text}}"

    return BubbleReplyConfig(
        scope=scope,  # type: ignore[arg-type]
        max_length_to_disable=_int(
            basic.get("max_length_to_disable", 0),
            0,
        ),
        blacklist=_blacklist(basic.get("blacklist", []), warn),
        text_rules=TextRuleConfig(
            extra_split_points=tuple(
                point
                for point in _string_tuple(split.get("extra_split_points", []))
                if point
            ),
            interpret_literal_newlines=_bool(
                split.get("interpret_literal_newlines", False),
                False,
            ),
            strip_segment_whitespace=_bool(
                split.get("strip_segment_whitespace", True),
                True,
            ),
            strip_segment_tail_chars=tail_chars,
            emoticon_protection=_bool(
                split.get("emoticon_protection", True),
                True,
            ),
        ),
        llm=LlmSegmenterConfig(
            enabled=_bool(llm.get("enable_llm_split", False), False),
            provider_id=str(llm.get("llm_split_provider", "") or "").strip(),
            runtime_rule=runtime_rule,
            remove_before_split_regex=_validated_patterns(
                llm.get("remove_before_split_regex", []),
                "llm_split_settings.remove_before_split_regex",
                warn,
            ),
            sanitize_llm_output_regex=_validated_patterns(
                llm.get("sanitize_llm_output_regex", []),
                "llm_split_settings.sanitize_llm_output_regex",
                warn,
            ),
            allow_text_changes=_bool(
                llm.get("allow_llm_text_changes", False),
                False,
            ),
        ),
        media=MediaPolicyConfig(
            image=image_policy,  # type: ignore[arg-type]
            face=face_policy,  # type: ignore[arg-type]
            at=at_policy,  # type: ignore[arg-type]
        ),
        quote=QuotePolicyConfig(
            mode=quote_mode,  # type: ignore[arg-type]
        ),
        delay=DelayConfig(
            seconds=_float(delay.get("delay_seconds", 0.8), 0.8),
            mode=delay_mode,  # type: ignore[arg-type]
            seconds_per_character=_float(
                delay.get("seconds_per_character", 0.025),
                0.025,
            ),
            minimum_seconds=minimum_delay_seconds,
            maximum_seconds=maximum_delay_seconds,
            jitter_seconds=_float(
                delay.get("random_jitter_seconds", 0.0),
                0.0,
            ),
        ),
        logging=LoggingConfig(
            log_original_text=_bool(
                logging.get("log_original_text", False),
                False,
            ),
            log_with_bot_id=_bool(
                logging.get("log_with_bot_id", False),
                False,
            ),
            debug_to_info=_bool(logging.get("debug_to_info", False), False),
        ),
    )
