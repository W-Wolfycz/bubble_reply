from __future__ import annotations

import re
from collections.abc import Iterable


EMOTICON_OPEN = "<bubble-reply-emoticon>"
EMOTICON_CLOSE = "</bubble-reply-emoticon>"

# 字面量换行在分段前使用不同占位符，避免关闭解释时丢失原始种类和顺序。
PLACEHOLDER_CR = "<bubble-cr/>"
PLACEHOLDER_LF = "<bubble-lf/>"
PLACEHOLDER_CRLF = "<bubble-crlf/>"
PLACEHOLDER_LFCR = "<bubble-lfcr/>"

LITERAL_NEWLINE_PLACEHOLDERS = (
    ("\\r\\n", PLACEHOLDER_CRLF),
    ("\\n\\r", PLACEHOLDER_LFCR),
    ("\\r", PLACEHOLDER_CR),
    ("\\n", PLACEHOLDER_LF),
)
NEWLINE_PLACEHOLDERS = tuple(
    placeholder for _literal, placeholder in LITERAL_NEWLINE_PLACEHOLDERS
)

_BUBBLE_REPLY_XML_TAG = re.compile(
    r"</?bubble-reply(?=[\s/>-])(?:-[A-Za-z0-9_.:-]+)?"
    r"(?:\s+[^<>]*)?\s*/?>",
    re.IGNORECASE,
)


def normalize_real_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def interpret_literal_newlines(text: str) -> str:
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n\\r", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )


def escape_literal_newlines(text: str) -> str:
    """把主 LLM 原文中的四种字面量换行替换为对应占位符。"""
    result = text
    for literal, placeholder in LITERAL_NEWLINE_PLACEHOLDERS:
        result = result.replace(literal, placeholder)
    return result


def restore_placeholder_newlines(text: str, interpret: bool) -> str:
    """按配置将占位符还原为真实换行或原始字面量。"""
    result = text
    for literal, placeholder in LITERAL_NEWLINE_PLACEHOLDERS:
        result = result.replace(placeholder, "\n" if interpret else literal)
    return result


def replace_extra_split_points(text: str, split_points: Iterable[str]) -> str:
    result = text
    for point in split_points:
        if point:
            result = result.replace(point, "\n")
    return result


def apply_regex_removals(text: str, patterns: Iterable[re.Pattern[str]]) -> str:
    result = text
    for pattern in patterns:
        result = pattern.sub("", result)
    return result


def strip_emoticon_tags(text: str) -> str:
    return text.replace(EMOTICON_OPEN, "").replace(EMOTICON_CLOSE, "")


def strip_bubble_reply_xml_tags(text: str) -> str:
    """兜底移除 bubble-reply 私有命名空间的 XML 标签，仅保留标签内容。"""
    return _BUBBLE_REPLY_XML_TAG.sub("", text)


def prepare_text_for_planning(
    text: str,
    *,
    extra_split_points: Iterable[str] = (),
    interpret_literals: bool = False,
    emoticon_protection: bool = True,
) -> str:
    # 输入合同:来自 LlmSegmenter 的文本,主 LLM 原文的字面量换行已替换为占位符。
    result = normalize_real_newlines(text)
    # 分段 LLM 新增的字面量(违反约束时)一律视为分段意图,折叠为真实换行。
    result = interpret_literal_newlines(result)
    # 占位符(主 LLM 原文的字面量)按配置还原:解释为换行或保留字面量。
    result = restore_placeholder_newlines(result, interpret_literals)
    result = replace_extra_split_points(result, extra_split_points)
    if emoticon_protection:
        result = strip_emoticon_tags(result)
    return strip_bubble_reply_xml_tags(result)


def strip_tail_chars(text: str, chars: str) -> str:
    return text.rstrip(chars) if chars else text
