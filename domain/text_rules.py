from __future__ import annotations

import re
from collections.abc import Iterable


EMOTICON_OPEN = "<bubble-reply-emoticon>"
EMOTICON_CLOSE = "</bubble-reply-emoticon>"


def normalize_real_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def interpret_literal_newlines(text: str) -> str:
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


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


def prepare_text_for_planning(
    text: str,
    *,
    extra_split_points: Iterable[str] = (),
    interpret_literals: bool = False,
    emoticon_protection: bool = True,
) -> str:
    result = normalize_real_newlines(text)
    if interpret_literals:
        result = interpret_literal_newlines(result)
    result = replace_extra_split_points(result, extra_split_points)
    if emoticon_protection:
        result = strip_emoticon_tags(result)
    return result


def split_nonempty_lines(text: str) -> list[str]:
    return [part for part in normalize_real_newlines(text).split("\n") if part.strip()]


def strip_tail_chars(text: str, chars: str) -> str:
    return text.rstrip(chars) if chars else text
