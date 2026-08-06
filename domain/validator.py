from __future__ import annotations

from .models import ValidationResult
from .text_rules import (
    EMOTICON_CLOSE,
    EMOTICON_OPEN,
    NEWLINE_PLACEHOLDERS,
    normalize_real_newlines,
)


_LITERAL_LAYOUT_TOKENS = ("\\r\\n", "\\n\\r", "\\n", "\\r")
_RELAXED_LENGTH_TOLERANCE_PERCENT = 10


def _without_real_newlines(text: str) -> str:
    return text.replace("\r\n", "").replace("\r", "").replace("\n", "")


def _without_layout_tokens(text: str) -> str:
    result = _without_real_newlines(text)
    for token in _LITERAL_LAYOUT_TOKENS:
        result = result.replace(token, "")
    for placeholder in NEWLINE_PLACEHOLDERS:
        result = result.replace(placeholder, "")
    return result.replace(EMOTICON_OPEN, "").replace(EMOTICON_CLOSE, "")


def _emoticon_tags_balanced(text: str) -> bool:
    cursor = 0
    depth = 0
    while cursor < len(text):
        open_at = text.find(EMOTICON_OPEN, cursor)
        close_at = text.find(EMOTICON_CLOSE, cursor)
        if open_at < 0 and close_at < 0:
            break
        if close_at >= 0 and (open_at < 0 or close_at < open_at):
            depth -= 1
            if depth < 0:
                return False
            cursor = close_at + len(EMOTICON_CLOSE)
        else:
            depth += 1
            if depth > 1:
                return False
            cursor = open_at + len(EMOTICON_OPEN)
    if depth != 0:
        return False
    without_known_tags = text.replace(EMOTICON_OPEN, "").replace(
        EMOTICON_CLOSE,
        "",
    )
    return not (
        "<bubble-reply-emoticon" in without_known_tags
        or "</bubble-reply-emoticon" in without_known_tags
    )


def _emoticon_tag_split_by_newline(text: str) -> bool:
    """保护协议标签:开闭标签之间不得出现真实换行(换行只能加在整体前后)。"""
    cursor = 0
    while True:
        open_at = text.find(EMOTICON_OPEN, cursor)
        if open_at < 0:
            return False
        close_at = text.find(EMOTICON_CLOSE, open_at + len(EMOTICON_OPEN))
        if close_at < 0:
            return False
        inner = text[open_at + len(EMOTICON_OPEN) : close_at]
        if "\n" in inner or "\r" in inner:
            return True
        cursor = close_at + len(EMOTICON_CLOSE)


def _marker_counts(text: str) -> tuple[int, ...]:
    """分别统计四种换行占位符及颜文字开、闭标签。"""
    return (
        *(text.count(placeholder) for placeholder in NEWLINE_PLACEHOLDERS),
        text.count(EMOTICON_OPEN),
        text.count(EMOTICON_CLOSE),
    )


def _protected_markers_preserved(baseline: str, candidate: str) -> bool:
    """受保护标记必须逐字保留:开/闭标签与占位符分别计数,任一侧增减都拦截。

    注意:整体删除一组标签 = 开、闭各自减一,两个计数都变化,同样会被拦截。
    """
    return _marker_counts(baseline) == _marker_counts(candidate)


def _has_unexpected_wrapper(baseline: str, candidate: str) -> bool:
    base = baseline.strip()
    value = candidate.strip()
    wrappers = (
        ("<output>", "</output>"),
        ("<result>", "</result>"),
        ("<input>", "</input>"),
    )
    has_tag_wrapper = any(
        value.startswith(start)
        and value.endswith(end)
        and not (base.startswith(start) and base.endswith(end))
        for start, end in wrappers
    )
    has_fence_wrapper = (
        value.startswith("```")
        and value.endswith("```")
        and not (base.startswith("```") and base.endswith("```"))
    )
    return has_tag_wrapper or has_fence_wrapper


def _literal_token_at(text: str, position: int) -> str | None:
    for token in _LITERAL_LAYOUT_TOKENS:
        if text.startswith(token, position):
            return token
    return None


def _mismatch_reason(baseline: str, candidate: str, i: int, j: int) -> str:
    if i >= len(baseline):
        return "added_character"
    if j >= len(candidate):
        return "removed_character"
    if j + 1 < len(candidate) and candidate[j + 1] == baseline[i]:
        return "added_character"
    if i + 1 < len(baseline) and candidate[j] == baseline[i + 1]:
        return "removed_character"
    return "changed_character"


def _basic_candidate_rejection(
    baseline: str,
    candidate: str,
) -> ValidationResult | None:
    baseline_count = len(_without_real_newlines(baseline))
    candidate_count = len(_without_real_newlines(candidate))
    if not baseline.strip():
        return ValidationResult(
            False,
            "empty_baseline",
            0,
            baseline_count,
            candidate_count,
        )
    if not candidate.strip():
        return ValidationResult(
            False,
            "empty_candidate",
            0,
            baseline_count,
            candidate_count,
        )
    if not _emoticon_tags_balanced(candidate):
        return ValidationResult(
            False,
            "unclosed_emoticon_tag",
            None,
            baseline_count,
            candidate_count,
        )
    if _emoticon_tag_split_by_newline(candidate):
        return ValidationResult(
            False,
            "emoticon_tag_split",
            None,
            baseline_count,
            candidate_count,
        )
    if not _protected_markers_preserved(baseline, candidate):
        marker_reason = (
            "placeholder_modified"
            if any(
                candidate.count(placeholder) != baseline.count(placeholder)
                for placeholder in NEWLINE_PLACEHOLDERS
            )
            else "emoticon_tag_count"
        )
        return ValidationResult(
            False,
            marker_reason,
            None,
            baseline_count,
            candidate_count,
        )
    if _has_unexpected_wrapper(baseline, candidate):
        return ValidationResult(
            False,
            "unexpected_wrapper",
            0,
            baseline_count,
            candidate_count,
        )
    return None


def validate_relaxed_candidate(
    baseline: str,
    candidate: str,
) -> ValidationResult:
    rejection = _basic_candidate_rejection(baseline, candidate)
    if rejection is not None:
        return rejection

    baseline_count = len(_without_layout_tokens(baseline))
    candidate_count = len(_without_layout_tokens(candidate))
    length_change_percent_exceeded = (
        abs(candidate_count - baseline_count) * 100
        > baseline_count * _RELAXED_LENGTH_TOLERANCE_PERCENT
    )
    if length_change_percent_exceeded:
        return ValidationResult(
            False,
            "excessive_length_change",
            None,
            baseline_count,
            candidate_count,
        )

    return ValidationResult(
        True,
        "accepted_relaxed",
        None,
        baseline_count,
        candidate_count,
        normalized_candidate=normalize_real_newlines(candidate),
    )


def validate_layout_only_change(
    baseline: str,
    candidate: str,
) -> ValidationResult:
    baseline_body = _without_real_newlines(baseline)
    baseline_count = len(baseline_body)
    candidate_count = len(_without_real_newlines(candidate))

    rejection = _basic_candidate_rejection(baseline, candidate)
    if rejection is not None:
        return rejection

    output: list[str] = []
    i = 0
    j = 0
    while j < len(candidate):
        if candidate.startswith("\r\n", j):
            output.append("\n")
            j += 2
            continue
        if candidate[j] in {"\r", "\n"}:
            output.append("\n")
            j += 1
            continue

        literal = _literal_token_at(candidate, j)
        if literal is not None:
            matching_content = next(
                (
                    token
                    for token in _LITERAL_LAYOUT_TOKENS
                    if candidate.startswith(token, j)
                    and baseline_body.startswith(token, i)
                ),
                None,
            )
            if matching_content is not None:
                output.append(matching_content)
                i += len(matching_content)
                j += len(matching_content)
            else:
                # 新增字面量:不再折叠,原样输出,由 prepare_text_for_planning
                # 统一折叠为真实换行(此时出现必为分段 LLM 新增)。
                output.append(literal)
                j += len(literal)
            continue

        if i < len(baseline_body) and candidate[j] == baseline_body[i]:
            output.append(candidate[j])
            i += 1
            j += 1
            continue

        return ValidationResult(
            False,
            _mismatch_reason(baseline_body, candidate, i, j),
            j,
            baseline_count,
            candidate_count,
        )

    if i != len(baseline_body):
        return ValidationResult(
            False,
            "removed_character",
            j,
            baseline_count,
            candidate_count,
        )

    return ValidationResult(
        True,
        "accepted",
        None,
        baseline_count,
        candidate_count,
        normalized_candidate="".join(output),
    )
