from __future__ import annotations

from .models import ValidationResult
from .text_rules import EMOTICON_CLOSE, EMOTICON_OPEN


_LITERAL_LAYOUT_TOKENS = ("\\r\\n", "\\n", "\\r")


def _without_real_newlines(text: str) -> str:
    return text.replace("\r\n", "").replace("\r", "").replace("\n", "")


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
    return depth == 0


def _has_unexpected_wrapper(baseline: str, candidate: str) -> bool:
    base = baseline.strip()
    value = candidate.strip()
    wrappers = (
        ("<output>", "</output>"),
        ("<result>", "</result>"),
        ("<input>", "</input>"),
    )
    return any(
        value.startswith(start)
        and value.endswith(end)
        and not (base.startswith(start) and base.endswith(end))
        for start, end in wrappers
    )


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


def validate_layout_only_change(
    baseline: str,
    candidate: str,
) -> ValidationResult:
    baseline_body = _without_real_newlines(baseline)
    baseline_count = len(baseline_body)
    candidate_count = len(_without_real_newlines(candidate))

    if candidate == "":
        return ValidationResult(
            False,
            "empty_candidate",
            0,
            baseline_count,
            0,
        )
    if not _emoticon_tags_balanced(candidate):
        return ValidationResult(
            False,
            "unclosed_emoticon_tag",
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
                output.append("\n")
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
