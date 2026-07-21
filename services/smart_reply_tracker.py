from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestMark:
    sequence: int
    message_key: str


@dataclass
class _SessionState:
    latest_sequence: int = 0
    message_marks: OrderedDict[str, int] = field(default_factory=OrderedDict)


class SmartReplyTracker:
    def __init__(
        self,
        *,
        max_sessions: int = 1000,
        max_marks_per_session: int = 200,
    ) -> None:
        self._max_sessions = max(1, max_sessions)
        self._max_marks_per_session = max(1, max_marks_per_session)
        self._sessions: OrderedDict[str, _SessionState] = OrderedDict()

    def begin(self, session_key: str, message_key: str) -> RequestMark:
        state = self._sessions.pop(session_key, _SessionState())
        state.latest_sequence += 1
        normalized_message_key = message_key or f"sequence:{state.latest_sequence}"
        state.message_marks[normalized_message_key] = state.latest_sequence
        state.message_marks.move_to_end(normalized_message_key)
        while len(state.message_marks) > self._max_marks_per_session:
            state.message_marks.popitem(last=False)
        self._sessions[session_key] = state
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return RequestMark(state.latest_sequence, normalized_message_key)

    def was_interrupted(self, session_key: str, mark: RequestMark | None) -> bool:
        if mark is None:
            return False
        state = self._sessions.get(session_key)
        if state is None:
            return False
        self._sessions.move_to_end(session_key)
        return state.latest_sequence > mark.sequence

    def finish(self, session_key: str, mark: RequestMark | None) -> None:
        if mark is not None and session_key in self._sessions:
            self._sessions.move_to_end(session_key)

    def clear(self) -> None:
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

