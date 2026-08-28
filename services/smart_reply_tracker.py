from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequestMark:
    sequence: int
    message_key: str
    noise_count: int = 0


@dataclass
class _SessionState:
    latest_sequence: int = 0
    noise_count: int = 0
    message_marks: OrderedDict[str, int] = field(default_factory=OrderedDict)


class SmartReplyTracker:
    def __init__(
        self,
        *,
        max_sessions: int = 1000,
        max_marks_per_session: int = 200,
        noise_threshold: int = 5,
    ) -> None:
        self._max_sessions = max(1, max_sessions)
        self._max_marks_per_session = max(1, max_marks_per_session)
        self._noise_threshold = max(0, noise_threshold)
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
        return RequestMark(
            state.latest_sequence,
            normalized_message_key,
            state.noise_count,
        )

    def note_noise(self, session_key: str) -> None:
        """记录一次非对话消息（系统事件 / 戳一戳等），不推进打断序号。

        这类消息本身不会造成回复指向歧义，因此默认不视为打断；仅当同会话
        自某条真实消息之后累积超过 ``noise_threshold`` 条时，才在
        ``was_interrupted`` 中视作打断（会话已被大量系统活动冲散）。
        """
        state = self._sessions.get(session_key)
        if state is None:
            return
        state.noise_count += 1
        self._sessions.move_to_end(session_key)

    def was_interrupted(self, session_key: str, mark: RequestMark | None) -> bool:
        if mark is None:
            return False
        state = self._sessions.get(session_key)
        if state is None:
            return False
        self._sessions.move_to_end(session_key)
        if state.latest_sequence > mark.sequence:
            return True
        return state.noise_count - mark.noise_count > self._noise_threshold

    def note_outgoing(self, session_key: str) -> None:
        """记录一次本会话内的插件自身回复（bot 消息），推进打断判定序号。

        插件直接发送的回复不会经过入站观察器（除非平台回送），这里显式推进
        序号，使后续同会话请求能识别“中间插入了 bot 自己的回复”而带上引用。
        """
        state = self._sessions.get(session_key)
        if state is None:
            return
        state.latest_sequence += 1
        self._sessions.move_to_end(session_key)

    def finish(self, session_key: str, mark: RequestMark | None) -> None:
        if mark is not None and session_key in self._sessions:
            self._sessions.move_to_end(session_key)

    def clear(self) -> None:
        self._sessions.clear()

    @property
    def session_count(self) -> int:
        return len(self._sessions)

