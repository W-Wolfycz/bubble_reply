from __future__ import annotations

import unittest

from bubble_reply.services.smart_reply_tracker import SmartReplyTracker


class SmartReplyTrackerTests(unittest.TestCase):
    def test_later_request_marks_previous_as_interrupted(self) -> None:
        tracker = SmartReplyTracker()
        first = tracker.begin("umo-demo", "message-1")
        second = tracker.begin("umo-demo", "message-2")
        self.assertTrue(tracker.was_interrupted("umo-demo", first))
        self.assertFalse(tracker.was_interrupted("umo-demo", second))

    def test_session_capacity_is_bounded(self) -> None:
        tracker = SmartReplyTracker(max_sessions=2)
        tracker.begin("umo-1", "m1")
        tracker.begin("umo-2", "m2")
        tracker.begin("umo-3", "m3")
        self.assertEqual(tracker.session_count, 2)

    def test_clear_removes_all_state(self) -> None:
        tracker = SmartReplyTracker()
        tracker.begin("umo-demo", "message-1")
        tracker.clear()
        self.assertEqual(tracker.session_count, 0)

    def test_note_outgoing_marks_interleaved_bot_reply_as_interruption(self) -> None:
        tracker = SmartReplyTracker()
        first = tracker.begin("umo-demo", "message-1")
        second = tracker.begin("umo-demo", "message-2")
        self.assertFalse(tracker.was_interrupted("umo-demo", second))
        # bot 先回复了 first（自身回复不会经过入站观察器），应让 second 判定被打断。
        tracker.note_outgoing("umo-demo")
        self.assertTrue(tracker.was_interrupted("umo-demo", second))

    def test_note_outgoing_is_noop_without_session(self) -> None:
        tracker = SmartReplyTracker()
        tracker.note_outgoing("umo-unknown")
        self.assertEqual(tracker.session_count, 0)

    def test_note_outgoing_does_not_interrupt_later_request(self) -> None:
        tracker = SmartReplyTracker()
        first = tracker.begin("umo-demo", "message-1")
        tracker.note_outgoing("umo-demo")
        third = tracker.begin("umo-demo", "message-3")
        # first 被后续消息与自身回复打断；third 之后没有新消息，不应被打断。
        self.assertTrue(tracker.was_interrupted("umo-demo", first))
        self.assertFalse(tracker.was_interrupted("umo-demo", third))

    def test_noise_within_threshold_does_not_interrupt(self) -> None:
        tracker = SmartReplyTracker(noise_threshold=5)
        mark = tracker.begin("umo-demo", "message-1")
        for _ in range(5):
            tracker.note_noise("umo-demo")
        self.assertFalse(tracker.was_interrupted("umo-demo", mark))

    def test_noise_exceeding_threshold_interrupts(self) -> None:
        tracker = SmartReplyTracker(noise_threshold=5)
        mark = tracker.begin("umo-demo", "message-1")
        for _ in range(6):
            tracker.note_noise("umo-demo")
        self.assertTrue(tracker.was_interrupted("umo-demo", mark))

    def test_noise_before_later_message_does_not_affect_later_request(self) -> None:
        tracker = SmartReplyTracker(noise_threshold=5)
        first = tracker.begin("umo-demo", "message-1")
        for _ in range(6):
            tracker.note_noise("umo-demo")
        second = tracker.begin("umo-demo", "message-2")
        # first 被噪声超阈值打断；second 之后没有噪声，不应被打断。
        self.assertTrue(tracker.was_interrupted("umo-demo", first))
        self.assertFalse(tracker.was_interrupted("umo-demo", second))

    def test_note_noise_is_noop_without_session(self) -> None:
        tracker = SmartReplyTracker()
        tracker.note_noise("umo-unknown")
        self.assertEqual(tracker.session_count, 0)


if __name__ == "__main__":
    unittest.main()
