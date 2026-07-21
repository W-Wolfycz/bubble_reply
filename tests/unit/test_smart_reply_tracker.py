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


if __name__ == "__main__":
    unittest.main()
