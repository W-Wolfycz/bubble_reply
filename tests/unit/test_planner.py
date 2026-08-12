from __future__ import annotations

import unittest

from bubble_reply.config import ComponentPolicy, MediaPolicyConfig
from bubble_reply.domain.models import ComponentToken
from bubble_reply.domain.planner import apply_segment_limit, plan_delivery


def media(
    image: ComponentPolicy = ComponentPolicy.EMBED,
) -> MediaPolicyConfig:
    return MediaPolicyConfig(image=image)


def plain_text(segment) -> str:
    return "".join(
        str(component.payload)
        for component in segment.components
        if component.kind == "Plain"
    )


class PlannerTests(unittest.TestCase):
    def test_plain_lines_leave_last_segment_to_respond(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", "a\nb\nc")],
            media=media(),
        )
        self.assertEqual(plan.mode, "segmented")
        self.assertEqual(len(plan.active_segments), 2)
        self.assertEqual(len(plan.respond_segments), 1)

    def test_segment_whitespace_is_stripped_and_empty_lines_are_removed(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", "  a  \n   \n\tb\t\n  ")],
            media=media(),
            strip_segment_whitespace=True,
        )
        self.assertEqual(
            [plain_text(segment) for segment in plan.all_segments],
            ["a", "b"],
        )

    def test_segment_whitespace_can_be_preserved(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", "  a  \n   \n b ")],
            media=media(),
            strip_segment_whitespace=False,
        )
        self.assertEqual(
            [plain_text(segment) for segment in plan.all_segments],
            ["  a  ", " b "],
        )

    def test_whitespace_exposed_by_tail_cleanup_is_removed(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", " a 。  \n 。 ")],
            media=media(),
            strip_segment_tail_chars="。",
            strip_segment_whitespace=True,
        )
        self.assertEqual(
            [plain_text(segment) for segment in plan.all_segments],
            ["a"],
        )

    def test_mixed_component_whitespace_only_cleans_segment_edges(self) -> None:
        at = object()
        plan = plan_delivery(
            [
                ComponentToken("Plain", "  a  "),
                ComponentToken("At", at),
                ComponentToken("Plain", "  b  "),
            ],
            media=media(),
            strip_segment_whitespace=True,
        )
        components = plan.all_segments[0].components
        self.assertEqual(
            [
                str(component.payload) if component.kind == "Plain" else component.kind
                for component in components
            ],
            ["a  ", "At", "  b"],
        )

    def test_empty_trailing_plain_continues_cleanup_inward(self) -> None:
        image = object()
        plan = plan_delivery(
            [
                ComponentToken("Plain", "  a。  "),
                ComponentToken("Image", image),
                ComponentToken("Plain", " 。 "),
            ],
            media=media(),
            strip_segment_tail_chars="。",
            strip_segment_whitespace=True,
        )
        components = plan.all_segments[0].components
        self.assertEqual(
            [
                str(component.payload) if component.kind == "Plain" else component.kind
                for component in components
            ],
            ["a", "Image"],
        )

    def test_unsafe_component_in_active_range_downgrades_all(self) -> None:
        plan = plan_delivery(
            [
                ComponentToken("Plain", "a\n"),
                ComponentToken("File", object()),
                ComponentToken("Plain", "\nb"),
            ],
            media=media(),
        )
        self.assertEqual(plan.mode, "respond_only")
        self.assertFalse(plan.active_segments)
        self.assertEqual(plan.reason, "unsafe_active_component")

    def test_unsafe_last_segment_may_remain_with_respond(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", "a\n"), ComponentToken("Record", object())],
            media=media(),
        )
        self.assertEqual(plan.mode, "segmented")
        self.assertEqual(plan.active_segments[0].component_kinds, ("Plain",))
        self.assertEqual(plan.respond_segments[0].component_kinds, ("Record",))

    def test_at_only_tail_is_active(self) -> None:
        plan = plan_delivery(
            [ComponentToken("Plain", "a\n"), ComponentToken("At", object())],
            media=media(),
        )
        self.assertEqual(len(plan.active_segments), 2)
        self.assertFalse(plan.respond_segments)

    def test_at_and_face_always_keep_original_position(self) -> None:
        at = object()
        face = object()
        plan = plan_delivery(
            [
                ComponentToken("Plain", "a"),
                ComponentToken("At", at),
                ComponentToken("Face", face),
                ComponentToken("Plain", "b"),
            ],
            media=media(image=ComponentPolicy.SEPARATE),
        )
        self.assertEqual(
            plan.all_segments[0].component_kinds,
            ("Plain", "At", "Face", "Plain"),
        )

    def test_separate_image_preserves_order(self) -> None:
        image = object()
        plan = plan_delivery(
            [
                ComponentToken("Plain", "a"),
                ComponentToken("Image", image),
                ComponentToken("Plain", "b"),
            ],
            media=media(image=ComponentPolicy.SEPARATE),
        )
        kinds = [segment.component_kinds for segment in plan.all_segments]
        self.assertEqual(kinds, [("Plain",), ("Image",), ("Plain",)])

    def test_auto_reply_is_added_once(self) -> None:
        reply = ComponentToken("Reply", object())
        plan = plan_delivery(
            [ComponentToken("Plain", "a\nb")],
            media=media(),
            should_add_reply=True,
            reply_token=reply,
        )
        self.assertEqual(plan.active_segments[0].component_kinds, ("Reply", "Plain"))
        self.assertIs(plan.active_segments[0].auto_reply_component, reply)

    def test_existing_reply_is_not_duplicated(self) -> None:
        existing = ComponentToken("Reply", object())
        plan = plan_delivery(
            [existing, ComponentToken("Plain", "a\nb")],
            media=media(),
            should_add_reply=True,
            reply_token=ComponentToken("Reply", object()),
        )
        reply_count = sum(
            component.kind == "Reply"
            for segment in plan.all_segments
            for component in segment.components
        )
        self.assertEqual(reply_count, 1)

    def test_segment_limit_falls_back_to_baseline_plan(self) -> None:
        baseline = plan_delivery([ComponentToken("Plain", "a\nb")], media=media())
        candidate = plan_delivery(
            [ComponentToken("Plain", "a\nb\nc")],
            media=media(),
        )
        selected = apply_segment_limit(candidate, baseline, 2)
        self.assertEqual(selected.reason, "segment_limit_fallback")
        self.assertEqual(len(selected.all_segments), 2)
        self.assertEqual(selected.mode, "segmented")

    def test_segment_limit_falls_back_to_one_respond_result_when_both_exceed(self) -> None:
        baseline = plan_delivery(
            [ComponentToken("Plain", "a\nb\nc")],
            media=media(),
        )
        candidate = plan_delivery(
            [ComponentToken("Plain", "a\nb\nc\nd")],
            media=media(),
        )
        selected = apply_segment_limit(candidate, baseline, 2)
        self.assertEqual(selected.reason, "segment_limit")
        self.assertEqual(selected.mode, "respond_only")
        self.assertFalse(selected.active_segments)
        self.assertEqual(len(selected.respond_segments), 3)

    def test_unlimited_segment_limit_keeps_candidate_plan(self) -> None:
        baseline = plan_delivery([ComponentToken("Plain", "a")], media=media())
        candidate = plan_delivery([ComponentToken("Plain", "a\nb")], media=media())
        self.assertIs(apply_segment_limit(candidate, baseline, 0), candidate)


if __name__ == "__main__":
    unittest.main()
