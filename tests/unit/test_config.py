from __future__ import annotations

import unittest

from bubble_reply.config import (
    ComponentPolicy,
    DelayMode,
    SplitScope,
    load_runtime_config,
    migrate_log_config_inplace,
)


class ConfigTests(unittest.TestCase):
    def test_unknown_enums_fall_back_and_warn(self) -> None:
        warnings: list[str] = []
        config = load_runtime_config(
            {
                "basic_settings": {"split_scope": "future"},
                "quote_media_settings": {"image_strategy": "future"},
            },
            warnings.append,
        )
        self.assertEqual(config.scope, SplitScope.LLM_ONLY)
        self.assertEqual(config.media.image, ComponentPolicy.EMBED)
        self.assertEqual(len(warnings), 2)

    def test_numeric_values_are_finite_and_non_negative(self) -> None:
        config = load_runtime_config(
            {
                "basic_settings": {"max_length_to_disable": -5},
                "delay_settings": {
                    "delay_seconds": -2,
                    "seconds_per_character": -1,
                    "random_jitter_seconds": -3,
                },
            }
        )
        self.assertEqual(config.max_length_to_disable, 0)
        self.assertEqual(config.delay.seconds, 0)
        self.assertEqual(config.delay.seconds_per_character, 0)
        self.assertEqual(config.delay.jitter_seconds, 0)

    def test_delay_defaults_preserve_fixed_interval_behavior(self) -> None:
        delay = load_runtime_config({}).delay
        self.assertEqual(delay.mode, DelayMode.FIXED)
        self.assertEqual(delay.seconds, 0.8)
        self.assertEqual(delay.seconds_per_character, 0.025)
        self.assertEqual(delay.minimum_seconds, 0.5)
        self.assertEqual(delay.maximum_seconds, 2.5)
        self.assertEqual(delay.jitter_seconds, 0)

    def test_segment_limit_defaults_to_unlimited(self) -> None:
        self.assertEqual(load_runtime_config({}).max_segments_to_disable, 0)
        config = load_runtime_config(
            {"basic_settings": {"max_segments_to_disable": 6}}
        )
        self.assertEqual(config.max_segments_to_disable, 6)

    def test_render_fallback_defaults_off(self) -> None:
        self.assertFalse(load_runtime_config({}).render_fallback_to_image)
        config = load_runtime_config(
            {"basic_settings": {"render_fallback_to_image": True}}
        )
        self.assertTrue(config.render_fallback_to_image)

    def test_delay_bounds_are_ordered_and_warn(self) -> None:
        warnings: list[str] = []
        delay = load_runtime_config(
            {
                "delay_settings": {
                    "delay_mode": "按字数",
                    "minimum_delay_seconds": 2,
                    "maximum_delay_seconds": 1,
                }
            },
            warnings.append,
        ).delay
        self.assertEqual(delay.mode, DelayMode.PER_CHARACTER)
        self.assertEqual(delay.minimum_seconds, 2)
        self.assertEqual(delay.maximum_seconds, 2)
        self.assertEqual(warnings, ["delay_settings.maximum_delay_seconds"])

    def test_llm_text_changes_are_opt_in(self) -> None:
        self.assertFalse(load_runtime_config({}).llm.allow_text_changes)
        config = load_runtime_config(
            {"llm_split_settings": {"allow_llm_text_changes": True}}
        )
        self.assertTrue(config.llm.allow_text_changes)

    def test_segment_whitespace_cleanup_defaults_to_enabled(self) -> None:
        self.assertTrue(load_runtime_config({}).text_rules.strip_segment_whitespace)
        config = load_runtime_config(
            {"split_settings": {"strip_segment_whitespace": False}}
        )
        self.assertFalse(config.text_rules.strip_segment_whitespace)

    def test_face_and_at_strategy_legacy_values_are_ignored(self) -> None:
        config = load_runtime_config(
            {
                "quote_media_settings": {
                    "image_strategy": "跟随下一段",
                    "face_strategy": "单独发送",
                    "at_strategy": "跟随上一段",
                }
            }
        )
        self.assertEqual(config.media.image, ComponentPolicy.FOLLOW_NEXT)
        self.assertEqual(config.media.for_kind("Face"), ComponentPolicy.EMBED)
        self.assertEqual(config.media.for_kind("At"), ComponentPolicy.EMBED)

    def test_invalid_regex_is_removed(self) -> None:
        warnings: list[str] = []
        config = load_runtime_config(
            {
                "llm_split_settings": {
                    "remove_before_split_regex": ["[", "ok"]
                }
            },
            warnings.append,
        )
        self.assertEqual(config.llm.remove_before_split_regex, ("ok",))
        self.assertEqual(len(warnings), 1)

    def test_blacklist_uses_group_and_friend_prefixes(self) -> None:
        warnings: list[str] = []
        config = load_runtime_config(
            {
                "basic_settings": {
                    "blacklist": ["G:10001", "f:10002", "invalid"]
                }
            },
            warnings.append,
        )
        self.assertEqual(config.blacklist.group_ids, frozenset({"10001"}))
        self.assertEqual(config.blacklist.friend_ids, frozenset({"10002"}))
        self.assertEqual(warnings, ["basic_settings.blacklist"])

    def test_missing_text_placeholder_is_added(self) -> None:
        warnings: list[str] = []
        config = load_runtime_config(
            {"llm_split_settings": {"llm_split_prompt": "请自然分段"}},
            warnings.append,
        )
        self.assertTrue(config.llm.runtime_rule.endswith("{text}"))
        self.assertEqual(warnings, ["llm_split_settings.llm_split_prompt"])

    def test_logging_defaults_are_flattened_and_debug_switch_removed(self) -> None:
        logging = load_runtime_config({}).logging
        self.assertFalse(logging.log_with_bot_id)
        self.assertFalse(logging.log_original_text)
        self.assertFalse(hasattr(logging, "debug_to_info"))

    def test_logging_reads_top_level_keys(self) -> None:
        logging = load_runtime_config(
            {"log_with_bot_id": True, "log_original_text": True}
        ).logging
        self.assertTrue(logging.log_with_bot_id)
        self.assertTrue(logging.log_original_text)

    def test_logging_legacy_group_overrides_top_level(self) -> None:
        # 升级后旧组仍存在时，其值为旧用户意图，优先于顶层键。
        logging = load_runtime_config(
            {
                "log_with_bot_id": False,
                "log_original_text": False,
                "log_config": {
                    "log_with_bot_id": True,
                    "log_original_text": True,
                },
            }
        ).logging
        self.assertTrue(logging.log_with_bot_id)
        self.assertTrue(logging.log_original_text)

    def test_logging_legacy_partial_key_falls_back_to_top_level(self) -> None:
        logging = load_runtime_config(
            {
                "log_with_bot_id": True,
                "log_config": {"log_original_text": True},
            }
        ).logging
        # 旧组缺失的键不覆盖顶层值。
        self.assertTrue(logging.log_with_bot_id)
        self.assertTrue(logging.log_original_text)

    def test_migrate_log_config_moves_values_and_removes_group(self) -> None:
        raw = {
            "log_with_bot_id": False,
            "log_original_text": False,
            "log_config": {
                "log_with_bot_id": True,
                "log_original_text": True,
                "debug_to_info": True,
            },
        }
        self.assertTrue(migrate_log_config_inplace(raw))
        self.assertEqual(raw["log_with_bot_id"], True)
        self.assertEqual(raw["log_original_text"], True)
        self.assertNotIn("log_config", raw)
        self.assertNotIn("debug_to_info", raw)

    def test_migrate_log_config_is_noop_without_legacy_group(self) -> None:
        raw = {"log_with_bot_id": False, "log_original_text": False}
        self.assertFalse(migrate_log_config_inplace(raw))
        self.assertEqual(raw, {"log_with_bot_id": False, "log_original_text": False})

    def test_migrate_log_config_preserves_top_level_for_missing_legacy_key(self) -> None:
        raw = {
            "log_with_bot_id": True,
            "log_config": {"log_original_text": True},
        }
        self.assertTrue(migrate_log_config_inplace(raw))
        self.assertEqual(raw["log_with_bot_id"], True)
        self.assertEqual(raw["log_original_text"], True)
        self.assertNotIn("log_config", raw)


if __name__ == "__main__":
    unittest.main()
