import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from air_combat_challenge.competition.models import ActionBatchV1

from . import agent as agent_module
from . import decision_logger as logger_module
from . import llm_client
from .agent import QwenCommanderAgent
from .decision_logger import DecisionLogger
from .tactics import validate_plan_bundle


VALID_PLAN_BUNDLE = {
    "survival_plan": {
        "tactic": "defensive_regroup",
        "primary_target": "blue_fighter_01",
        "secondary_target": "blue_fighter_02",
        "role_red_1": "defend",
        "role_red_2": "cover",
        "duration_steps": 4,
        "risk_bias": "low",
        "brief": "红方保持互援并降低风险。",
    },
    "attack_plan": {
        "tactic": "focus_fire",
        "primary_target": "blue_fighter_01",
        "secondary_target": "blue_fighter_02",
        "role_red_1": "press",
        "role_red_2": "cover",
        "duration_steps": 4,
        "risk_bias": "high",
        "brief": "红方集中火力打击首要目标。",
    },
    "recommended_plan": "survival",
    "recommendation_brief": "红方互援不足，当前优先降低暴露风险。",
}


class PlanSelectionTests(unittest.TestCase):
    def test_bundle_requires_valid_llm_recommendation(self):
        self.assertEqual(validate_plan_bundle(VALID_PLAN_BUNDLE), (True, ""))

        missing_recommendation = dict(VALID_PLAN_BUNDLE)
        missing_recommendation.pop("recommended_plan")
        self.assertEqual(
            validate_plan_bundle(missing_recommendation),
            (False, "invalid_recommended_plan"),
        )

        unknown_recommendation = dict(VALID_PLAN_BUNDLE, recommended_plan="unknown")
        self.assertEqual(
            validate_plan_bundle(unknown_recommendation),
            (False, "invalid_recommended_plan"),
        )

    def test_user_and_llm_modes_select_different_cached_plans(self):
        agent = QwenCommanderAgent.__new__(QwenCommanderAgent)
        agent._cached_plan_bundle = VALID_PLAN_BUNDLE

        with patch.object(agent_module, "get_llm_plan_selection_mode", return_value="user"), patch.object(
            agent_module,
            "get_llm_plan_selection",
            return_value="attack",
        ):
            selected = agent._selected_plan()
        self.assertEqual(selected[0], "attack")
        self.assertEqual(selected[1]["tactic"], "focus_fire")
        self.assertEqual(selected[2:4], ("user", "user_config"))

        with patch.object(agent_module, "get_llm_plan_selection_mode", return_value="llm"):
            selected = agent._selected_plan()
        self.assertEqual(selected[0], "survival")
        self.assertEqual(selected[1]["tactic"], "defensive_regroup")
        self.assertEqual(selected[2:4], ("llm", "llm_recommendation"))

    def test_selection_config_does_not_change_llm_request(self):
        state_summary = {}
        base_config = {
            "model": "test-model",
            "temperature": 0.2,
            "max_tokens": 120,
            "llm_plan_selection": "attack",
        }
        with patch.object(
            llm_client,
            "_load_config",
            return_value=dict(base_config, llm_plan_selection_mode="user"),
        ):
            user_request = llm_client.build_chat_completion_request("prompt", state_summary)
        with patch.object(
            llm_client,
            "_load_config",
            return_value=dict(base_config, llm_plan_selection_mode="llm"),
        ):
            llm_request = llm_client.build_chat_completion_request("prompt", state_summary)
        self.assertEqual(user_request, llm_request)

    def test_llm_enabled_respects_explicit_enable_flag(self):
        with patch.object(
            llm_client,
            "_load_config",
            return_value={"enable_llm": True, "base_url": "https://example.test/v1"},
        ):
            self.assertTrue(llm_client.llm_enabled())

        with patch.object(
            llm_client,
            "_load_config",
            return_value={"enable_llm": False, "base_url": "https://example.test/v1"},
        ):
            self.assertFalse(llm_client.llm_enabled())

        with patch.object(
            llm_client,
            "_load_config",
            return_value={"base_url": "https://example.test/v1"},
        ):
            self.assertTrue(llm_client.llm_enabled())

    def test_disabled_llm_uses_config_fallback_without_cached_plan(self):
        agent = QwenCommanderAgent.__new__(QwenCommanderAgent)
        observation = SimpleNamespace(step_index=7, sim_time=52.0)
        batch = ActionBatchV1()
        agent._cached_plan_bundle = VALID_PLAN_BUNDLE
        agent._last_failure_reason = ""
        agent._last_launch_time = {}
        agent._bvr_controller = SimpleNamespace(metadata=lambda: {})
        agent._decision_logger = MagicMock()

        with patch.object(
            agent_module,
            "fallback_action_batch",
            return_value=batch,
        ), patch.object(
            agent_module,
            "translate_tactic_to_actions",
            side_effect=AssertionError("cached llm plan should not be used when disabled"),
        ), patch.object(
            QwenCommanderAgent,
            "_apply_bvr_overrides",
            return_value=batch,
        ):
            result = agent._action_from_cache_or_bootstrap(
                observation,
                enemy_contact_state={},
                missile_threat_state={},
                allow_cached_llm=False,
            )

        self.assertIs(result, batch)
        self.assertEqual(agent._last_failure_reason, "llm_disabled_by_config")
        self.assertEqual(agent._decision_logger.write_action_source.call_args.args[1], "config_fallback")
        self.assertEqual(
            agent._decision_logger.write_action_source.call_args.kwargs["contact_mode"],
            "config_fallback",
        )

    def test_act_skips_request_start_when_llm_disabled(self):
        agent = QwenCommanderAgent.__new__(QwenCommanderAgent)
        observation = SimpleNamespace(own_units=[object()], sim_time=15.0)
        enemy_contact_state = {"lost_contact_ids": []}
        missile_threat_state = {"active_threats": []}
        result_batch = ActionBatchV1()

        agent._force_status_tracker = SimpleNamespace(update=MagicMock(return_value={}))
        agent._enemy_contact_tracker = SimpleNamespace(update=MagicMock(return_value=enemy_contact_state))
        agent._missile_threat_tracker = SimpleNamespace(update=MagicMock(return_value=missile_threat_state))
        agent._collect_completed_request = MagicMock(return_value=False)
        agent._should_start_request = MagicMock(side_effect=AssertionError("should not start llm request"))
        agent._start_request = MagicMock(side_effect=AssertionError("llm request should stay disabled"))
        agent._action_from_cache_or_bootstrap = MagicMock(return_value=result_batch)
        agent._last_failure_reason = ""

        with patch.object(agent_module, "build_state_summary", return_value={}), patch.object(
            agent_module, "llm_enabled", return_value=False
        ), patch.object(agent_module, "get_bvr_config", return_value=SimpleNamespace(threat_timeout_s=110.0)):
            result = agent.act(observation)

        self.assertIs(result, result_batch)
        agent._action_from_cache_or_bootstrap.assert_called_once_with(
            observation,
            False,
            enemy_contact_state,
            missile_threat_state,
            allow_cached_llm=False,
        )

    def test_logs_record_recommendation_and_actual_selection(self):
        observation = SimpleNamespace(step_index=4, sim_time=34.0)
        result = {
            "ok": True,
            "assistant_content": "{}",
            "reasoning_content": None,
            "tactic": VALID_PLAN_BUNDLE,
        }
        with TemporaryDirectory() as temp_dir, patch.object(
            logger_module,
            "LOG_ROOT",
            Path(temp_dir),
        ):
            logger = DecisionLogger()
            logger.reset("episode", "red", "20260716_143015_482")
            call = logger.write_input(observation, {"model": "test-model"})
            logger.write_output(
                call,
                result,
                validation_ok=True,
                selected_plan="attack",
                selected_tactic=VALID_PLAN_BUNDLE["attack_plan"],
                selection_mode="user",
                selection_source="user_config",
            )
            logger.write_action_source(
                observation,
                "llm_cached",
                VALID_PLAN_BUNDLE["attack_plan"],
                call,
                "attack",
                "user",
                "user_config",
                "survival",
                VALID_PLAN_BUNDLE["recommendation_brief"],
                contact_mode="reacquire",
                lost_contact_ids=["blue_fighter_01"],
                last_contact_age_s=12.0,
                bvr_metadata={
                    "bvr_mode": {"red_fighter_01": "missile_defense"},
                    "active_missile_threats": [{"target_id": "red_fighter_01"}],
                    "threatened_platform_ids": ["red_fighter_01"],
                    "bvr_fired_actions": [],
                    "bvr_launch_envelopes": [],
                },
            )

            output = json.loads(
                (
                    Path(temp_dir)
                    / "20260716_143015_482"
                    / "red"
                    / "step_0004_call_0001.output.json"
                ).read_text(
                    encoding="utf-8"
                )
            )
            action = json.loads(
                (
                    Path(temp_dir)
                    / "20260716_143015_482"
                    / "red"
                    / "action_sources.jsonl"
                ).read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(output["episode_id"], "episode")
        self.assertEqual(output["log_timestamp"], "20260716_143015_482")
        self.assertEqual(output["side"], "red")
        self.assertEqual(output["model_recommended_plan"], "survival")
        self.assertEqual(output["selected_plan"], "attack")
        self.assertEqual(output["selection_source"], "user_config")
        self.assertEqual(action["model_recommended_plan"], "survival")
        self.assertEqual(action["episode_id"], "episode")
        self.assertEqual(action["log_timestamp"], "20260716_143015_482")
        self.assertEqual(action["side"], "red")
        self.assertEqual(action["selected_plan"], "attack")
        self.assertEqual(action["contact_mode"], "reacquire")
        self.assertEqual(action["lost_contact_ids"], ["blue_fighter_01"])
        self.assertEqual(action["bvr_mode"]["red_fighter_01"], "missile_defense")
        self.assertEqual(action["threatened_platform_ids"], ["red_fighter_01"])

    def test_loggers_partition_the_same_timestamp_by_side(self):
        observation = SimpleNamespace(step_index=0, sim_time=30.0)
        with TemporaryDirectory() as temp_dir, patch.object(
            logger_module,
            "LOG_ROOT",
            Path(temp_dir),
        ):
            red_logger = DecisionLogger()
            red_logger.reset("episode", "red", "20260716_143015_482")
            blue_logger = DecisionLogger()
            blue_logger.reset("episode", "blue", "20260716_143015_482")
            red_logger.write_action_source(observation, "config_fallback")
            blue_logger.write_action_source(observation, "config_fallback")

            red_path = Path(temp_dir) / "20260716_143015_482" / "red" / "action_sources.jsonl"
            blue_path = Path(temp_dir) / "20260716_143015_482" / "blue" / "action_sources.jsonl"
            red_record = json.loads(red_path.read_text(encoding="utf-8"))
            blue_record = json.loads(blue_path.read_text(encoding="utf-8"))

        self.assertEqual(red_record["side"], "red")
        self.assertEqual(blue_record["side"], "blue")
        self.assertEqual(red_record["log_timestamp"], blue_record["log_timestamp"])


if __name__ == "__main__":
    unittest.main()
