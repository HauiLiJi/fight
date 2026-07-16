from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1

from .bvr import BvrController, get_bvr_config
from .decision_logger import DecisionLogger
from .enemy_contact import EnemyContactTracker
from .fallback_rules import fallback_action_batch
from .force_status import RedForceStatusTracker
from .llm_client import (
    build_chat_completion_request,
    get_llm_plan_selection,
    get_llm_plan_selection_mode,
    get_llm_trigger_mode,
    get_min_call_interval_sim_s,
    llm_enabled,
    request_tactic,
)
from .missile_threat import MissileThreatTracker
from .state_summary import build_state_summary
from .tactics import (
    PLAN_KEYS,
    contact_mode_for_tactic,
    translate_tactic_to_actions,
    validate_plan_bundle,
)


MIN_REMAINING_BUDGET_S = 0.45
STEP_TIMEOUT_S = 35.0


class QwenCommanderAgent(BaseAgent):
    def __init__(self):
        self._last_launch_time = {}
        self._cached_plan_bundle = None
        self._cached_tactic_signature = None
        self._cached_tactic_source_call = None
        self._last_requested_signature = None
        self._last_request_sim_time = None
        self._last_failure_reason = ""
        self._prompt_text = _load_prompt()
        self._decision_logger = DecisionLogger()
        self._force_status_tracker = RedForceStatusTracker()
        self._enemy_contact_tracker = EnemyContactTracker()
        self._missile_threat_tracker = MissileThreatTracker()
        self._bvr_controller = BvrController()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="red-llm")
        self._pending_request = None
        self._generation = 0

    def reset(self, context):
        super().reset(context)
        self._generation += 1
        self._last_launch_time.clear()
        self._cached_plan_bundle = None
        self._cached_tactic_signature = None
        self._cached_tactic_source_call = None
        self._last_requested_signature = None
        self._last_request_sim_time = None
        self._last_failure_reason = ""
        self._force_status_tracker.reset(context.controlled_platform_ids)
        self._enemy_contact_tracker.reset()
        self._missile_threat_tracker.reset()
        self._bvr_controller.reset()
        # Keep a previous request reachable so its late response is logged as stale.
        self._decision_logger = DecisionLogger()
        self._decision_logger.reset(
            context.episode_id,
            context.side,
            getattr(context, "log_timestamp", None),
        )

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
        super().close()

    def act(self, observation):
        step_started_at = time.perf_counter()
        if not observation.own_units:
            return ActionBatchV1()

        try:
            red_force_status = self._force_status_tracker.update(observation)
            enemy_contact_state = self._enemy_contact_tracker.update(observation)
            missile_threat_state = self._missile_threat_tracker.update(
                observation,
                get_bvr_config().threat_timeout_s,
            )
            state_summary = build_state_summary(
                observation,
                red_force_status,
                enemy_contact_state,
                missile_threat_state,
            )
        except Exception as error:
            self._last_failure_reason = f"state_summary_error:{type(error).__name__}"
            return self._action_from_cache_or_bootstrap(
                observation,
                enemy_contact_state=None,
                missile_threat_state=None,
            )

        tactical_signature = _tactical_signature(state_summary)
        enabled = llm_enabled()
        applied = self._collect_completed_request(tactical_signature, enabled)

        if enabled and self._should_start_request(
            tactical_signature,
            observation.sim_time,
            get_min_call_interval_sim_s(),
            get_llm_trigger_mode(),
        ):
            elapsed = time.perf_counter() - step_started_at
            if STEP_TIMEOUT_S - elapsed < MIN_REMAINING_BUDGET_S:
                self._last_failure_reason = "insufficient_time_budget"
            else:
                self._start_request(observation, state_summary, tactical_signature)

        return self._action_from_cache_or_bootstrap(
            observation,
            applied,
            enemy_contact_state,
            missile_threat_state,
            allow_cached_llm=enabled,
        )

    def _action_from_cache_or_bootstrap(
        self,
        observation,
        applied=False,
        enemy_contact_state=None,
        missile_threat_state=None,
        allow_cached_llm=True,
    ):
        if allow_cached_llm and self._cached_plan_bundle is not None:
            action_source = "llm_applied" if applied else "llm_cached"
            (
                selected_plan,
                tactic_payload,
                selection_mode,
                selection_source,
                model_recommended_plan,
                recommendation_brief,
            ) = self._selected_plan()
            contact_mode = contact_mode_for_tactic(
                observation,
                tactic_payload,
                enemy_contact_state,
            )
            try:
                batch = translate_tactic_to_actions(
                    observation,
                    tactic_payload,
                    self._last_launch_time,
                    enemy_contact_state,
                    self._bvr_controller,
                )
                batch = self._apply_bvr_overrides(
                    observation,
                    batch,
                    missile_threat_state,
                )
            except Exception as error:
                self._record_failure(f"translate_error:{type(error).__name__}")
                self._decision_logger.write_action_source(
                    observation,
                    action_source,
                    tactic_payload,
                    self._cached_tactic_source_call,
                    selected_plan,
                    selection_mode,
                    selection_source,
                    model_recommended_plan,
                    recommendation_brief,
                    contact_mode=contact_mode,
                    lost_contact_ids=(enemy_contact_state or {}).get("lost_contact_ids", []),
                    last_contact_age_s=(enemy_contact_state or {}).get("last_contact_age_s"),
                    bvr_metadata=self._bvr_controller.metadata(),
                )
                return ActionBatchV1()
            self._decision_logger.write_action_source(
                observation,
                action_source,
                tactic_payload,
                self._cached_tactic_source_call,
                selected_plan,
                selection_mode,
                selection_source,
                model_recommended_plan,
                recommendation_brief,
                contact_mode=contact_mode,
                lost_contact_ids=(enemy_contact_state or {}).get("lost_contact_ids", []),
                last_contact_age_s=(enemy_contact_state or {}).get("last_contact_age_s"),
                bvr_metadata=self._bvr_controller.metadata(),
            )
            return batch

        if allow_cached_llm:
            self._last_failure_reason = "bootstrap_fallback_waiting_for_first_llm"
            action_source = "bootstrap_fallback"
            contact_mode = "bootstrap_fallback"
        else:
            self._last_failure_reason = "llm_disabled_by_config"
            action_source = "config_fallback"
            contact_mode = "config_fallback"
        batch = fallback_action_batch(
            observation,
            self._last_launch_time,
            self._bvr_controller,
        )
        batch = self._apply_bvr_overrides(observation, batch, missile_threat_state)
        self._decision_logger.write_action_source(
            observation,
            action_source,
            contact_mode=contact_mode,
            lost_contact_ids=(enemy_contact_state or {}).get("lost_contact_ids", []),
            last_contact_age_s=(enemy_contact_state or {}).get("last_contact_age_s"),
            bvr_metadata=self._bvr_controller.metadata(),
        )
        return batch

    def _apply_bvr_overrides(self, observation, batch, missile_threat_state):
        actions = [
            action.model_dump() if hasattr(action, "model_dump") else dict(action)
            for action in batch.actions
        ]
        return ActionBatchV1.model_validate(
            {
                "actions": self._bvr_controller.apply(
                    observation,
                    actions,
                    self._last_launch_time,
                    missile_threat_state,
                )
            }
        )

    def _should_start_request(self, tactical_signature, sim_time, interval_s, trigger_mode):
        if self._pending_request is not None:
            return False
        if self._last_request_sim_time is None:
            return True
        if trigger_mode == "default":
            return sim_time - self._last_request_sim_time >= interval_s
        return tactical_signature != self._last_requested_signature

    def _start_request(self, observation, state_summary, tactical_signature):
        request_payload = build_chat_completion_request(self._prompt_text, state_summary)
        call = self._decision_logger.write_input(observation, request_payload)
        self._last_requested_signature = tactical_signature
        self._last_request_sim_time = observation.sim_time
        try:
            future = self._executor.submit(
                request_tactic,
                self._prompt_text,
                state_summary,
                request_payload=request_payload,
            )
        except RuntimeError as error:
            result = {"ok": False, "error": "submit_failed", "message": str(error)}
            self._decision_logger.write_output(
                call,
                result,
                validation_reason="submit_failed",
                fallback_used=True,
                request_status="request_failed",
            )
            self._record_failure("submit_failed")
            return
        self._pending_request = {
            "future": future,
            "signature": tactical_signature,
            "generation": self._generation,
            "call": call,
            "logger": self._decision_logger,
        }

    def _collect_completed_request(self, tactical_signature, llm_is_enabled):
        pending = self._pending_request
        if pending is None or not pending["future"].done():
            return False
        self._pending_request = None
        try:
            result = pending["future"].result()
        except Exception as error:
            result = {
                "ok": False,
                "error": "background_exception",
                "message": f"{type(error).__name__}: {error}",
            }

        if pending["generation"] != self._generation or not llm_is_enabled:
            pending["logger"].write_output(
                pending["call"],
                result,
                validation_reason="stale_episode_or_disabled",
                fallback_used=True,
                request_status="stale_discarded",
            )
            return False
        if pending["signature"] != tactical_signature:
            pending["logger"].write_output(
                pending["call"],
                result,
                validation_reason="stale_tactical_signature",
                fallback_used=True,
                request_status="stale_discarded",
            )
            return False
        if not result.get("ok"):
            pending["logger"].write_output(
                pending["call"],
                result,
                validation_reason=result.get("error", "llm_error"),
                fallback_used=True,
                request_status="request_failed",
            )
            self._record_failure(result.get("error", "llm_error"))
            return False

        plan_bundle = result["tactic"]
        is_valid, reason = validate_plan_bundle(plan_bundle)
        selected_plan = None
        selected_tactic = None
        selection_mode = None
        selection_source = None
        if is_valid:
            (
                selected_plan,
                selected_tactic,
                selection_mode,
                selection_source,
                _,
                _,
            ) = self._selected_plan(plan_bundle)
        pending["logger"].write_output(
            pending["call"],
            result,
            validation_ok=is_valid,
            validation_reason=reason,
            fallback_used=not is_valid,
            request_status="applied" if is_valid else "validation_failed",
            selected_plan=selected_plan if is_valid else None,
            selected_tactic=selected_tactic if is_valid else None,
            selection_mode=selection_mode if is_valid else None,
            selection_source=selection_source if is_valid else None,
        )
        if not is_valid:
            self._record_failure(reason)
            return False
        self._record_success(plan_bundle, pending["signature"], pending["call"])
        return True

    def _record_failure(self, reason):
        self._last_failure_reason = reason

    def _record_success(self, plan_bundle, tactical_signature, source_call):
        self._last_failure_reason = ""
        self._cached_plan_bundle = plan_bundle
        self._cached_tactic_signature = tactical_signature
        self._cached_tactic_source_call = source_call

    def _selected_plan(self, plan_bundle=None):
        plan_bundle = self._cached_plan_bundle if plan_bundle is None else plan_bundle
        selection_mode = get_llm_plan_selection_mode()
        if selection_mode == "llm":
            selection = plan_bundle["recommended_plan"]
            selection_source = "llm_recommendation"
        else:
            selection = get_llm_plan_selection()
            selection_source = "user_config"
        return (
            selection,
            plan_bundle[PLAN_KEYS[selection]],
            selection_mode,
            selection_source,
            plan_bundle["recommended_plan"],
            plan_bundle["recommendation_brief"],
        )


def _risk_bucket_from_summary(state_summary):
    units = state_summary.get("red_units", [])
    highest = max((unit.get("risk_score", 0.0) for unit in units), default=0.0)
    if highest >= 5.0:
        return "high"
    if highest >= 3.0:
        return "medium"
    return "low"


def _tactical_signature(state_summary):
    threat_eval = state_summary.get("threat_eval", {})
    pair_geometry = state_summary.get("pair_geometry", {})
    force_status = state_summary.get("red_force_status", {})
    contact_status = state_summary.get("blue_contact_status", {})
    missile_threats = state_summary.get("incoming_missile_threats", {})
    return (
        tuple(sorted(unit.get("platform_id", "") for unit in state_summary.get("red_units", []))),
        tuple(sorted(track.get("target_id", "") for track in state_summary.get("blue_tracks", []))),
        state_summary.get("engagement_phase", "regroup"),
        _risk_bucket_from_summary(state_summary),
        threat_eval.get("most_threatened_red", ""),
        bool(threat_eval.get("double_pressure", False)),
        bool(pair_geometry.get("mutual_support", False)),
        tuple(
            sorted(
                (unit.get("platform_id", ""), unit.get("status", ""))
                for unit in force_status.get("units", [])
            )
        ),
        tuple(contact_status.get("visible_ids", [])),
        tuple(contact_status.get("lost_contact_ids", [])),
        tuple(
            sorted(
                (item.get("shooter_id", ""), item.get("target_id", ""))
                for item in missile_threats.get("active_threats", [])
            )
        ),
    )


def _load_prompt():
    prompt_path = Path(__file__).with_name("prompt.txt")
    return prompt_path.read_text(encoding="utf-8")
