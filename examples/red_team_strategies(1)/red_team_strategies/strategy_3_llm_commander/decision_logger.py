"""Best-effort, per-request decision logs for the local LLM commander."""

import json
from pathlib import Path


LOG_ROOT = Path(__file__).resolve().parents[3] / "runs" / "llm_decisions"


class DecisionLogger:
    def __init__(self):
        self._episode_id = None
        self._log_timestamp = None
        self._side = None
        self._call_index = 0

    def reset(self, episode_id, side="unknown", log_timestamp=None):
        self._episode_id = str(episode_id)
        self._log_timestamp = str(log_timestamp or episode_id)
        self._side = str(side)
        self._call_index = 0

    def write_input(self, observation, request_payload):
        self._call_index += 1
        record = self._record_context(observation)
        record["request"] = request_payload
        path = self._path_for(observation.step_index, "input")
        self._write_json(path, record)
        return {
            "call_index": self._call_index,
            "step_index": observation.step_index,
            "sim_time": observation.sim_time,
        }

    def write_output(
        self,
        call,
        result,
        validation_ok=None,
        validation_reason="",
        fallback_used=False,
        request_status="completed",
        selected_plan=None,
        selected_tactic=None,
        selection_mode=None,
        selection_source=None,
    ):
        generated_plans = result.get("tactic") if isinstance(result, dict) else None
        model_recommended_plan = (
            generated_plans.get("recommended_plan")
            if isinstance(generated_plans, dict)
            else None
        )
        recommendation_brief = (
            generated_plans.get("recommendation_brief")
            if isinstance(generated_plans, dict)
            else None
        )
        record = {
            "api_version": "1.0",
            "episode_id": self._episode_id,
            "log_timestamp": self._log_timestamp,
            "side": self._side,
            "step_index": call["step_index"],
            "sim_time": call["sim_time"],
            "call_index": call["call_index"],
            "model_reply": {
                "assistant_content": result.get("assistant_content"),
                "reasoning_content": result.get("reasoning_content"),
            },
            "generated_plans": generated_plans,
            "selection_mode": selection_mode,
            "selection_source": selection_source,
            "model_recommended_plan": model_recommended_plan,
            "recommendation_brief": recommendation_brief,
            "selected_plan": selected_plan,
            "selected_tactic": selected_tactic,
            "tactic_brief": (
                selected_tactic.get("brief") if isinstance(selected_tactic, dict) else None
            ),
            "request_status": request_status,
            "result": result,
            "validation": {
                "ok": validation_ok,
                "reason": validation_reason,
            },
            "fallback_used": fallback_used,
        }
        path = self._path_for(call["step_index"], "output", call["call_index"])
        self._write_json(path, record)

    def write_action_source(
        self,
        observation,
        action_source,
        tactic=None,
        source_call=None,
        selected_plan=None,
        selection_mode=None,
        selection_source=None,
        model_recommended_plan=None,
        recommendation_brief=None,
        contact_mode=None,
        lost_contact_ids=None,
        last_contact_age_s=None,
        bvr_metadata=None,
    ):
        record = {
            "api_version": "1.0",
            "episode_id": self._episode_id,
            "log_timestamp": self._log_timestamp,
            "side": self._side,
            "step_index": observation.step_index,
            "sim_time": observation.sim_time,
            "action_source": action_source,
            "selected_plan": selected_plan,
            "selection_mode": selection_mode,
            "selection_source": selection_source,
            "model_recommended_plan": model_recommended_plan,
            "recommendation_brief": recommendation_brief,
            "contact_mode": contact_mode,
            "lost_contact_ids": list(lost_contact_ids or []),
            "last_contact_age_s": last_contact_age_s,
            "source_call_index": None if source_call is None else source_call.get("call_index"),
            "tactic": None if tactic is None else tactic.get("tactic"),
            "tactic_brief": None if tactic is None else tactic.get("brief"),
            "bvr_mode": (bvr_metadata or {}).get("bvr_mode", {}),
            "active_missile_threats": (bvr_metadata or {}).get(
                "active_missile_threats", []
            ),
            "threatened_platform_ids": (bvr_metadata or {}).get(
                "threatened_platform_ids", []
            ),
            "bvr_fired_actions": (bvr_metadata or {}).get("bvr_fired_actions", []),
            "bvr_launch_envelopes": (bvr_metadata or {}).get(
                "bvr_launch_envelopes", []
            ),
        }
        path = self._log_directory() / "action_sources.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                stream.write("\n")
        except (OSError, TypeError, ValueError):
            return

    def _record_context(self, observation):
        return {
            "api_version": "1.0",
            "episode_id": self._episode_id,
            "log_timestamp": self._log_timestamp,
            "side": self._side,
            "step_index": observation.step_index,
            "sim_time": observation.sim_time,
            "call_index": self._call_index,
        }

    def _path_for(self, step_index, suffix, call_index=None):
        call_index = self._call_index if call_index is None else call_index
        filename = f"step_{step_index:04d}_call_{call_index:04d}.{suffix}.json"
        return self._log_directory() / filename

    def _log_directory(self):
        return LOG_ROOT / self._log_timestamp / self._side

    @staticmethod
    def _write_json(path, record):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            # Logging must not consume a decision step or suppress the fallback.
            return
