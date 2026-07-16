import json
import os
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import FIRE_RANGE_M
from .engagement_envelope import evaluate_engagement_geometry


@dataclass(frozen=True)
class DecisionTrace:
    episode_id: str
    step_index: int
    sim_time: float
    visible_enemy_ids: list
    known_enemy_ids: list
    track_states: dict
    belief_posterior: dict
    belief_report_label: object
    belief_entropy: object
    current_plan_before: object
    manager_action: object
    manager_reason: object
    selected_plan: object
    plan_source: object
    candidate_count: int
    rule_candidate_count: int
    llm_candidate_count: int
    current_plan_score: object
    selected_plan_score: object
    score_delta: object
    llm_status: object
    llm_latency_ms: object
    llm_request_id: object
    llm_request_step: object
    llm_completed_step: object
    llm_summary_hash: object
    llm_response_age_steps: object
    llm_candidate_ids: list
    llm_consumed: bool
    llm_stale_reasons: list
    llm_error: object
    llm_request_count: int
    llm_response_count: int
    llm_consumed_count: int
    llm_discarded_count: int
    llm_discard_reasons: dict
    llm_failure_event_count: int
    llm_timeout_event_count: int
    llm_stale_event_count: int
    llm_status_step_counts: dict
    llm_stale_count: int
    generated_rule_count: int
    generated_llm_count: int
    validated_count: int
    dedupe_before_count: int
    dedupe_after_count: int
    scored_count: int
    candidate_policy: object
    llm_primary_active: bool
    rule_candidates_used_as_fallback: bool
    llm_wait_reason: object
    candidate_audit: list
    switch_gate: object
    fallback_reason: object
    actions_summary: list
    engagement_envelope: list
    fire_eligibility: list
    scorer_executor_consistency: dict
    module_timings_ms: dict
    total_act_ms: float
    errors: list = field(default_factory=list)


class DiagnosticsRecorder:
    def __init__(self, params_path=None):
        self._params_path = Path(params_path) if params_path else Path(__file__).with_name("diagnostics_params.json")
        self._params = json.loads(self._params_path.read_text(encoding="utf-8"))
        self._buffer = deque(maxlen=int(self._params["memory_history_size"]))
        self._file = None
        self._file_path = None
        self._episode_id = "episode"
        self._pending_writes = 0
        self._internal_errors = []

    def reset(self, context):
        self.close()
        self._buffer = deque(maxlen=int(self._params["memory_history_size"]))
        self._internal_errors = []
        self._pending_writes = 0
        self._episode_id = _safe_name(_episode_id(context))
        directory = os.environ.get("BAIYANG_DIAGNOSTICS_DIR")
        if not directory:
            return
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self._file_path = path / f"{self._episode_id}.jsonl"
            self._file = self._file_path.open("a", encoding="utf-8")
        except Exception as error:
            self._file = None
            self._remember_error(error)

    def record(self, trace):
        if self._internal_errors:
            errors = list(trace.errors) + list(self._internal_errors)
            trace = DecisionTrace(**{**asdict(trace), "errors": errors})
        self._buffer.append(trace)
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(asdict(trace), ensure_ascii=True, sort_keys=True) + "\n")
            self._pending_writes += 1
            if self._pending_writes >= int(self._params["flush_every_steps"]):
                self.flush()
        except Exception as error:
            self._remember_error(error)

    def recent_traces(self):
        return tuple(self._buffer)

    def flush(self):
        if self._file is None:
            return
        try:
            self._file.flush()
            self._pending_writes = 0
        except Exception as error:
            self._remember_error(error)

    def close(self):
        if self._file is None:
            return
        try:
            self.flush()
            self._file.close()
        except Exception as error:
            self._remember_error(error)
        finally:
            self._file = None

    def _remember_error(self, error):
        limit = int(self._params.get("max_error_text_chars", 500))
        self._internal_errors.append(_truncate(f"{type(error).__name__}: {error}", limit))
        self._internal_errors = self._internal_errors[-5:]


def build_decision_trace(
    observation,
    memory_snapshot,
    belief,
    current_plan_before,
    decision,
    selected_plan,
    current_plan_score,
    selected_plan_score,
    rule_candidates,
    llm_result,
    actions,
    timings,
    total_act_ms,
    fallback_reason=None,
    errors=None,
    episode_id="episode",
    llm_stats=None,
    fire_eligibility=None,
):
    metadata = getattr(decision, "metadata", {}) or {}
    llm_stats = llm_stats or {}
    return DecisionTrace(
        episode_id=str(episode_id),
        step_index=int(getattr(observation, "step_index", 0)),
        sim_time=float(getattr(observation, "sim_time", 0.0)),
        visible_enemy_ids=sorted(getattr(memory_snapshot, "visible_target_ids", ()) or ()),
        known_enemy_ids=sorted((getattr(memory_snapshot, "tracks", {}) or {}).keys()),
        track_states={
            target_id: getattr(track, "status", None)
            for target_id, track in sorted((getattr(memory_snapshot, "tracks", {}) or {}).items())
        },
        belief_posterior=dict(getattr(belief, "posterior", {}) or {}),
        belief_report_label=getattr(belief, "report_label", None),
        belief_entropy=getattr(belief, "normalized_entropy", None),
        current_plan_before=_plan_summary(current_plan_before),
        manager_action=_enum_value(getattr(decision, "action", None)),
        manager_reason=getattr(decision, "reason", None),
        selected_plan=_plan_summary(selected_plan),
        plan_source=_enum_value(getattr(selected_plan, "source", None)),
        candidate_count=int(getattr(decision, "candidate_count", len(rule_candidates or [])) if decision is not None else len(rule_candidates or [])),
        rule_candidate_count=len(rule_candidates or []),
        llm_candidate_count=len(getattr(llm_result, "candidates", ()) or ()),
        current_plan_score=_score_summary(current_plan_score),
        selected_plan_score=_score_summary(selected_plan_score),
        score_delta=getattr(decision, "score_delta", None),
        llm_status=_enum_value(getattr(llm_result, "status", None)),
        llm_latency_ms=getattr(llm_result, "latency_ms", None),
        llm_request_id=getattr(llm_result, "request_id", None),
        llm_request_step=getattr(llm_result, "request_step", None),
        llm_completed_step=getattr(llm_result, "completed_step", None),
        llm_summary_hash=getattr(llm_result, "summary_hash", None),
        llm_response_age_steps=getattr(llm_result, "response_age_steps", None),
        llm_candidate_ids=list(getattr(llm_result, "candidate_ids", ()) or ()),
        llm_consumed=bool(metadata.get("llm_consumed", getattr(llm_result, "consumed", False))),
        llm_stale_reasons=list(metadata.get("llm_stale_reasons") or getattr(llm_result, "stale_reasons", ()) or ()),
        llm_error=_truncate(getattr(llm_result, "error", None), int(_diagnostics_params().get("max_error_text_chars", 500))) if getattr(llm_result, "error", None) else None,
        llm_request_count=int(llm_stats.get("request_count", 0)),
        llm_response_count=int(llm_stats.get("response_count", 0)),
        llm_consumed_count=int(llm_stats.get("consumed_count", 0)),
        llm_discarded_count=int(llm_stats.get("discarded_count", 0)),
        llm_discard_reasons=dict(llm_stats.get("discard_reasons", {}) or {}),
        llm_failure_event_count=int(llm_stats.get("failure_event_count", 0)),
        llm_timeout_event_count=int(llm_stats.get("timeout_event_count", 0)),
        llm_stale_event_count=int(llm_stats.get("stale_event_count", llm_stats.get("stale_count", 0))),
        llm_status_step_counts=dict(llm_stats.get("status_step_counts", {}) or {}),
        llm_stale_count=int(llm_stats.get("stale_count", 0)),
        generated_rule_count=int(metadata.get("generated_rule_count", len(rule_candidates or []))),
        generated_llm_count=int(metadata.get("generated_llm_count", len(getattr(llm_result, "candidates", ()) or ()))),
        validated_count=int(metadata.get("validated_count", 0)),
        dedupe_before_count=int(metadata.get("dedupe_before_count", 0)),
        dedupe_after_count=int(metadata.get("dedupe_after_count", 0)),
        scored_count=int(metadata.get("scored_count", 0)),
        candidate_policy=metadata.get("candidate_policy"),
        llm_primary_active=bool(metadata.get("llm_primary_active", False)),
        rule_candidates_used_as_fallback=bool(metadata.get("rule_candidates_used_as_fallback", False)),
        llm_wait_reason=metadata.get("llm_wait_reason"),
        candidate_audit=_candidate_audit_summary(metadata.get("candidate_audit", [])),
        switch_gate=metadata.get("switch_gate"),
        fallback_reason=fallback_reason,
        actions_summary=summarize_actions(actions),
        engagement_envelope=_engagement_envelope_summary(observation, memory_snapshot, selected_plan),
        fire_eligibility=list(fire_eligibility or ()),
        scorer_executor_consistency=_scorer_executor_consistency(fire_eligibility or (), current_plan_score),
        module_timings_ms={key: round(float(value), 3) for key, value in sorted((timings or {}).items())},
        total_act_ms=round(float(total_act_ms), 3),
        errors=list(errors or []),
    )


def summarize_actions(action_batch):
    if action_batch is None:
        return []
    if hasattr(action_batch, "model_dump"):
        data = action_batch.model_dump()
    elif hasattr(action_batch, "dict"):
        data = action_batch.dict()
    else:
        data = action_batch
    actions = data.get("actions", []) if isinstance(data, dict) else []
    summary = []
    for action in actions:
        if not isinstance(action, dict):
            if hasattr(action, "model_dump"):
                action = action.model_dump()
            elif hasattr(action, "dict"):
                action = action.dict()
            else:
                continue
        item = {
            "type": action.get("type"),
            "platform_id": action.get("platform_id"),
            "target_id": action.get("target_id"),
            "guider_id": action.get("guider_id"),
        }
        if action.get("type") == "set_flight":
            item.update(
                {
                    "heading_deg": action.get("heading_deg"),
                    "altitude_m": action.get("altitude_m"),
                    "mach": action.get("mach"),
                }
            )
        summary.append({key: value for key, value in item.items() if value is not None})
    return summary


def _engagement_envelope_summary(observation, memory_snapshot, selected_plan):
    if selected_plan is None:
        return []
    tracks = getattr(memory_snapshot, "tracks", {}) or {}
    own_units = {
        unit.platform_id: unit
        for unit in getattr(observation, "own_units", ()) or ()
        if unit.platform_id in set(getattr(observation, "controlled_platform_ids", ()) or ())
    }
    output = []
    for platform_id, target_id in sorted((getattr(selected_plan, "target_assignments", {}) or {}).items()):
        if target_id is None:
            continue
        ownship = own_units.get(platform_id)
        target = tracks.get(target_id)
        if ownship is None or target is None:
            continue
        if not (
            hasattr(ownship, "position")
            and hasattr(ownship, "attitude")
            and hasattr(target, "position")
            and hasattr(target, "attitude")
        ):
            continue
        geometry = evaluate_engagement_geometry(
            ownship.position,
            ownship.attitude.heading_deg,
            target.position,
            target.attitude.heading_deg,
        )
        output.append(
            {
                "platform_id": platform_id,
                "target_id": target_id,
                "distance_m": round(float(geometry.distance_m), 3),
                "target_aspect_deg": round(float(geometry.target_aspect_deg), 3),
                "aspect_class": geometry.aspect_class.value,
                "shooter_heading_error_deg": round(float(geometry.shooter_heading_error_deg), 3),
                "dynamic_launch_range_m": round(float(geometry.dynamic_launch_range_m), 3),
                "within_dynamic_range": bool(geometry.within_dynamic_range),
                "heading_aligned_normal": bool(geometry.heading_aligned_normal),
                "heading_aligned_counter": bool(geometry.heading_aligned_counter),
                "old_fixed_50km_eligible": bool(geometry.distance_m <= FIRE_RANGE_M),
            }
        )
    return output


def _scorer_executor_consistency(fire_eligibility, score):
    executor_eligible = sum(1 for item in fire_eligibility if item.get("executor_fire_eligible_now"))
    predicted = _max_predicted_fire_window(score)
    low_threshold = 0.35
    high_threshold = 0.70
    return {
        "max_predicted_fire_window_probability": predicted,
        "executor_eligible_count": executor_eligible,
        "executor_eligible_scorer_low_count": executor_eligible if executor_eligible and predicted < low_threshold else 0,
        "executor_ineligible_scorer_high_count": 1 if not executor_eligible and predicted > high_threshold else 0,
        "current_ineligible_future_possible_count": 1 if not executor_eligible and low_threshold <= predicted <= high_threshold else 0,
    }


def _max_predicted_fire_window(score):
    values = []
    for hypothesis in getattr(score, "hypothesis_scores", ()) or ():
        diagnostics = getattr(hypothesis, "diagnostics", {}) or {}
        values.append(float(diagnostics.get("max_own_fire_window_probability", 0.0) or 0.0))
        for chain in diagnostics.get("own_threat_chains", ()) or ():
            values.append(float(chain.get("predicted_fire_window_probability", 0.0) or 0.0))
    return round(max(values) if values else 0.0, 3)


def _plan_summary(plan):
    if plan is None:
        return None
    return {
        "plan_id": plan.plan_id,
        "mode": _enum_value(plan.mode),
        "tactic": _enum_value(plan.tactic),
        "roles": {key: _enum_value(value) for key, value in sorted(plan.roles.items())},
        "target_assignments": dict(sorted(plan.target_assignments.items())),
        "primary_target": plan.primary_target,
        "valid_for_steps": plan.valid_for_steps,
        "source": _enum_value(plan.source),
        "rationale": list(plan.rationale[:3]),
    }


def _score_summary(score):
    if score is None:
        return None
    return {
        "valid": bool(score.valid),
        "final_score": _finite_or_none(score.final_score),
        "expected_utility": _finite_or_none(score.expected_utility),
        "worst_case_utility": _finite_or_none(score.worst_case_utility),
        "switch_cost": _finite_or_none(score.switch_cost),
        "summary": score.summary,
    }


def _candidate_audit_summary(records):
    params = _diagnostics_params()
    limit = int(params.get("max_candidates_logged", params.get("max_ranked_plans_logged", 5)))
    include_breakdown = bool(params.get("record_candidate_breakdown", True))
    output = []
    for record in list(records or [])[:limit]:
        if not isinstance(record, dict):
            continue
        item = {
            "plan_id": record.get("plan_id"),
            "source": record.get("source"),
            "tactic": record.get("tactic"),
            "request_id": record.get("request_id"),
            "validated": record.get("validated"),
            "dedupe_result": record.get("dedupe_result"),
            "scored": record.get("scored"),
            "rank": record.get("rank"),
            "final_score": record.get("final_score"),
            "expected_score": record.get("expected_score"),
            "worst_score": record.get("worst_score"),
            "switch_cost": record.get("switch_cost"),
            "score_delta_to_current": record.get("score_delta_to_current"),
            "gate_passed": record.get("gate_passed"),
            "reject_reason": record.get("reject_reason"),
            "adopted": record.get("adopted"),
            "executed": record.get("executed"),
            "invalid_reasons": record.get("invalid_reasons", []),
            "duplicate_of": record.get("duplicate_of"),
        }
        if include_breakdown:
            item["utility_breakdown"] = record.get("utility_breakdown")
        output.append({key: value for key, value in item.items() if value is not None})
    return output


def _diagnostics_params():
    try:
        path = Path(__file__).with_name("diagnostics_params.json")
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _finite_or_none(value):
    try:
        if value == float("inf") or value == float("-inf"):
            return None
        return float(value)
    except Exception:
        return None


def _enum_value(value):
    return getattr(value, "value", value)


def _episode_id(context):
    for name in ("episode_id", "scenario_id", "run_id"):
        value = getattr(context, name, None)
        if value is not None:
            return str(value)
    return "episode"


def _safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80]
    return value or "episode"


def _truncate(value, limit):
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
