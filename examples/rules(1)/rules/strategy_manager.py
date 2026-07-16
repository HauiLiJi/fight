import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .llm_commander import LLMStatus
from .strategy import PlanSource, Role, Tactic, TeamPlan, validate_plan
from .team_memory import LOST


class ManagerAction(Enum):
    CONTINUE = "CONTINUE"
    LOCAL_REPAIR = "LOCAL_REPAIR"
    FULL_REPLAN = "FULL_REPLAN"
    EMERGENCY = "EMERGENCY"


@dataclass(frozen=True)
class ManagerDecision:
    action: ManagerAction
    reason: str
    current_plan: object
    selected_plan: object
    candidate_count: int
    llm_request_id: object
    switch_allowed: bool
    score_delta: float
    emergency_platform_id: object = None
    emergency_target_id: object = None
    metadata: dict = field(default_factory=dict)


class StrategyManager:
    def __init__(self, params_path=None):
        path = Path(params_path) if params_path else Path(__file__).with_name("strategy_manager_params.json")
        self._params = json.loads(path.read_text(encoding="utf-8"))
        self.current_plan = None
        self.current_plan_score = None
        self.belief_at_plan_start = None
        self.plan_start_step = None
        self.plan_valid_until = None
        self.minimum_hold_until = 0
        self.last_replan_step = None
        self.last_switch_step = None
        self.pending_full_replan = False
        self.last_decision = None
        self.last_active_own_ids = set()
        self.last_known_enemy_ids = set()
        self.last_weapon_counts = {}
        self.belief_shift_streak = 0
        self.score_drop_streak = 0
        self.decision_history = deque(maxlen=int(self._params["decision_history_size"]))
        self._last_llm_request_step = None
        self.leader_key = None
        self.leader_streak = 0
        self.first_lead_step = None
        self.last_lead_step = None
        self._last_belief_label = None
        self._belief_label_streak = 0
        self._handled_belief_label = None
        self._belief_label_at_plan_start = None
        self._risk_event_active = {
            "enemy_fire_window_high": False,
            "unpressed_enemy_high_risk": False,
        }
        self._risk_event_last_step = {
            "enemy_fire_window_high": None,
            "unpressed_enemy_high_risk": None,
        }
        self._llm_primary_wait_until = None

    def reset(self):
        self.__init__()

    def ensure_initial_plan(self, observation):
        if self.current_plan is not None:
            return self.current_plan
        own_ids = _live_own_ids(observation)
        self.current_plan = TeamPlan(
            plan_id="manager_initial_safe_hold",
            created_step=observation.step_index,
            created_sim_time=observation.sim_time,
            mode=_peer_mode(),
            tactic=Tactic.DISENGAGE,
            roles={platform_id: Role.DEFENDER for platform_id in own_ids},
            target_assignments={platform_id: None for platform_id in own_ids},
            primary_target=None,
            valid_for_steps=1,
            source=PlanSource.BASELINE,
            rationale=["initial safe hold before enemy contact"],
            metadata={"legacy_no_target": True},
        )
        self._mark_plan_start(observation, None)
        return self.current_plan

    def decide(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        rule_candidates,
        tactical_summary,
        llm_result,
        current_plan_score,
        scorer,
        llm_commander,
    ):
        self.ensure_initial_plan(observation)
        self.current_plan_score = current_plan_score
        triggers = self._triggers(observation, memory_snapshot, situation, belief, current_plan_score)
        if getattr(llm_result, "status", None) == LLMStatus.READY:
            triggers.append("new llm ready")

        emergency = self._emergency_threat(situation)
        if emergency is not None:
            return self._record(
                ManagerDecision(
                    ManagerAction.EMERGENCY,
                    "emergency geometry threat",
                    self.current_plan,
                    self.current_plan,
                    len(rule_candidates),
                    getattr(llm_result, "request_id", None),
                    True,
                    0.0,
                    emergency_platform_id=emergency[0],
                    emergency_target_id=emergency[1],
                    metadata={"triggers": triggers},
                )
            )

        if self._should_local_repair(triggers):
            repaired, repaired_score = self._local_repair(
                observation,
                memory_snapshot,
                situation,
                belief,
                scorer,
            )
            if repaired is not None and self._switch_allowed(observation, repaired_score, current_plan_score, triggers):
                old_plan = self.current_plan
                self._adopt_plan(observation, belief, repaired, repaired_score)
                return self._record(
                    ManagerDecision(
                        ManagerAction.LOCAL_REPAIR,
                        "local repair accepted",
                        old_plan,
                        repaired,
                        len(rule_candidates),
                        getattr(llm_result, "request_id", None),
                        True,
                        repaired_score.final_score - current_plan_score.final_score,
                        metadata={"triggers": triggers},
                    )
                )
            triggers.append("local repair unavailable")

        if triggers or self.pending_full_replan:
            return self._full_replan(
                observation,
                memory_snapshot,
                situation,
                belief,
                rule_candidates,
                tactical_summary,
                llm_result,
                current_plan_score,
                scorer,
                llm_commander,
                triggers,
            )

        metadata = {"candidate_policy": _candidate_policy(self._params)}
        if self._should_preplan_llm(observation, llm_commander):
            request_id = self._submit_llm_if_needed(observation, tactical_summary, llm_commander, "preplan")
            if request_id is not None:
                metadata.update(
                    {
                        "llm_request_submitted": True,
                        "llm_submit_reason": "preplan",
                        "llm_request_id": request_id,
                        "llm_wait_reason": "preplan before next review or expiry",
                    }
                )
        return self._record(
            ManagerDecision(
                ManagerAction.CONTINUE,
                "current plan remains valid",
                self.current_plan,
                self.current_plan,
                len(rule_candidates),
                getattr(llm_result, "request_id", None),
                False,
                0.0,
                metadata=metadata,
            )
        )

    def _triggers(self, observation, memory_snapshot, situation, belief, current_plan_score):
        triggers = []
        own_ids = set(_live_own_ids(observation))
        enemy_ids = {
            target_id
            for target_id, track in memory_snapshot.tracks.items()
            if track.status != LOST
        }
        observed_enemy_ids = set(memory_snapshot.visible_target_ids)
        if observed_enemy_ids and not self.last_known_enemy_ids:
            triggers.append("first observed enemy")
        if self.last_active_own_ids and own_ids != self.last_active_own_ids:
            triggers.append("own aircraft set changed")
        if self.last_known_enemy_ids and enemy_ids != self.last_known_enemy_ids:
            triggers.append("enemy aircraft set changed")

        validation = validate_plan(self.current_plan, observation, memory_snapshot, situation)
        if not validation.valid:
            triggers.append("current plan invalid")
        if self._primary_target_lost(memory_snapshot):
            triggers.append("primary target lost")
        if observation.step_index >= (self.plan_valid_until or 0):
            triggers.append("plan expired")
        if self.last_replan_step is None or observation.step_index - self.last_replan_step >= int(self._params["review_interval_steps"]):
            triggers.append("review interval")

        tv = _belief_tv(getattr(belief, "posterior", {}), self.belief_at_plan_start)
        if tv > float(self._params["belief_tv_threshold"]):
            self.belief_shift_streak += 1
        else:
            self.belief_shift_streak = 0
        if self.belief_shift_streak >= int(self._params["belief_shift_required_steps"]):
            triggers.append("posterior shift")

        label = getattr(belief, "report_label", None)
        if label == self._last_belief_label:
            self._belief_label_streak += 1
        else:
            self._last_belief_label = label
            self._belief_label_streak = 1
        if (
            label is not None
            and label != self._belief_label_at_plan_start
            and label != self._handled_belief_label
            and self._belief_label_streak >= int(self._params["belief_label_stable_steps"])
        ):
            triggers.append("stable label shift")
            self._handled_belief_label = label

        if current_plan_score is not None and current_plan_score.final_score < float(self._params["current_score_minimum"]):
            self.score_drop_streak += 1
        else:
            self.score_drop_streak = 0
        if self.score_drop_streak >= int(self._params["score_drop_required_steps"]):
            triggers.append("current score degrading")

        weapon_counts = _weapon_counts(observation)
        if self.last_weapon_counts and weapon_counts != self.last_weapon_counts:
            triggers.append("weapon count changed")
        if self._supporter_better_than_shooter(situation, observation):
            triggers.append("role advantage swap")
        if _enemy_split_threat(situation):
            triggers.append("enemy split threat")
        triggers.extend(self._risk_replan_triggers(observation.step_index, current_plan_score))

        self.last_active_own_ids = own_ids
        self.last_known_enemy_ids = enemy_ids
        self.last_weapon_counts = weapon_counts
        return triggers

    def _risk_replan_triggers(self, step_index, current_plan_score):
        breakdown = _weighted_breakdown(current_plan_score) if current_plan_score is not None else None
        if breakdown is None:
            return []
        triggers = []
        risk_specs = (
            (
                "enemy_fire_window_high",
                "enemy fire window high",
                "enemy_fire_window_risk",
                "enemy_fire_window_enter_threshold",
                "enemy_fire_window_exit_threshold",
            ),
            (
                "unpressed_enemy_high_risk",
                "unpressed enemy high risk",
                "unpressed_enemy_risk",
                "unpressed_enemy_enter_threshold",
                "unpressed_enemy_exit_threshold",
            ),
        )
        cooldown = int(self._params["risk_event_cooldown_steps"])
        for key, trigger, field, enter_key, exit_key in risk_specs:
            value = float(breakdown.get(field, 0.0) or 0.0)
            if self._risk_event_active.get(key, False):
                if value <= float(self._params[exit_key]):
                    self._risk_event_active[key] = False
                continue
            last_step = self._risk_event_last_step.get(key)
            in_cooldown = last_step is not None and int(step_index) - int(last_step) < cooldown
            if value >= float(self._params[enter_key]) and not in_cooldown:
                triggers.append(trigger)
                self._risk_event_active[key] = True
                self._risk_event_last_step[key] = int(step_index)
        return triggers

    def _full_replan(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        rule_candidates,
        tactical_summary,
        llm_result,
        current_plan_score,
        scorer,
        llm_commander,
        triggers,
    ):
        self.pending_full_replan = True
        policy = _candidate_policy(self._params)
        llm_only = policy == "LLM_ONLY"
        llm_primary = policy in {"LLM_ONLY", "LLM_PRIMARY"}
        candidates = [] if llm_primary else list(rule_candidates)
        metadata = {
            "triggers": list(triggers),
            "candidate_policy": policy,
            "llm_primary_active": False,
            "rule_candidates_used_as_fallback": False,
            "llm_wait_reason": None,
            "generated_rule_count": len(rule_candidates),
            "generated_llm_count": len(getattr(llm_result, "candidates", ()) or ()),
            "llm_request_id": getattr(llm_result, "request_id", None),
            "llm_request_step": getattr(llm_result, "request_step", None),
            "llm_completed_step": getattr(llm_result, "completed_step", None),
            "llm_summary_hash": getattr(llm_result, "summary_hash", None),
            "llm_response_age_steps": getattr(llm_result, "response_age_steps", None),
            "llm_candidate_ids": list(getattr(llm_result, "candidate_ids", ()) or ()),
            "llm_stale_reasons": [],
        }
        llm_candidates_added = False
        if getattr(llm_result, "status", None) == LLMStatus.READY:
            stale_reasons = _llm_ready_stale_reasons(
                llm_result,
                observation,
                memory_snapshot,
                self.current_plan,
                self._params,
            )
            metadata["llm_stale_reasons"] = stale_reasons
            if stale_reasons:
                metadata["llm_discarded"] = True
                metadata["llm_discard_reason"] = "stale"
                if llm_commander is not None:
                    llm_commander.discard_ready_result(stale_reasons, reason="stale")
            elif not getattr(llm_result, "candidates", ()) or ():
                metadata["llm_discarded"] = True
                metadata["llm_discard_reason"] = "no_valid_candidate"
                if llm_commander is not None:
                    llm_commander.discard_ready_result(("no_valid_candidate",), reason="no_valid_candidate")
                if not llm_only:
                    metadata["rule_candidates_used_as_fallback"] = True
                    candidates = list(rule_candidates)
            else:
                if llm_only:
                    candidates.extend(llm_result.candidates)
                    metadata["llm_primary_active"] = True
                elif llm_primary:
                    candidates.extend(_safety_candidates(rule_candidates, self.current_plan))
                    candidates.extend(llm_result.candidates)
                    metadata["llm_primary_active"] = True
                else:
                    candidates.extend(llm_result.candidates)
                llm_candidates_added = bool(llm_result.candidates)
                self.pending_full_replan = False
            if stale_reasons:
                if not llm_only:
                    metadata["rule_candidates_used_as_fallback"] = True
                    candidates = list(rule_candidates)
        elif (
            llm_commander is not None
            and not llm_commander.has_inflight_request()
            and getattr(llm_result, "status", None) != LLMStatus.READY
        ):
            request_id = self._submit_llm_if_needed(observation, tactical_summary, llm_commander, "full_replan")
            if request_id is not None:
                metadata["llm_request_submitted"] = True
                metadata["llm_submit_reason"] = "full_replan"
                metadata["llm_request_id"] = request_id

        force = any(trigger in triggers for trigger in {"first observed enemy", "current plan invalid", "primary target lost", "own aircraft set changed"})
        hard_force = any(trigger in triggers for trigger in {"current plan invalid", "primary target lost", "own aircraft set changed"})
        valid_and_not_expired = not any(trigger in triggers for trigger in {"current plan invalid", "plan expired", "primary target lost"})
        can_wait_for_llm = (
            llm_primary
            and llm_commander is not None
            and llm_commander.has_inflight_request()
            and not hard_force
            and self._llm_primary_wait_until is not None
            and int(observation.step_index) <= int(self._llm_primary_wait_until)
        )
        if can_wait_for_llm or (not llm_primary and llm_commander is not None and llm_commander.has_inflight_request() and valid_and_not_expired and not force):
            wait_reason = "waiting for LLM primary candidates" if llm_primary else "waiting for pending LLM result while current plan remains valid"
            return self._record(
                ManagerDecision(
                    ManagerAction.CONTINUE,
                    wait_reason,
                    self.current_plan,
                    self.current_plan,
                    len(candidates),
                    getattr(llm_result, "request_id", None) or self._last_llm_request_step,
                    False,
                    0.0,
                    metadata={**metadata, "pending_full_replan": True, "llm_wait_reason": wait_reason},
                )
            )

        if not candidates and llm_only:
            return self._continue_without_llm_candidates(
                observation,
                llm_result,
                candidates,
                triggers,
                metadata,
                "no usable LLM candidates; continuing current plan",
            )
        if not candidates:
            candidates = list(rule_candidates)
            metadata["rule_candidates_used_as_fallback"] = True
        deduped_candidates, candidate_records = _dedupe_plans_with_records(candidates)
        candidates = deduped_candidates
        metadata["dedupe_before_count"] = len(candidate_records)
        metadata["dedupe_after_count"] = len(candidates)
        metadata["validated_count"] = len(candidates)
        metadata["scored_count"] = len(candidates)
        metadata["llm_unique_count"] = sum(
            1
            for record in candidate_records
            if record["source"] == PlanSource.LLM.value and record["dedupe_result"] == "kept"
        )
        if llm_candidates_added and metadata["llm_unique_count"] == 0:
            metadata["llm_discarded"] = True
            metadata["llm_discard_reason"] = "all_deduped"
            if llm_commander is not None:
                llm_commander.discard_ready_result(("all_deduped",), reason="all_deduped")
            if llm_only:
                return self._continue_without_llm_candidates(
                    observation,
                    llm_result,
                    candidates,
                    triggers,
                    metadata,
                    "LLM candidates all deduped; continuing current plan",
                )
            if llm_primary:
                metadata["rule_candidates_used_as_fallback"] = True
                return self._full_replan_with_rule_fallback(
                    observation,
                    memory_snapshot,
                    situation,
                    belief,
                    rule_candidates,
                    llm_result,
                    current_plan_score,
                    scorer,
                    triggers,
                    metadata,
                )
        try:
            scored = scorer.score_candidates(
                observation,
                memory_snapshot,
                situation,
                belief,
                candidates,
                self.current_plan,
            )
        except Exception as error:
            metadata["scoring_error"] = f"{type(error).__name__}: {error}"
            if llm_candidates_added and llm_commander is not None:
                llm_commander.discard_ready_result(("scoring_error",), reason="scoring_error")
            raise
        metadata["candidate_audit"] = _candidate_audit(
            candidate_records,
            scored,
            current_plan_score,
            selected_plan_id=None,
        )
        valid_llm_scored = any(score.valid and score.plan.source == PlanSource.LLM for score in scored.scored_plans)
        if llm_candidates_added and llm_primary and not valid_llm_scored:
            metadata["llm_discarded"] = True
            metadata["llm_discard_reason"] = "no_valid_llm_candidate"
            if llm_commander is not None:
                llm_commander.discard_ready_result(("no_valid_llm_candidate",), reason="no_valid_llm_candidate")
            if llm_only:
                return self._continue_without_llm_candidates(
                    observation,
                    llm_result,
                    candidates,
                    triggers,
                    metadata,
                    "no valid LLM scored plan; continuing current plan",
                )
            metadata["rule_candidates_used_as_fallback"] = True
            return self._full_replan_with_rule_fallback(
                observation,
                memory_snapshot,
                situation,
                belief,
                rule_candidates,
                llm_result,
                current_plan_score,
                scorer,
                triggers,
                metadata,
            )
        best = scored.ranked_plans[0] if scored.ranked_plans else None
        if best is None:
            if llm_candidates_added and llm_commander is not None and any(score.plan.source == PlanSource.LLM for score in scored.scored_plans if score.valid):
                consumed = llm_commander.consume_ready_result()
                metadata["llm_consumed"] = consumed is not None
            self._mark_soft_review_handled(observation, triggers)
            if llm_commander is None or not llm_commander.has_inflight_request():
                self.pending_full_replan = False
            return self._record(
                ManagerDecision(
                    ManagerAction.FULL_REPLAN,
                    "no ranked plan available",
                    self.current_plan,
                    self.current_plan,
                    len(candidates),
                    getattr(llm_result, "request_id", None),
                    False,
                    0.0,
                    metadata={**metadata, "switch_gate": {"passed": False, "reason": "no ranked plan available"}},
                )
            )

        gate_passed, gate_detail = self._switch_allowed_with_reason(observation, best, current_plan_score, triggers, force=force)
        metadata["switch_gate"] = {
            **gate_detail,
            "best_plan_id": best.plan.plan_id,
            "evaluation_time_ms": scored.evaluation_time_ms,
        }
        if gate_passed:
            old_plan = self.current_plan
            self._adopt_plan(observation, belief, best.plan, best)
            self.pending_full_replan = False
            metadata["candidate_audit"] = _candidate_audit(
                candidate_records,
                scored,
                current_plan_score,
                selected_plan_id=best.plan.plan_id,
            )
            if llm_candidates_added and llm_commander is not None and any(score.plan.source == PlanSource.LLM for score in scored.scored_plans if score.valid):
                consumed = llm_commander.consume_ready_result()
                metadata["llm_consumed"] = consumed is not None
            return self._record(
                ManagerDecision(
                    ManagerAction.FULL_REPLAN,
                    "full replan selected ranked plan",
                    old_plan,
                    best.plan,
                    len(candidates),
                    getattr(llm_result, "request_id", None),
                    True,
                    best.final_score - current_plan_score.final_score if current_plan_score else 0.0,
                    metadata=metadata,
                )
            )

        if llm_commander is not None and llm_commander.has_inflight_request() and valid_and_not_expired:
            self.pending_full_replan = True
        elif not any(trigger in triggers for trigger in {"current plan invalid", "primary target lost", "own aircraft set changed"}):
            self.pending_full_replan = False
        self._mark_soft_review_handled(observation, triggers)
        if llm_candidates_added and llm_commander is not None and any(score.plan.source == PlanSource.LLM for score in scored.scored_plans if score.valid):
            consumed = llm_commander.consume_ready_result()
            metadata["llm_consumed"] = consumed is not None
        return self._record(
            ManagerDecision(
                ManagerAction.CONTINUE,
                "switch gate rejected ranked plan",
                self.current_plan,
                self.current_plan,
                len(candidates),
                getattr(llm_result, "request_id", None),
                False,
                best.final_score - current_plan_score.final_score if current_plan_score else 0.0,
                metadata=metadata,
            )
        )

    def _continue_without_llm_candidates(self, observation, llm_result, candidates, triggers, metadata, reason):
        self._mark_soft_review_handled(observation, triggers)
        self.pending_full_replan = False
        metadata = {
            **metadata,
            "llm_only_no_candidate_reason": reason,
            "dedupe_before_count": metadata.get("dedupe_before_count", 0),
            "dedupe_after_count": metadata.get("dedupe_after_count", len(candidates)),
            "validated_count": metadata.get("validated_count", len(candidates)),
            "scored_count": 0,
            "switch_gate": {"passed": False, "reason": reason},
        }
        return self._record(
            ManagerDecision(
                ManagerAction.CONTINUE,
                reason,
                self.current_plan,
                self.current_plan,
                len(candidates),
                getattr(llm_result, "request_id", None),
                False,
                0.0,
                metadata=metadata,
            )
        )

    def _full_replan_with_rule_fallback(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        rule_candidates,
        llm_result,
        current_plan_score,
        scorer,
        triggers,
        inherited_metadata,
    ):
        candidates, candidate_records = _dedupe_plans_with_records(list(rule_candidates))
        metadata = {
            **inherited_metadata,
            "candidate_policy": "LLM_PRIMARY_RULE_FALLBACK",
            "llm_primary_active": False,
            "rule_candidates_used_as_fallback": True,
            "dedupe_before_count": len(candidate_records),
            "dedupe_after_count": len(candidates),
            "validated_count": len(candidates),
            "scored_count": len(candidates),
            "llm_unique_count": 0,
        }
        scored = scorer.score_candidates(
            observation,
            memory_snapshot,
            situation,
            belief,
            candidates,
            self.current_plan,
        )
        metadata["candidate_audit"] = _candidate_audit(
            candidate_records,
            scored,
            current_plan_score,
            selected_plan_id=None,
        )
        best = scored.ranked_plans[0] if scored.ranked_plans else None
        if best is None:
            self._mark_soft_review_handled(observation, triggers)
            self.pending_full_replan = False
            return self._record(
                ManagerDecision(
                    ManagerAction.FULL_REPLAN,
                    "LLM candidates unusable and no rule fallback ranked plan available",
                    self.current_plan,
                    self.current_plan,
                    len(candidates),
                    getattr(llm_result, "request_id", None),
                    False,
                    0.0,
                    metadata={**metadata, "switch_gate": {"passed": False, "reason": "no ranked plan available"}},
                )
            )
        force = any(trigger in triggers for trigger in {"first observed enemy", "current plan invalid", "primary target lost", "own aircraft set changed"})
        gate_passed, gate_detail = self._switch_allowed_with_reason(observation, best, current_plan_score, triggers, force=force)
        metadata["switch_gate"] = {
            **gate_detail,
            "best_plan_id": best.plan.plan_id,
            "evaluation_time_ms": scored.evaluation_time_ms,
        }
        if gate_passed:
            old_plan = self.current_plan
            self._adopt_plan(observation, belief, best.plan, best)
            self.pending_full_replan = False
            metadata["candidate_audit"] = _candidate_audit(candidate_records, scored, current_plan_score, selected_plan_id=best.plan.plan_id)
            return self._record(
                ManagerDecision(
                    ManagerAction.FULL_REPLAN,
                    "LLM candidates unusable; rule fallback selected ranked plan",
                    old_plan,
                    best.plan,
                    len(candidates),
                    getattr(llm_result, "request_id", None),
                    True,
                    best.final_score - current_plan_score.final_score if current_plan_score else 0.0,
                    metadata=metadata,
                )
            )
        self._mark_soft_review_handled(observation, triggers)
        self.pending_full_replan = False
        return self._record(
            ManagerDecision(
                ManagerAction.CONTINUE,
                "LLM candidates unusable; rule fallback switch gate rejected ranked plan",
                self.current_plan,
                self.current_plan,
                len(candidates),
                getattr(llm_result, "request_id", None),
                False,
                best.final_score - current_plan_score.final_score if current_plan_score else 0.0,
                metadata=metadata,
            )
        )

    def _local_repair(self, observation, memory_snapshot, situation, belief, scorer):
        plan = self.current_plan
        variants = []
        variants.extend(_role_swap_variants(plan, Role.SHOOTER, Role.SUPPORTER))
        variants.extend(_role_swap_variants(plan, Role.DEFENDER, Role.PRESSER))
        replacement = _replace_lost_targets(plan, memory_snapshot)
        if replacement is not None:
            variants.append(replacement)
        best = None
        for variant in variants:
            score = scorer.score_current_plan(
                observation,
                memory_snapshot,
                situation,
                belief,
                variant,
            )
            if not score.valid:
                continue
            if best is None or score.final_score > best.final_score:
                best = score
        if best is None or self.current_plan_score is None:
            return None, None
        if best.final_score > self.current_plan_score.final_score:
            return best.plan, best
        return None, None

    def _should_local_repair(self, triggers):
        return any(trigger in triggers for trigger in {"role advantage swap", "primary target lost", "weapon count changed"})

    def _switch_allowed(self, observation, new_score, current_score, triggers, force=False):
        return self._switch_allowed_with_reason(observation, new_score, current_score, triggers, force)[0]

    def _switch_allowed_with_reason(self, observation, new_score, current_score, triggers, force=False):
        strong_events = _strong_events(triggers)
        threshold_multiplier = (
            float(self._params["strong_event_threshold_multiplier"])
            if strong_events
            else 1.0
        )
        required_streak = (
            int(self._params["strong_event_required_reviews"])
            if strong_events
            else int(self._params["leader_required_reviews"])
        )
        detail = {
            "passed": False,
            "current_score": _score_value(current_score),
            "candidate_score": _score_value(new_score),
            "absolute_delta": None,
            "relative_delta": None,
            "leader_key": None,
            "leader_streak": self.leader_streak,
            "required_streak": required_streak,
            "active_triggers": list(triggers),
            "strong_events": strong_events,
            "threshold_multiplier": threshold_multiplier,
            "hold_steps": int(observation.step_index) - int(self.plan_start_step or observation.step_index),
            "gate_passed": False,
            "reject_reasons": [],
            "adopted_plan_id": None,
            "adopted_plan_source": None,
            "reason": None,
        }
        if new_score is None or not new_score.valid:
            detail["reject_reasons"].append("new plan invalid")
            detail["reason"] = "new plan invalid"
            self._reset_leader()
            return False, detail

        leader_key = _leader_key(new_score.plan, self._params)
        detail["leader_key"] = leader_key

        hard_force = force or any(trigger in triggers for trigger in {"current plan invalid", "primary target lost", "own aircraft set changed"})
        if current_score is None or not current_score.valid:
            hard_force = True
            detail["strong_events"].append("current_score_invalid")

        delta = None
        relative_delta = None
        if current_score is not None and current_score.valid:
            delta = new_score.final_score - current_score.final_score
            relative_delta = delta / max(abs(current_score.final_score), float(self._params["score_delta_epsilon"]))
        detail["absolute_delta"] = _finite_or_none(delta)
        detail["relative_delta"] = _finite_or_none(relative_delta)

        if hard_force:
            self._update_leader(leader_key, observation.step_index, force=True)
            detail["leader_streak"] = self.leader_streak
            detail["passed"] = True
            detail["gate_passed"] = True
            detail["reason"] = "forced by hard trigger"
            detail["adopted_plan_id"] = new_score.plan.plan_id
            detail["adopted_plan_source"] = new_score.plan.source.value
            return True, detail

        if observation.step_index < self.minimum_hold_until:
            detail["reject_reasons"].append("minimum hold active")

        absolute_threshold = float(self._params["switch_absolute_advantage"]) * threshold_multiplier
        relative_threshold = float(self._params["switch_relative_advantage"]) * threshold_multiplier
        absolute_ok = delta is not None and delta >= absolute_threshold
        relative_ok = relative_delta is not None and relative_delta >= relative_threshold
        advantage_ok = absolute_ok or relative_ok
        if not advantage_ok:
            detail["reject_reasons"].append("score advantage below threshold")

        if advantage_ok:
            self._update_leader(leader_key, observation.step_index)
        else:
            self._reset_leader()
        detail["leader_streak"] = self.leader_streak
        if self.leader_streak < required_streak:
            detail["reject_reasons"].append("leader streak below required reviews")

        degradation = current_score.worst_case_utility - new_score.worst_case_utility if current_score is not None and current_score.valid else 0.0
        if degradation > float(self._params["worst_case_degradation_limit"]):
            detail["reject_reasons"].append("worst case degradation too large")

        passed = not detail["reject_reasons"]
        detail["passed"] = passed
        detail["gate_passed"] = passed
        detail["reason"] = "switch gate passed" if passed else "; ".join(detail["reject_reasons"])
        if passed:
            detail["adopted_plan_id"] = new_score.plan.plan_id
            detail["adopted_plan_source"] = new_score.plan.source.value
        return passed, detail

    def _adopt_plan(self, observation, belief, plan, score):
        self.current_plan = plan
        self.current_plan_score = score
        self.belief_at_plan_start = dict(getattr(belief, "posterior", {}) or {})
        self._belief_label_at_plan_start = getattr(belief, "report_label", None)
        self._handled_belief_label = self._belief_label_at_plan_start
        self.plan_start_step = observation.step_index
        self.plan_valid_until = observation.step_index + max(1, plan.valid_for_steps)
        self.minimum_hold_until = observation.step_index + int(self._params["minimum_hold_steps"])
        self.last_replan_step = observation.step_index
        self.last_switch_step = observation.step_index
        self.belief_shift_streak = 0
        self.score_drop_streak = 0
        self._reset_leader()

    def _mark_plan_start(self, observation, belief):
        self.belief_at_plan_start = dict(getattr(belief, "posterior", {}) or {}) if belief is not None else None
        self._belief_label_at_plan_start = getattr(belief, "report_label", None) if belief is not None else None
        self._handled_belief_label = self._belief_label_at_plan_start
        self.plan_start_step = observation.step_index
        self.plan_valid_until = observation.step_index + max(1, self.current_plan.valid_for_steps)
        self.minimum_hold_until = observation.step_index + int(self._params["minimum_hold_steps"])
        self.last_replan_step = observation.step_index

    def _update_leader(self, leader_key, step_index, force=False):
        if force or leader_key == self.leader_key:
            self.leader_streak += 1
        else:
            self.leader_key = leader_key
            self.leader_streak = 1
            self.first_lead_step = step_index
        self.leader_key = leader_key
        self.last_lead_step = step_index

    def _reset_leader(self):
        self.leader_key = None
        self.leader_streak = 0
        self.first_lead_step = None
        self.last_lead_step = None

    def _mark_soft_review_handled(self, observation, triggers):
        if any(trigger in triggers for trigger in {"current plan invalid", "primary target lost", "own aircraft set changed"}):
            return
        self.last_replan_step = observation.step_index
        self.plan_valid_until = max(
            self.plan_valid_until or 0,
            observation.step_index + int(self._params["review_interval_steps"]),
        )
        self.score_drop_streak = 0
        self.belief_shift_streak = 0

    def _primary_target_lost(self, memory_snapshot):
        target_id = self.current_plan.primary_target if self.current_plan else None
        if target_id is None and self.current_plan is not None:
            assigned = [target for target in self.current_plan.target_assignments.values() if target is not None]
            target_id = assigned[0] if assigned else None
        if target_id is None:
            return False
        track = memory_snapshot.tracks.get(target_id)
        return track is None or track.status == LOST

    def _emergency_threat(self, situation):
        for track in situation.tracks:
            if not track.is_observed:
                continue
            if (
                track.pair.distance_3d_m <= float(self._params["emergency_distance_m"])
                and track.pair.closing_speed_mps >= float(self._params["emergency_closure_mps"])
                and track.pair.alignment >= float(self._params["emergency_alignment"])
            ):
                return track.ownship_id, track.target_id
        return None

    def _supporter_better_than_shooter(self, situation, observation):
        if self.current_plan is None:
            return False
        shooters = [pid for pid, role in self.current_plan.roles.items() if role == Role.SHOOTER]
        supporters = [pid for pid, role in self.current_plan.roles.items() if role == Role.SUPPORTER]
        if not shooters or not supporters:
            return False
        target_id = self.current_plan.primary_target
        if target_id is None:
            return False
        scores = {
            pid: _attack_geometry(pid, target_id, situation, observation)
            for pid in shooters + supporters
        }
        return max(scores[pid] for pid in supporters) - max(scores[pid] for pid in shooters) >= float(self._params["role_swap_advantage"])

    def _should_preplan_llm(self, observation, llm_commander):
        if _candidate_policy(self._params) not in {"LLM_ONLY", "LLM_PRIMARY"}:
            return False
        if llm_commander is None or llm_commander.has_inflight_request():
            return False
        if self.plan_valid_until is None:
            return False
        if int(observation.step_index) < int(self.plan_valid_until) - int(self._params["llm_preplan_margin_steps"]):
            return False
        return _can_submit_llm(observation.step_index, self._last_llm_request_step, self._params)

    def _submit_llm_if_needed(self, observation, tactical_summary, llm_commander, reason):
        if llm_commander is None or llm_commander.has_inflight_request():
            return None
        if not _can_submit_llm(observation.step_index, self._last_llm_request_step, self._params):
            return None
        request_id = llm_commander.submit(tactical_summary, observation.step_index)
        if request_id is None:
            return None
        self._last_llm_request_step = observation.step_index
        self._llm_primary_wait_until = int(observation.step_index) + int(self._params["llm_primary_wait_steps"])
        return request_id

    def _record(self, decision):
        self.last_decision = decision
        self.decision_history.append(decision)
        return decision


def _peer_mode():
    from .strategy import StrategyMode

    return StrategyMode.PEER


def _live_own_ids(observation):
    controlled = set(observation.controlled_platform_ids)
    return [unit.platform_id for unit in observation.own_units if unit.platform_id in controlled]


def _weapon_counts(observation):
    counts = {}
    for unit in observation.own_units:
        count = 0
        for weapon in unit.weapons:
            if weapon.name == "aam_medium":
                count = int(weapon.count)
        counts[unit.platform_id] = count
    return counts


def _belief_tv(current, start):
    if not current or not start:
        return 0.0
    keys = set(current) | set(start)
    return 0.5 * sum(abs(float(current.get(key, 0.0)) - float(start.get(key, 0.0))) for key in keys)


def _dedupe_plans(plans):
    unique = []
    seen = set()
    for plan in plans:
        signature = _plan_signature(plan)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(plan)
    return unique


def _dedupe_plans_with_records(plans):
    unique = []
    seen = {}
    records = []
    for index, plan in enumerate(plans):
        signature = _plan_signature(plan)
        duplicate_of = seen.get(signature)
        deduped = duplicate_of is None
        if deduped:
            seen[signature] = plan.plan_id
            unique.append(plan)
        records.append(
            {
                "plan_id": plan.plan_id,
                "source": plan.source.value,
                "tactic": plan.tactic.value,
                "request_id": plan.metadata.get("request_id") if isinstance(plan.metadata, dict) else None,
                "generated_index": index,
                "validated": True,
                "dedupe_result": "kept" if deduped else "duplicate",
                "duplicate_of": duplicate_of,
                "signature": signature,
            }
        )
    return unique, records


def _candidate_policy(params):
    policy = str(params.get("candidate_policy", "RULE_PRIMARY")).upper()
    if policy not in {"LLM_ONLY", "LLM_PRIMARY", "RULE_PRIMARY", "RULE_ONLY"}:
        return "RULE_PRIMARY"
    return policy


def _safety_candidates(rule_candidates, current_plan):
    candidates = []
    if current_plan is not None:
        candidates.append(current_plan)
    for plan in rule_candidates:
        if plan.tactic == Tactic.DISENGAGE:
            candidates.append(plan)
    return _dedupe_plans(candidates)


def _plan_signature(plan):
    return (
        plan.mode.value,
        plan.tactic.value,
        tuple((key, value.value) for key, value in sorted(plan.roles.items())),
        tuple(sorted(plan.target_assignments.items())),
        plan.primary_target,
    )


def _candidate_audit(candidate_records, scoring_result, current_score, selected_plan_id):
    records = [dict(record) for record in candidate_records]
    by_plan_id = {record["plan_id"]: record for record in records}
    ranked_plan_ids = [score.plan.plan_id for score in scoring_result.ranked_plans]
    for score in scoring_result.scored_plans:
        record = by_plan_id.get(score.plan.plan_id)
        if record is None:
            continue
        record.update(
            {
                "scored": True,
                "rank": ranked_plan_ids.index(score.plan.plan_id) + 1 if score.plan.plan_id in ranked_plan_ids else None,
                "final_score": _finite_or_none(score.final_score),
                "expected_score": _finite_or_none(score.expected_utility),
                "worst_score": _finite_or_none(score.worst_case_utility),
                "switch_cost": _finite_or_none(score.switch_cost),
                "score_delta_to_current": _score_delta(score, current_score),
                "utility_breakdown": _weighted_breakdown(score),
                "gate_passed": False,
                "reject_reason": "not ranked" if not score.valid else "not selected",
                "adopted": score.plan.plan_id == selected_plan_id,
                "executed": score.plan.plan_id == selected_plan_id,
                "invalid_reasons": list(score.invalid_reasons),
            }
        )
    for record in records:
        if "scored" not in record:
            record.update(
                {
                    "scored": False,
                    "rank": None,
                    "final_score": None,
                    "expected_score": None,
                    "worst_score": None,
                    "switch_cost": None,
                    "score_delta_to_current": None,
                    "utility_breakdown": None,
                    "gate_passed": False,
                    "reject_reason": "deduped duplicate",
                    "adopted": False,
                    "executed": False,
                    "invalid_reasons": [],
                }
            )
    if selected_plan_id in by_plan_id:
        by_plan_id[selected_plan_id]["gate_passed"] = True
        by_plan_id[selected_plan_id]["reject_reason"] = None
    return records


def _weighted_breakdown(score):
    if not score.valid or not score.hypothesis_scores:
        return None
    fields = (
        "attack_opportunity",
        "own_fire_opportunity",
        "survivability",
        "terminal_survival",
        "exchange_value",
        "coordination",
        "counter_effect",
        "local_advantage",
        "bracket_risk",
        "unpressed_enemy_risk",
        "enemy_fire_window_risk",
        "separation_risk",
        "ammo_waste",
        "duplicate_attack_waste",
        "total",
    )
    values = {field: 0.0 for field in fields}
    diagnostics = {
        "own_pending_shots": 0.0,
        "enemy_pending_shots": 0.0,
        "max_own_hit_probability": 0.0,
        "max_enemy_hit_probability": 0.0,
        "expected_own_losses": 0.0,
        "expected_enemy_losses": 0.0,
        "prelaunch_own_expected_losses": 0.0,
        "prelaunch_enemy_expected_losses": 0.0,
        "engagement_own_expected_losses": 0.0,
        "engagement_enemy_expected_losses": 0.0,
        "enemy_threat_chain_count": 0.0,
        "own_threat_chain_count": 0.0,
        "max_enemy_launch_probability": 0.0,
        "max_own_launch_probability": 0.0,
    }
    for hypothesis_score in score.hypothesis_scores:
        for field in fields:
            values[field] += hypothesis_score.probability * getattr(hypothesis_score.breakdown, field)
        diag = getattr(hypothesis_score, "diagnostics", {}) or {}
        for key in (
            "own_pending_shots",
            "enemy_pending_shots",
            "expected_own_losses",
            "expected_enemy_losses",
            "prelaunch_own_expected_losses",
            "prelaunch_enemy_expected_losses",
            "engagement_own_expected_losses",
            "engagement_enemy_expected_losses",
            "enemy_threat_chain_count",
            "own_threat_chain_count",
        ):
            diagnostics[key] += hypothesis_score.probability * float(diag.get(key, 0.0))
        diagnostics["max_own_hit_probability"] = max(diagnostics["max_own_hit_probability"], float(diag.get("max_own_hit_probability", 0.0)))
        diagnostics["max_enemy_hit_probability"] = max(diagnostics["max_enemy_hit_probability"], float(diag.get("max_enemy_hit_probability", 0.0)))
        diagnostics["max_enemy_launch_probability"] = max(diagnostics["max_enemy_launch_probability"], float(diag.get("max_enemy_launch_probability", 0.0)))
        diagnostics["max_own_launch_probability"] = max(diagnostics["max_own_launch_probability"], float(diag.get("max_own_launch_probability", 0.0)))
    result = {field: _finite_or_none(value) for field, value in values.items()}
    result.update({key: _finite_or_none(value) for key, value in diagnostics.items()})
    return result


def _score_delta(score, current_score):
    if current_score is None or not getattr(current_score, "valid", False) or not score.valid:
        return None
    return _finite_or_none(score.final_score - current_score.final_score)


def _score_value(score):
    if score is None or not getattr(score, "valid", False):
        return None
    return _finite_or_none(score.final_score)


def _strong_events(triggers):
    mapping = {
        "new llm ready": "new_llm_ready",
        "current plan invalid": "current_plan_invalid",
        "primary target lost": "current_target_lost_or_destroyed",
        "own aircraft set changed": "ownship_destroyed",
        "stable label shift": "stable_belief_shift",
        "enemy split threat": "enemy_split_threat",
        "current score degrading": "current_score_degrading",
        "enemy fire window high": "enemy_fire_window_high",
        "unpressed enemy high risk": "unpressed_enemy_high_risk",
    }
    return [mapping[trigger] for trigger in triggers if trigger in mapping]


def _leader_key(plan, params):
    mode = str(params.get("leader_identity_mode", "semantic"))
    if mode == "plan_id":
        return plan.plan_id
    metadata = {}
    if isinstance(plan.metadata, dict):
        for key in ("bracket_sides", "presser", "supporter", "defender", "threat_target", "support_target"):
            if key in plan.metadata:
                metadata[key] = plan.metadata[key]
    payload = (
        plan.mode.value,
        plan.tactic.value,
        tuple((key, value.value) for key, value in sorted(plan.roles.items())),
        tuple(sorted(plan.target_assignments.items())),
        plan.primary_target,
        tuple(sorted((key, _stable_metadata_value(value)) for key, value in metadata.items())),
    )
    return repr(payload)


def _stable_metadata_value(value):
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    if isinstance(value, list):
        return tuple(value)
    return value


def _enemy_split_threat(situation):
    threatened = {}
    for track in getattr(situation, "tracks", ()) or ():
        if not getattr(track, "is_observed", False):
            continue
        pair = track.pair
        if (
            pair.closing_speed_mps > 100.0
            and pair.alignment > 0.45
            and pair.distance_3d_m < 70000.0
        ):
            threatened.setdefault(track.target_id, set()).add(track.ownship_id)
    if len(threatened) < 2:
        return False
    own_targets = set()
    for own_ids in threatened.values():
        own_targets.update(own_ids)
    return len(own_targets) >= 2


def _finite_or_none(value):
    try:
        if value == float("inf") or value == float("-inf"):
            return None
        return float(value)
    except Exception:
        return None


def _llm_ready_stale_reasons(llm_result, observation, memory_snapshot, current_plan, params):
    reasons = []
    max_age = int(params.get("llm_ready_max_age_steps", 120))
    completed_step = getattr(llm_result, "completed_step", None)
    if completed_step is not None and int(observation.step_index) - int(completed_step) > max_age:
        reasons.append("ready_age_exceeded")
    request_plan_id = getattr(llm_result, "request_plan_id", None)
    current_plan_id = getattr(current_plan, "plan_id", None)
    if request_plan_id is not None and current_plan_id is not None and request_plan_id != current_plan_id:
        reasons.append("request_plan_changed")
    known_targets = set(memory_snapshot.tracks)
    live_platform_ids = {unit.platform_id for unit in observation.own_units}
    for plan in getattr(llm_result, "candidates", ()) or ():
        for target_id in plan.target_assignments.values():
            if target_id is not None and target_id not in known_targets:
                reasons.append("candidate_target_missing")
                break
        if plan.primary_target is not None and plan.primary_target not in known_targets:
            reasons.append("candidate_primary_target_missing")
        for platform_id in plan.roles:
            if platform_id not in live_platform_ids and platform_id in observation.controlled_platform_ids:
                reasons.append("candidate_platform_not_live")
                break
    return sorted(set(reasons))


def _can_submit_llm(step_index, last_request_step, params):
    if last_request_step is None:
        return True
    wait_steps = int(params.get("llm_wait_max_steps", 1))
    return int(step_index) - int(last_request_step) >= max(1, wait_steps)


def _role_swap_variants(plan, role_a, role_b):
    ids_a = [pid for pid, role in plan.roles.items() if role == role_a]
    ids_b = [pid for pid, role in plan.roles.items() if role == role_b]
    variants = []
    for platform_a in ids_a:
        for platform_b in ids_b:
            roles = dict(plan.roles)
            roles[platform_a], roles[platform_b] = roles[platform_b], roles[platform_a]
            variants.append(_copy_plan(plan, roles=roles))
    return variants


def _replace_lost_targets(plan, memory_snapshot):
    valid_targets = [
        target_id
        for target_id, track in sorted(memory_snapshot.tracks.items())
        if track.status != LOST
    ]
    if not valid_targets:
        return None
    changed = False
    assignments = {}
    for platform_id, target_id in plan.target_assignments.items():
        if target_id is None:
            assignments[platform_id] = None
            continue
        track = memory_snapshot.tracks.get(target_id)
        if track is None or track.status == LOST:
            assignments[platform_id] = valid_targets[0]
            changed = True
        else:
            assignments[platform_id] = target_id
    if not changed:
        return None
    primary_target = plan.primary_target
    if primary_target is not None:
        track = memory_snapshot.tracks.get(primary_target)
        if track is None or track.status == LOST:
            primary_target = valid_targets[0]
    return _copy_plan(plan, target_assignments=assignments, primary_target=primary_target)


def _copy_plan(plan, roles=None, target_assignments=None, primary_target=None):
    return TeamPlan(
        plan_id=f"{plan.plan_id}_repair",
        created_step=plan.created_step,
        created_sim_time=plan.created_sim_time,
        mode=plan.mode,
        tactic=plan.tactic,
        roles=roles if roles is not None else dict(plan.roles),
        target_assignments=target_assignments if target_assignments is not None else dict(plan.target_assignments),
        primary_target=plan.primary_target if primary_target is None else primary_target,
        valid_for_steps=plan.valid_for_steps,
        source=plan.source,
        rationale=list(plan.rationale) + ["local repair"],
        metadata=dict(plan.metadata),
    )


def _attack_geometry(platform_id, target_id, situation, observation):
    weapon_bonus = 0.0
    for unit in observation.own_units:
        if unit.platform_id != platform_id:
            continue
        for weapon in unit.weapons:
            if weapon.name == "aam_medium" and weapon.enabled and weapon.count > 0:
                weapon_bonus = 0.1
    for track in situation.tracks:
        if track.ownship_id == platform_id and track.target_id == target_id:
            return (
                track.pair.own_alignment
                + max(track.pair.closing_speed_mps, 0.0) / 500.0
                - track.pair.distance_3d_m / 100000.0
                + weapon_bonus
            )
    return -1.0
