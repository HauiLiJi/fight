import math
import time

from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1

from .config import (
    EARTH_RADIUS_M,
)
from . import chain_logger
from .diagnostics import DiagnosticsRecorder, build_decision_trace
from .executor import Executor
from .fallback import FallbackPolicy
from .llm_commander import LLMCommander
from .opponent_belief import OpponentBelief
from .runtime_validation import assert_runtime_configuration_valid
from .situation import SituationAnalyzer
from .strategy_scorer import StrategyScorer
from .strategy_manager import ManagerAction, StrategyManager
from .strategy import (
    PlanSource,
    Role,
    RuleCandidateGenerator,
    StrategyMode,
    Tactic,
    TeamPlan,
    validate_plan,
)
from .team_memory import TeamMemory


class BaiyangRuleAgent(BaseAgent):
    """Team deterministic A2A baseline using the competition API."""

    def __init__(self):
        self._memory = TeamMemory()
        self._analyzer = SituationAnalyzer()
        self._belief = OpponentBelief()
        self._candidate_generator = RuleCandidateGenerator()
        self._executor = Executor()
        self._fallback = FallbackPolicy()
        self._strategy_scorer = StrategyScorer()
        self._llm_commander = LLMCommander()
        self._strategy_manager = StrategyManager()
        self._diagnostics = DiagnosticsRecorder()
        self._configuration_report = None
        self._episode_id = "episode"
        self._last_situation = None
        self._last_belief = None
        self._last_rule_candidates = []
        self._last_execution_plan = None
        self._last_current_plan_score = None
        self._last_tactical_summary = None
        self._last_llm_result = None
        self._last_manager_decision = None

    def reset(self, context):
        self._configuration_report = assert_runtime_configuration_valid()
        super().reset(context)
        self._memory.reset()
        self._belief.reset()
        self._executor.reset()
        self._llm_commander.reset()
        self._strategy_manager.reset()
        self._diagnostics.reset(context)
        chain_logger.reset(context)
        self._episode_id = _episode_id(context)
        self._last_situation = None
        self._last_belief = None
        self._last_rule_candidates = []
        self._last_execution_plan = None
        self._last_current_plan_score = None
        self._last_tactical_summary = None
        self._last_llm_result = None
        self._last_manager_decision = None

    def close(self):
        self._llm_commander.close()
        self._diagnostics.close()
        chain_logger.close()

    def act(self, observation):
        chain_logger.set_context(observation)
        start = time.perf_counter()
        timings = {}
        state = _ActState()
        try:
            actions = self._act_impl(observation, timings, state)
            self._record_trace(observation, timings, start, state, actions)
            return actions
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            errors = [_error_text(error)]
            try:
                actions = self._fallback.build_safe_hold(observation)
                fallback_reason = "top-level exception safe_hold"
            except Exception as fallback_error:
                errors.append(_error_text(fallback_error))
                actions = ActionBatchV1()
                fallback_reason = "top-level exception empty ActionBatchV1"
            state.fallback_reason = fallback_reason
            self._record_trace(observation, timings, start, state, actions, errors=errors)
            return actions

    def _act_impl(self, observation, timings, state):
        step_start = time.perf_counter()
        memory_snapshot = self._memory.update(observation)
        timings["memory"] = _elapsed_ms(step_start)
        state.memory_snapshot = memory_snapshot

        step_start = time.perf_counter()
        situation = self._analyzer.compute(observation, memory_snapshot)
        timings["situation"] = _elapsed_ms(step_start)
        state.situation = situation

        step_start = time.perf_counter()
        belief = self._belief.update(observation, situation)
        timings["belief"] = _elapsed_ms(step_start)
        state.belief = belief
        self._log_belief_update(observation, memory_snapshot, situation, belief)

        step_start = time.perf_counter()
        rule_candidates = self._candidate_generator.generate(
            observation,
            memory_snapshot,
            situation,
            belief,
        )
        timings["rule_candidates"] = _elapsed_ms(step_start)
        state.rule_candidates = rule_candidates
        self._last_situation = situation
        self._last_belief = belief
        self._last_rule_candidates = rule_candidates

        current_plan = self._strategy_manager.ensure_initial_plan(observation)
        state.current_plan_before = current_plan
        self._last_execution_plan = current_plan

        step_start = time.perf_counter()
        self._llm_commander.prepare_parse_context(
            observation,
            memory_snapshot,
            situation,
        )
        timings["summary"] = _elapsed_ms(step_start)

        step_start = time.perf_counter()
        self._last_llm_result = self._llm_commander.poll(observation.step_index)
        timings["llm_poll"] = _elapsed_ms(step_start)
        state.llm_result = self._last_llm_result

        step_start = time.perf_counter()
        self._last_current_plan_score = self._strategy_scorer.score_current_plan(
            observation,
            memory_snapshot,
            situation,
            belief,
            current_plan,
        )
        timings["current_plan_score"] = _elapsed_ms(step_start)
        state.current_plan_score = self._last_current_plan_score

        step_start = time.perf_counter()
        self._last_tactical_summary = self._llm_commander.build_summary(
            observation,
            memory_snapshot,
            situation,
            belief,
            current_plan,
            self._last_current_plan_score,
            rule_candidates,
        )
        timings["summary"] = _elapsed_ms(step_start)

        step_start = time.perf_counter()
        decision = self._strategy_manager.decide(
            observation,
            memory_snapshot,
            situation,
            belief,
            rule_candidates,
            self._last_tactical_summary,
            self._last_llm_result,
            self._last_current_plan_score,
            self._strategy_scorer,
            self._llm_commander,
        )
        timings["strategy_manager"] = _elapsed_ms(step_start)
        self._last_manager_decision = decision
        state.decision = decision
        selected_plan = decision.selected_plan or current_plan
        state.selected_plan = selected_plan
        state.selected_plan_score = self._strategy_manager.current_plan_score
        self._log_manager_decision(decision, current_plan, selected_plan)
        self._last_execution_plan = selected_plan
        self._memory.set_plan_context(
            plan_id=selected_plan.plan_id,
            tactic=selected_plan.tactic.value,
            roles={key: value.value for key, value in selected_plan.roles.items()},
            target_assignments=dict(selected_plan.target_assignments),
        )

        step_start = time.perf_counter()
        if decision.action == ManagerAction.EMERGENCY:
            actions = self._fallback.build_emergency_evasion(
                observation,
                situation,
                decision.emergency_platform_id,
                decision.emergency_target_id,
            )
            timings["executor_or_fallback"] = _elapsed_ms(step_start)
            state.fallback_reason = "emergency_evasion"
            return actions
        validation = validate_plan(
            selected_plan,
            observation,
            memory_snapshot,
            situation,
        )
        if not validation.valid:
            actions = self._fallback.build_invalid_plan(
                observation,
                memory_snapshot,
                situation,
                validation.errors,
            )
            timings["executor_or_fallback"] = _elapsed_ms(step_start)
            state.fallback_reason = f"invalid plan: {validation.errors}"
            return actions
        state.fire_eligibility = self._executor.fire_eligibility_summary(
            observation,
            memory_snapshot,
            selected_plan,
        )
        actions = self._executor.build_actions(
            observation,
            memory_snapshot,
            situation,
            selected_plan,
        )
        timings["executor_or_fallback"] = _elapsed_ms(step_start)
        self._log_step_actions(observation, actions)
        return actions

    def _record_trace(self, observation, timings, start, state, actions, errors=None):
        timings["total"] = _elapsed_ms(start)
        memory_snapshot = state.memory_snapshot or _EmptyMemorySnapshot()
        trace = build_decision_trace(
            observation=observation,
            memory_snapshot=memory_snapshot,
            belief=state.belief,
            current_plan_before=state.current_plan_before,
            decision=state.decision,
            selected_plan=state.selected_plan,
            current_plan_score=state.current_plan_score,
            selected_plan_score=state.selected_plan_score,
            rule_candidates=state.rule_candidates,
            llm_result=state.llm_result,
            actions=actions,
            timings=timings,
            total_act_ms=timings["total"],
            fallback_reason=state.fallback_reason,
            errors=errors,
            episode_id=self._episode_id,
            llm_stats=self._llm_commander.stats(),
            fire_eligibility=state.fire_eligibility,
        )
        self._diagnostics.record(trace)

    def _log_belief_update(self, observation, memory_snapshot, situation, belief):
        tracks = {}
        for target_id, track in sorted((getattr(memory_snapshot, "tracks", {}) or {}).items()):
            tracks[target_id] = {
                "status": getattr(track, "status", None),
                "observed": getattr(track, "status", None) == "OBSERVED",
                "target_side": getattr(track, "target_side", None),
                "model": getattr(track, "model", None),
                "altitude_m": _round_or_none(getattr(getattr(track, "position", None), "altitude_m", None)),
                "speed_mps": _round_or_none(_speed_mps(getattr(track, "velocity", None))),
                "heading_deg": _round_or_none(getattr(getattr(track, "attitude", None), "heading_deg", None)),
                "detected_by": list(getattr(track, "detected_by", ()) or ()),
                "track_age_s": _round_or_none(getattr(track, "track_age_s", None)),
                "time_since_last_seen_s": _round_or_none(getattr(track, "time_since_last_seen_s", None)),
            }
        pair_geometry = []
        for item in getattr(situation, "tracks", ()) or ():
            pair = getattr(item, "pair", None)
            pair_geometry.append(
                {
                    "ownship_id": getattr(item, "ownship_id", None),
                    "target_id": getattr(item, "target_id", None),
                    "is_observed": bool(getattr(item, "is_observed", False)),
                    "distance_3d_m": _round_or_none(getattr(pair, "distance_3d_m", None)),
                    "closing_speed_mps": _round_or_none(getattr(pair, "closing_speed_mps", None)),
                    "enemy_alignment": _round_or_none(getattr(pair, "alignment", None)),
                    "own_alignment": _round_or_none(getattr(pair, "own_alignment", None)),
                }
            )
        chain_logger.log_event(
            "belief_update",
            {
                "visible_enemy_ids": sorted(getattr(memory_snapshot, "visible_target_ids", ()) or ()),
                "known_enemy_ids": sorted((getattr(memory_snapshot, "tracks", {}) or {}).keys()),
                "belief_posterior": dict(getattr(belief, "posterior", {}) or {}),
                "belief_report_label": getattr(belief, "report_label", None),
                "belief_entropy": getattr(belief, "normalized_entropy", None),
                "enemy_tracks": tracks,
                "pair_geometry": pair_geometry,
            },
        )
        for event in getattr(observation, "events", ()) or ():
            chain_logger.log_event(
                "combat_event",
                {
                    "event_type": getattr(event, "event_type", getattr(event, "type", None)),
                    "shooter": getattr(event, "shooter", getattr(event, "shooter_id", None)),
                    "target": getattr(event, "target", getattr(event, "target_id", None)),
                    "weapon": getattr(event, "weapon", getattr(event, "weapon_name", None)),
                },
            )

    def _log_manager_decision(self, decision, current_plan, selected_plan):
        metadata = getattr(decision, "metadata", {}) or {}
        chain_logger.log_event(
            "plan_selection",
            {
                "manager_action": getattr(getattr(decision, "action", None), "value", getattr(decision, "action", None)),
                "manager_reason": getattr(decision, "reason", None),
                "current_plan": chain_logger.plan_summary(current_plan),
                "selected_plan": chain_logger.plan_summary(selected_plan),
                "selected_plan_score": chain_logger.score_summary(self._strategy_manager.current_plan_score),
                "candidate_count": getattr(decision, "candidate_count", None),
                "llm_request_id": getattr(decision, "llm_request_id", None),
                "switch_allowed": getattr(decision, "switch_allowed", None),
                "score_delta": getattr(decision, "score_delta", None),
                "triggers": metadata.get("triggers", []),
                "llm_consumed": metadata.get("llm_consumed", False),
                "llm_wait_reason": metadata.get("llm_wait_reason"),
                "switch_gate": metadata.get("switch_gate"),
            },
        )
        if metadata.get("candidate_audit") is not None:
            chain_logger.log_event(
                "lightweight_simulation",
                {
                    "candidate_policy": metadata.get("candidate_policy"),
                    "generated_rule_count": metadata.get("generated_rule_count"),
                    "generated_llm_count": metadata.get("generated_llm_count"),
                    "validated_count": metadata.get("validated_count"),
                    "dedupe_before_count": metadata.get("dedupe_before_count"),
                    "dedupe_after_count": metadata.get("dedupe_after_count"),
                    "scored_count": metadata.get("scored_count"),
                    "candidate_audit": metadata.get("candidate_audit"),
                    "selected_plan_id": getattr(selected_plan, "plan_id", None),
                },
            )

    def _log_step_actions(self, observation, actions):
        chain_logger.log_event(
            "actions_built",
            {
                "actions": _summarize_actions(actions),
                "own_ammo": {
                    unit.platform_id: _weapon_remaining(unit, "aam_medium")
                    for unit in getattr(observation, "own_units", ()) or ()
                    if unit.platform_id in set(getattr(observation, "controlled_platform_ids", ()) or ())
                },
            },
        )

    def _build_legacy_equivalent_plan(self, observation, memory_snapshot):
        controlled_ids = set(observation.controlled_platform_ids)
        ownships = [
            unit
            for unit in observation.own_units
            if unit.platform_id in controlled_ids
        ]
        enemies = [
            track
            for track in observation.tracks
            if track.target_side != observation.side
        ]
        if not enemies:
            return TeamPlan(
                plan_id="baseline_legacy_no_target",
                created_step=observation.step_index,
                created_sim_time=observation.sim_time,
                mode=StrategyMode.PEER,
                tactic=Tactic.DISENGAGE,
                roles={ownship.platform_id: Role.DEFENDER for ownship in ownships},
                target_assignments={ownship.platform_id: None for ownship in ownships},
                primary_target=None,
                valid_for_steps=1,
                source=PlanSource.BASELINE,
                rationale=["legacy baseline no-target safe cruise"],
                metadata={"legacy_no_target": True},
            )

        target_assignments = {}
        for ownship in ownships:
            target, _ = min(
                (
                    (track, _distance_m(ownship.position, track.position))
                    for track in enemies
                ),
                key=lambda item: item[1],
            )
            target_assignments[ownship.platform_id] = target.target_id

        tactic = Tactic.SEPARATE_ATTACK
        primary_target = None
        if target_assignments and len(set(target_assignments.values())) == 1:
            tactic = Tactic.FOCUS_FIRE
            primary_target = next(iter(target_assignments.values()))
        return TeamPlan(
            plan_id="baseline_legacy_attack",
            created_step=observation.step_index,
            created_sim_time=observation.sim_time,
            mode=StrategyMode.PEER,
            tactic=tactic,
            roles={ownship.platform_id: Role.PRESSER for ownship in ownships},
            target_assignments=target_assignments,
            primary_target=primary_target,
            valid_for_steps=1,
            source=PlanSource.BASELINE,
            rationale=["legacy baseline nearest-target assignment"],
            metadata={"legacy_baseline": True},
        )

    @staticmethod
    def _flight_action(platform_id, heading, altitude, mach):
        return {
            "type": "set_flight",
            "platform_id": platform_id,
            "heading_deg": heading % 360,
            "altitude_m": altitude,
            "mach": mach,
        }


def _clamp(value, low, high):
    return max(low, min(high, value))


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0


def _speed_mps(velocity):
    if velocity is None:
        return None
    return (
        velocity.north_mps * velocity.north_mps
        + velocity.east_mps * velocity.east_mps
        + velocity.up_mps * velocity.up_mps
    ) ** 0.5


def _round_or_none(value):
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _summarize_actions(action_batch):
    if action_batch is None:
        return []
    if hasattr(action_batch, "model_dump"):
        data = action_batch.model_dump()
    elif hasattr(action_batch, "dict"):
        data = action_batch.dict()
    else:
        data = action_batch
    actions = data.get("actions", []) if isinstance(data, dict) else []
    output = []
    for action in actions:
        if not isinstance(action, dict):
            if hasattr(action, "model_dump"):
                action = action.model_dump()
            elif hasattr(action, "dict"):
                action = action.dict()
            else:
                continue
        output.append(
            {
                key: value
                for key, value in {
                    "type": action.get("type"),
                    "platform_id": action.get("platform_id"),
                    "target_id": action.get("target_id"),
                    "guider_id": action.get("guider_id"),
                    "heading_deg": _round_or_none(action.get("heading_deg")),
                    "altitude_m": _round_or_none(action.get("altitude_m")),
                    "mach": _round_or_none(action.get("mach")),
                }.items()
                if value is not None
            }
        )
    return output


def _distance_m(a, b):
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    horizontal = EARTH_RADIUS_M * 2 * math.asin(math.sqrt(h))
    vertical = a.altitude_m - b.altitude_m
    return math.hypot(horizontal, vertical)


def _bearing_deg(a, b):
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class _ActState:
    def __init__(self):
        self.memory_snapshot = None
        self.situation = None
        self.belief = None
        self.rule_candidates = []
        self.current_plan_before = None
        self.current_plan_score = None
        self.llm_result = None
        self.decision = None
        self.selected_plan = None
        self.selected_plan_score = None
        self.fallback_reason = None
        self.fire_eligibility = []


class _EmptyMemorySnapshot:
    visible_target_ids = ()
    tracks = {}


def _elapsed_ms(start):
    return (time.perf_counter() - start) * 1000.0


def _error_text(error):
    text = f"{type(error).__name__}: {error}"
    return text[:500] if len(text) <= 500 else text[:497] + "..."


def _episode_id(context):
    for name in ("episode_id", "scenario_id", "run_id"):
        value = getattr(context, name, None)
        if value is not None:
            return str(value)
    return "episode"
