import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import EARTH_RADIUS_M
from .engagement_envelope import evaluate_engagement_geometry
from .executor import Executor
from .opponent_belief import STATE_NAMES
from .strategy import PlanSource, Role, Tactic, validate_plan
from .team_memory import COASTING, OBSERVED


@dataclass(frozen=True)
class PredictedAircraftState:
    aircraft_id: str
    north_m: float
    east_m: float
    altitude_m: float
    speed_mps: float
    heading_deg: float
    alive: bool
    observed: bool
    target_id: object


@dataclass(frozen=True)
class RolloutFrame:
    time_offset_s: float
    own_states: tuple
    enemy_states: tuple


@dataclass(frozen=True)
class PendingShot:
    shooter_id: str
    target_id: str
    shooter_side: str
    launch_time_s: float
    estimated_time_to_impact_s: float
    hit_probability: float
    active: bool


@dataclass(frozen=True)
class PredictedThreatChain:
    shooter_id: str
    target_id: str
    shooter_side: str
    time_offset_s: float
    time_to_fire_s: float
    launch_probability: float
    hit_risk: float
    pressed_probability: float
    dynamic_launch_range_m: float = 0.0
    aspect_class: str = ""
    heading_error_deg: float = 0.0
    window_probability: float = 0.0


@dataclass(frozen=True)
class HypothesisRollout:
    hypothesis: str
    frames: tuple
    valid: bool
    warnings: tuple
    pending_shots: tuple = ()
    threat_chains: tuple = ()


@dataclass(frozen=True)
class UtilityBreakdown:
    attack_opportunity: float
    own_fire_opportunity: float
    survivability: float
    terminal_survival: float
    exchange_value: float
    coordination: float
    counter_effect: float
    local_advantage: float
    bracket_risk: float
    unpressed_enemy_risk: float
    enemy_fire_window_risk: float
    separation_risk: float
    ammo_waste: float
    duplicate_attack_waste: float
    total: float


@dataclass(frozen=True)
class HypothesisScore:
    hypothesis: str
    probability: float
    utility: float
    breakdown: UtilityBreakdown
    rollout_warnings: tuple
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoredPlan:
    plan: object
    valid: bool
    invalid_reasons: tuple
    final_score: float
    expected_utility: float
    worst_case_utility: float
    switch_cost: float
    hypothesis_scores: tuple
    summary: str


@dataclass(frozen=True)
class ScoringResult:
    scored_plans: tuple
    ranked_plans: tuple
    best_plan: object
    evaluation_time_ms: float


@dataclass(frozen=True)
class _GuidanceIntent:
    target_id: object
    heading_deg: float
    altitude_m: float
    mach: float


class StrategyScorer:
    def __init__(self, params_path=None):
        path = Path(params_path) if params_path else Path(__file__).with_name("scorer_params.json")
        self._params = json.loads(path.read_text(encoding="utf-8"))
        self._executor = Executor()

    def score_candidates(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        candidates,
        current_plan,
    ):
        start = time.perf_counter()
        scored = tuple(
            self._score_plan(
                observation,
                memory_snapshot,
                situation,
                belief,
                candidate,
                current_plan,
            )
            for candidate in candidates
        )
        ranked = tuple(
            sorted(
                (score for score in scored if score.valid),
                key=lambda score: (
                    -score.final_score,
                    -score.expected_utility,
                    -score.worst_case_utility,
                    score.plan.plan_id,
                ),
            )
        )
        return ScoringResult(
            scored_plans=scored,
            ranked_plans=ranked,
            best_plan=ranked[0].plan if ranked else None,
            evaluation_time_ms=(time.perf_counter() - start) * 1000.0,
        )

    def score_current_plan(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        current_plan,
    ):
        start = time.perf_counter()
        scored = self._score_plan(
            observation,
            memory_snapshot,
            situation,
            belief,
            current_plan,
            current_plan,
        )
        return ScoredPlan(
            plan=scored.plan,
            valid=scored.valid,
            invalid_reasons=scored.invalid_reasons,
            final_score=scored.final_score,
            expected_utility=scored.expected_utility,
            worst_case_utility=scored.worst_case_utility,
            switch_cost=scored.switch_cost,
            hypothesis_scores=scored.hypothesis_scores,
            summary=f"{scored.summary}; evaluation_time_ms={(time.perf_counter() - start) * 1000.0:.3f}",
        )

    def _score_plan(
        self,
        observation,
        memory_snapshot,
        situation,
        belief,
        plan,
        current_plan,
    ):
        validation = validate_plan(plan, observation, memory_snapshot, situation)
        if not validation.valid:
            return ScoredPlan(
                plan=plan,
                valid=False,
                invalid_reasons=tuple(validation.errors),
                final_score=float("-inf"),
                expected_utility=float("-inf"),
                worst_case_utility=float("-inf"),
                switch_cost=0.0,
                hypothesis_scores=(),
                summary="invalid plan",
            )
        probabilities = _belief_probabilities(belief)
        hypothesis_scores = []
        for hypothesis in STATE_NAMES:
            rollout = self.rollout(
                observation,
                memory_snapshot,
                situation,
                plan,
                hypothesis,
            )
            reference = self._reference_rollout(
                observation,
                memory_snapshot,
                hypothesis,
            )
            breakdown = self._utility_breakdown(
                observation,
                memory_snapshot,
                plan,
                rollout,
                reference,
            )
            probability = probabilities[hypothesis]
            hypothesis_scores.append(
                HypothesisScore(
                    hypothesis=hypothesis,
                    probability=probability,
                    utility=breakdown.total,
                    breakdown=breakdown,
                    rollout_warnings=rollout.warnings,
                    diagnostics=_rollout_diagnostics(rollout),
                )
            )
        expected = sum(score.probability * score.utility for score in hypothesis_scores)
        worst = min(score.utility for score in hypothesis_scores)
        switch_cost = self._switch_cost(plan, current_plan)
        aggregation = self._params["aggregation"]
        final = (
            float(aggregation["expected_weight"]) * expected
            + float(aggregation["worst_case_weight"]) * worst
            - float(aggregation["switch_cost_weight"]) * switch_cost
        )
        return ScoredPlan(
            plan=plan,
            valid=True,
            invalid_reasons=(),
            final_score=_finite(final),
            expected_utility=_finite(expected),
            worst_case_utility=_finite(worst),
            switch_cost=switch_cost,
            hypothesis_scores=tuple(hypothesis_scores),
            summary=f"expected={expected:.3f} worst={worst:.3f} switch={switch_cost:.3f}",
        )

    def _utility_breakdown(self, observation, memory_snapshot, plan, rollout, reference):
        attack = _attack_opportunity(observation, memory_snapshot, plan, rollout)
        own_fire = _own_fire_opportunity(observation, memory_snapshot, plan, rollout, self._params)
        survivability = _survivability(rollout, self._params)
        terminal = _terminal_survival(rollout, self._params)
        exchange = _exchange_value(rollout, self._params)
        coordination = _coordination(plan, rollout, self._params)
        counter = _counter_effect(rollout, reference, self._params)
        local = _local_advantage(rollout, self._params)
        bracket = _bracket_risk(rollout, self._params)
        unpressed = _unpressed_enemy_risk(plan, rollout, self._params)
        enemy_fire = _enemy_fire_window_risk(rollout, self._params)
        separation = _separation_risk(rollout, self._params)
        ammo = _ammo_waste(observation, memory_snapshot, plan, rollout, self._params)
        duplicate = _duplicate_attack_waste(observation, memory_snapshot, plan, rollout, self._params)
        weights = self._params["utility_weights"]
        total = (
            float(weights["attack_opportunity"]) * attack
            + float(weights["own_fire_opportunity"]) * own_fire
            + float(weights["survivability"]) * survivability
            + float(weights["terminal_survival"]) * terminal
            + float(weights["exchange_value"]) * exchange
            + float(weights["coordination"]) * coordination
            + float(weights["counter_effect"]) * counter
            + float(weights["local_advantage"]) * local
            - float(weights["bracket_risk"]) * bracket
            - float(weights["unpressed_enemy_risk"]) * unpressed
            - float(weights["enemy_fire_window_risk"]) * enemy_fire
            - float(weights["separation_risk"]) * separation
            - float(weights["ammo_waste"]) * ammo
            - float(weights["duplicate_attack_waste"]) * duplicate
        )
        return UtilityBreakdown(
            attack,
            own_fire,
            survivability,
            terminal,
            exchange,
            coordination,
            counter,
            local,
            bracket,
            unpressed,
            enemy_fire,
            separation,
            ammo,
            duplicate,
            _clamp(total, -1.0, 1.0),
        )

    def _switch_cost(self, plan, current_plan):
        if current_plan is None:
            return 0.0
        if _is_initial_baseline_hold(current_plan):
            return 0.0
        weights = self._params["switch_cost"]
        cost = 0.0
        if plan.mode != current_plan.mode:
            cost += float(weights["mode_change"])
        if plan.tactic != current_plan.tactic:
            cost += float(weights["tactic_change"])
        platform_ids = sorted(set(plan.roles) | set(current_plan.roles))
        for platform_id in platform_ids:
            if plan.roles.get(platform_id) != current_plan.roles.get(platform_id):
                cost += float(weights["per_role_change"])
            if plan.target_assignments.get(platform_id) != current_plan.target_assignments.get(platform_id):
                cost += float(weights["per_target_change"])
        return _clamp(cost, 0.0, 1.0)

    def _reference_rollout(self, observation, memory_snapshot, hypothesis):
        origin = _own_centroid_position(observation)
        if origin is None:
            return HypothesisRollout(hypothesis, (), False, ("missing live ownship centroid",))
        own_states = tuple(_own_initial_states(observation, origin))
        enemy_states = tuple(_enemy_initial_states(memory_snapshot, origin))
        frames = [RolloutFrame(0.0, own_states, enemy_states)]
        prediction = self._params["prediction"]
        dt = float(prediction["dt_s"])
        steps = _rollout_steps(self._params)
        for step in range(1, steps + 1):
            own_guidance = {
                state.aircraft_id: _GuidanceIntent(
                    state.target_id,
                    state.heading_deg,
                    state.altitude_m,
                    state.speed_mps / float(prediction["mach_to_mps"]),
                )
                for state in own_states
            }
            warnings = []
            enemy_guidance = self._enemy_response_guidance(
                hypothesis,
                own_states,
                enemy_states,
                warnings,
            )
            own_states = tuple(
                _step_state(state, own_guidance[state.aircraft_id], prediction, dt)
                for state in own_states
            )
            enemy_states = tuple(
                _step_state(state, enemy_guidance.get(state.aircraft_id), prediction, dt)
                for state in enemy_states
            )
            frames.append(RolloutFrame(step * dt, own_states, enemy_states))
        return HypothesisRollout(hypothesis, tuple(frames), True, ())

    def rollout(self, observation, memory_snapshot, situation, plan, hypothesis):
        warnings = []
        if hypothesis not in STATE_NAMES:
            return HypothesisRollout(hypothesis, (), False, (f"unsupported hypothesis: {hypothesis}",))

        origin = _own_centroid_position(observation)
        if origin is None:
            return HypothesisRollout(hypothesis, (), False, ("missing live ownship centroid",))

        own_guidance = self._executor.preview_flight_guidance(
            observation,
            memory_snapshot,
            situation,
            plan,
        )
        own_states = tuple(_own_initial_states(observation, origin))
        enemy_states = tuple(_enemy_initial_states(memory_snapshot, origin))
        if not enemy_states:
            warnings.append("no known enemy tracks for rollout")

        frames = [RolloutFrame(0.0, own_states, enemy_states)]
        prediction = self._params["prediction"]
        dt = float(prediction["dt_s"])
        steps = _rollout_steps(self._params)
        for step in range(1, steps + 1):
            enemy_guidance = self._enemy_response_guidance(
                hypothesis,
                own_states,
                enemy_states,
                warnings,
            )
            own_states = tuple(
                _step_state(state, own_guidance.get(state.aircraft_id), prediction, dt)
                for state in own_states
            )
            enemy_states = tuple(
                _step_state(state, enemy_guidance.get(state.aircraft_id), prediction, dt)
                for state in enemy_states
            )
            frames.append(RolloutFrame(step * dt, own_states, enemy_states))
        valid = not any(_has_bad_numbers(frame) for frame in frames)
        if not valid:
            warnings.append("rollout produced non-finite values")
        pending_shots = _estimate_pending_shots(observation, memory_snapshot, plan, frames, self._params)
        threat_chains = _estimate_prelaunch_threat_chains(observation, memory_snapshot, plan, frames, self._params)
        return HypothesisRollout(hypothesis, tuple(frames), valid, tuple(warnings), tuple(pending_shots), tuple(threat_chains))

    def _enemy_response_guidance(self, hypothesis, own_states, enemy_states, warnings):
        if not enemy_states:
            return {}
        if not own_states:
            warnings.append("missing ownship states for enemy response")
            return {}
        response = self._params["enemy_response"]
        own_slots = list(own_states)
        own_centroid = _centroid_state(own_states)
        guidance = {}

        if hypothesis == "FOCUS_BLUE_1":
            target = own_slots[0]
            for enemy in enemy_states:
                guidance[enemy.aircraft_id] = _intent_to_state(enemy, target, response["focus_mach"])
        elif hypothesis == "FOCUS_BLUE_2":
            target = own_slots[min(1, len(own_slots) - 1)]
            for enemy in enemy_states:
                guidance[enemy.aircraft_id] = _intent_to_state(enemy, target, response["focus_mach"])
        elif hypothesis == "SPLIT_ATTACK":
            for index, enemy in enumerate(enemy_states):
                target = own_slots[index % len(own_slots)]
                guidance[enemy.aircraft_id] = _intent_to_state(enemy, target, response["split_mach"])
        elif hypothesis == "BRACKET":
            for index, enemy in enumerate(enemy_states):
                side = -1.0 if index % 2 == 0 else 1.0
                heading = _bearing_local(enemy, own_centroid) + side * response["bracket_heading_offset_deg"]
                guidance[enemy.aircraft_id] = _GuidanceIntent(None, heading % 360.0, own_centroid.altitude_m, response["split_mach"])
        elif hypothesis == "ATTACK_SUPPORT":
            lead = min(enemy_states, key=lambda enemy: _distance_local(enemy, own_centroid))
            for enemy in enemy_states:
                heading = _bearing_local(enemy, own_centroid)
                mach = response["counter_press_mach"]
                if enemy.aircraft_id != lead.aircraft_id:
                    heading += response["support_heading_offset_deg"]
                    mach = response["split_mach"]
                guidance[enemy.aircraft_id] = _GuidanceIntent(None, heading % 360.0, own_centroid.altitude_m, mach)
        elif hypothesis == "BAIT_COUNTER":
            bait = min(enemy_states, key=lambda enemy: _distance_local(enemy, own_centroid))
            target = min(own_states, key=lambda own: _distance_local(own, bait))
            for enemy in enemy_states:
                if enemy.aircraft_id == bait.aircraft_id:
                    heading = (_bearing_local(enemy, target) + 180.0) % 360.0
                    mach = response["bait_escape_mach"]
                else:
                    heading = _bearing_local(enemy, target)
                    mach = response["counter_press_mach"]
                guidance[enemy.aircraft_id] = _GuidanceIntent(target.aircraft_id, heading, target.altitude_m, mach)
        elif hypothesis == "DISENGAGE":
            for enemy in enemy_states:
                heading = (_bearing_local(enemy, own_centroid) + 180.0) % 360.0
                guidance[enemy.aircraft_id] = _GuidanceIntent(None, heading, enemy.altitude_m, response["disengage_mach"])
        return guidance


def _attack_opportunity(observation, memory_snapshot, plan, rollout):
    if not rollout.frames:
        return 0.0
    initial = rollout.frames[0]
    final = rollout.frames[-1]
    own_by_id_0 = {state.aircraft_id: state for state in initial.own_states}
    own_by_id_1 = {state.aircraft_id: state for state in final.own_states}
    enemy_by_id_0 = {state.aircraft_id: state for state in initial.enemy_states}
    enemy_by_id_1 = {state.aircraft_id: state for state in final.enemy_states}
    weapon_by_id = {
        unit.platform_id: _has_weapon(unit)
        for unit in observation.own_units
    }
    values = []
    for platform_id, target_id in sorted(plan.target_assignments.items()):
        if target_id is None or platform_id not in own_by_id_1 or target_id not in enemy_by_id_1:
            continue
        track = memory_snapshot.tracks.get(target_id)
        observed = 1.0 if track is not None and track.status == OBSERVED else -0.3
        weapon = 1.0 if weapon_by_id.get(platform_id, False) else -0.5
        start_distance = _distance_local(own_by_id_0[platform_id], enemy_by_id_0.get(target_id, enemy_by_id_1[target_id]))
        end_distance = _distance_local(own_by_id_1[platform_id], enemy_by_id_1[target_id])
        distance_gain = _clamp((start_distance - end_distance) / max(start_distance, 1.0), -1.0, 1.0)
        alignment = _alignment_to(own_by_id_1[platform_id], enemy_by_id_1[target_id])
        values.append(_clamp((distance_gain + alignment + observed + weapon) / 4.0, -1.0, 1.0))
    return _average(values, 0.0)


def _survivability(rollout, params):
    risk = _risk_score(rollout, params)
    return _clamp(1.0 - 2.0 * risk, -1.0, 1.0)


def _own_fire_opportunity(observation, memory_snapshot, plan, rollout, params):
    if not rollout.frames:
        return 0.0
    weapon_by_id = {unit.platform_id: _has_weapon(unit) for unit in observation.own_units}
    track_status = {target_id: track.status for target_id, track in memory_snapshot.tracks.items()}
    values = []
    for frame in rollout.frames:
        own_by_id = {state.aircraft_id: state for state in frame.own_states}
        enemy_by_id = {state.aircraft_id: state for state in frame.enemy_states}
        for platform_id, target_id in plan.target_assignments.items():
            if platform_id not in own_by_id or target_id not in enemy_by_id:
                continue
            observed = track_status.get(target_id) == OBSERVED
            coasting = track_status.get(target_id) == COASTING
            values.append(
                _fire_window_estimate(
                    own_by_id[platform_id],
                    enemy_by_id[target_id],
                    bool(weapon_by_id.get(platform_id, False)),
                    observed,
                    coasting,
                    params,
                )[0]
            )
    return _clamp(max(values) if values else 0.0, 0.0, 1.0)


def _terminal_survival(rollout, params):
    if not rollout.frames:
        return 0.0
    own_count = max(len(rollout.frames[0].own_states), 1)
    enemy_count = max(len(rollout.frames[0].enemy_states), 1)
    own_loss, enemy_loss = _expected_losses_from_engagement(rollout, params)
    own_alive = _clamp((own_count - own_loss) / own_count, 0.0, 1.0)
    enemy_alive = _clamp((enemy_count - enemy_loss) / enemy_count, 0.0, 1.0)
    return _clamp(own_alive - enemy_alive, -1.0, 1.0)


def _exchange_value(rollout, params):
    if not rollout.frames:
        return 0.0
    own_count = max(len(rollout.frames[0].own_states), 1)
    enemy_count = max(len(rollout.frames[0].enemy_states), 1)
    own_loss, enemy_loss = _expected_losses_from_engagement(rollout, params)
    return _clamp((enemy_loss / enemy_count) - (own_loss / own_count), -1.0, 1.0)


def _coordination(plan, rollout, params):
    if not rollout.frames:
        return 0.0
    final = rollout.frames[-1]
    thresholds = params["risk_thresholds"]
    spacing_score = 0.0
    if len(final.own_states) >= 2:
        spacing = _distance_local(final.own_states[0], final.own_states[1])
        low = float(thresholds["formation_min_distance_m"])
        high = float(thresholds["formation_max_distance_m"])
        if low <= spacing <= high:
            spacing_score = 1.0
        else:
            spacing_score = -_clamp(min(abs(spacing - low), abs(spacing - high)) / high, 0.0, 1.0)
    role_score = _role_tactic_score(plan)
    target_score = _target_assignment_score(plan)
    support_score = _support_position_score(plan, final, params)
    return _clamp((spacing_score + role_score + target_score + support_score) / 4.0, -1.0, 1.0)


def _counter_effect(rollout, reference, params):
    if not rollout.frames or not reference.frames:
        return 0.0
    risk_improvement = _risk_score(reference, params) - _risk_score(rollout, params)
    bracket_improvement = _bracket_risk(reference, params) - _bracket_risk(rollout, params)
    local_improvement = _local_advantage(rollout, params) - _local_advantage(reference, params)
    return _clamp((risk_improvement + bracket_improvement + local_improvement) / 3.0, -1.0, 1.0)


def _local_advantage(rollout, params):
    if not rollout.frames:
        return 0.0
    final = rollout.frames[-1]
    threshold = float(params["normalization"]["local_advantage_range_m"])
    own_advantages = []
    for enemy in final.enemy_states:
        count = sum(
            1
            for own in final.own_states
            if _distance_local(own, enemy) <= threshold and _alignment_to(own, enemy) > 0.0
        )
        own_advantages.append(_clamp((count - 1.0) / 2.0, -1.0, 1.0))
    enemy_advantages = []
    for own in final.own_states:
        count = sum(
            1
            for enemy in final.enemy_states
            if _distance_local(enemy, own) <= threshold and _alignment_to(enemy, own) > 0.0
        )
        enemy_advantages.append(_clamp((count - 1.0) / 2.0, -1.0, 1.0))
    return _clamp(_average(own_advantages, 0.0) - _average(enemy_advantages, 0.0), -1.0, 1.0)


def _bracket_risk(rollout, params):
    if not rollout.frames:
        return 0.0
    final = rollout.frames[-1]
    observed_enemies = [enemy for enemy in final.enemy_states if enemy.observed]
    if len(observed_enemies) < 2:
        return 0.0
    thresholds = params["risk_thresholds"]
    bearing_threshold = float(params["normalization"]["bracket_bearing_threshold_deg"])
    risks = []
    for own in final.own_states:
        attackers = [
            enemy
            for enemy in observed_enemies
            if _closing_speed(enemy, own) > float(thresholds["dangerous_closure_mps"])
            and _alignment_to(enemy, own) > float(thresholds["strong_alignment"])
        ]
        if len(attackers) < 2:
            risks.append(0.0)
            continue
        bearings = sorted(_bearing_local(own, enemy) for enemy in attackers)
        max_sep = max(_angle_delta_abs(a, b) for a in bearings for b in bearings)
        risks.append(1.0 if max_sep >= bearing_threshold else max_sep / bearing_threshold)
    return _clamp(_average(risks, 0.0), 0.0, 1.0)


def _unpressed_enemy_risk(plan, rollout, params):
    if not rollout.frames:
        return 0.0
    assigned_targets = {target_id for target_id in plan.target_assignments.values() if target_id is not None}
    risks = []
    for frame in rollout.frames:
        for enemy in frame.enemy_states:
            if enemy.aircraft_id in assigned_targets:
                continue
            threat = max(
                (
                    _fire_window_estimate(enemy, own, bool(params["fire_window"].get("enemy_assumed_ammo", True)), own.observed, False, params)[0]
                    for own in frame.own_states
                ),
                default=0.0,
            )
            interception = max(
                (
                    _fire_window_estimate(own, enemy, True, enemy.observed, not enemy.observed, params)[0]
                    for own in frame.own_states
                ),
                default=0.0,
            )
            risks.append(_clamp(threat * (1.0 - 0.7 * interception), 0.0, 1.0))
    return _clamp(max(risks) if risks else 0.0, 0.0, 1.0)


def _enemy_fire_window_risk(rollout, params):
    if not rollout.frames:
        return 0.0
    values = []
    for frame in rollout.frames:
        for enemy in frame.enemy_states:
            for own in frame.own_states:
                values.append(
                    _fire_window_estimate(
                        enemy,
                        own,
                        bool(params["fire_window"].get("enemy_assumed_ammo", True)),
                        own.observed,
                        False,
                        params,
                    )[0]
                )
    return _clamp(max(values) if values else 0.0, 0.0, 1.0)


def _separation_risk(rollout, params):
    if not rollout.frames or len(rollout.frames[-1].own_states) < 2:
        return 0.0
    thresholds = params["risk_thresholds"]
    spacing = _distance_local(rollout.frames[-1].own_states[0], rollout.frames[-1].own_states[1])
    low = float(thresholds["formation_min_distance_m"])
    high = float(thresholds["formation_max_distance_m"])
    if low <= spacing <= high:
        return 0.0
    if spacing < low:
        return _clamp((low - spacing) / low, 0.0, 1.0)
    return _clamp((spacing - high) / high, 0.0, 1.0)


def _ammo_waste(observation, memory_snapshot, plan, rollout, params):
    del params
    waste = []
    weapon_by_id = {
        unit.platform_id: _has_weapon(unit)
        for unit in observation.own_units
    }
    for platform_id, role in plan.roles.items():
        if role == Role.SHOOTER and not weapon_by_id.get(platform_id, False):
            waste.append(1.0)
    for platform_id, target_id in plan.target_assignments.items():
        if target_id is None:
            continue
        track = memory_snapshot.tracks.get(target_id)
        if track is not None and track.status == COASTING:
            waste.append(1.0)
    final = rollout.frames[-1] if rollout.frames else None
    if final is not None:
        assigned = {}
        for platform_id, target_id in plan.target_assignments.items():
            if target_id is not None:
                assigned.setdefault(target_id, []).append(platform_id)
        own_by_id = {state.aircraft_id: state for state in final.own_states}
        enemy_by_id = {state.aircraft_id: state for state in final.enemy_states}
        for target_id, platform_ids in assigned.items():
            if len(platform_ids) < 2 or target_id not in enemy_by_id:
                continue
            good_geometry = sum(
                1
                for platform_id in platform_ids
                if platform_id in own_by_id and _alignment_to(own_by_id[platform_id], enemy_by_id[target_id]) > 0.0
            )
            if good_geometry < 2:
                waste.append(1.0)
    return _clamp(_average(waste, 0.0), 0.0, 1.0)


def _duplicate_attack_waste(observation, memory_snapshot, plan, rollout, params):
    if not rollout.frames:
        return 0.0
    weapon_by_id = {unit.platform_id: _has_weapon(unit) for unit in observation.own_units}
    assigned = {}
    for platform_id, target_id in plan.target_assignments.items():
        if target_id is not None:
            assigned.setdefault(target_id, []).append(platform_id)
    penalties = []
    for target_id, platform_ids in assigned.items():
        if len(platform_ids) < 2:
            continue
        best_window = 0.0
        for frame in rollout.frames:
            own_by_id = {state.aircraft_id: state for state in frame.own_states}
            enemy_by_id = {state.aircraft_id: state for state in frame.enemy_states}
            if target_id not in enemy_by_id:
                continue
            for platform_id in platform_ids:
                if platform_id in own_by_id:
                    best_window = max(
                        best_window,
                        _fire_window_estimate(
                            own_by_id[platform_id],
                            enemy_by_id[target_id],
                            bool(weapon_by_id.get(platform_id, False)),
                            enemy_by_id[target_id].observed,
                            not enemy_by_id[target_id].observed,
                            params,
                        )[0],
                    )
        unpressed = _unpressed_enemy_risk(plan, rollout, params)
        high_coverage = 1.0 if best_window >= float(params["fire_window"]["high_pending_hit_probability"]) else best_window
        penalties.append(_clamp(0.4 * high_coverage + 0.6 * unpressed, 0.0, 1.0))
    return _clamp(_average(penalties, 0.0), 0.0, 1.0)


def _estimate_pending_shots(observation, memory_snapshot, plan, frames, params):
    if not frames:
        return []
    weapon_by_id = {unit.platform_id: _has_weapon(unit) for unit in observation.own_units}
    track_status = {target_id: track.status for target_id, track in memory_snapshot.tracks.items()}
    pending = []
    covered_targets = {}
    for frame in frames:
        own_by_id = {state.aircraft_id: state for state in frame.own_states}
        enemy_by_id = {state.aircraft_id: state for state in frame.enemy_states}
        for platform_id, target_id in plan.target_assignments.items():
            if platform_id not in own_by_id or target_id not in enemy_by_id:
                continue
            probability, time_to_fire = _fire_window_estimate(
                own_by_id[platform_id],
                enemy_by_id[target_id],
                bool(weapon_by_id.get(platform_id, False)),
                track_status.get(target_id) == OBSERVED,
                track_status.get(target_id) == COASTING,
                params,
            )
            if probability <= 0.45:
                continue
            existing = covered_targets.get(("own", target_id), 0.0)
            adjusted_probability = probability * (1.0 - 0.5 * existing)
            pending.append(_pending_shot(platform_id, target_id, "own", frame.time_offset_s + time_to_fire, adjusted_probability, own_by_id[platform_id], enemy_by_id[target_id], params))
            covered_targets[("own", target_id)] = max(existing, probability)
            break
        for enemy in frame.enemy_states:
            for own in frame.own_states:
                probability, time_to_fire = _fire_window_estimate(
                    enemy,
                    own,
                    bool(params["fire_window"].get("enemy_assumed_ammo", True)),
                    own.observed,
                    False,
                    params,
                )
                if probability <= 0.50:
                    continue
                existing = covered_targets.get(("enemy", own.aircraft_id), 0.0)
                adjusted_probability = probability * (1.0 - 0.5 * existing)
                pending.append(_pending_shot(enemy.aircraft_id, own.aircraft_id, "enemy", frame.time_offset_s + time_to_fire, adjusted_probability, enemy, own, params))
                covered_targets[("enemy", own.aircraft_id)] = max(existing, probability)
                break
    return pending


def _estimate_prelaunch_threat_chains(observation, memory_snapshot, plan, frames, params):
    if not frames:
        return []
    fire = params["fire_window"]
    horizon = min(float(fire["threat_prediction_horizon_s"]), frames[-1].time_offset_s)
    threshold = float(fire["predicted_launch_probability_threshold"])
    hit_scale = float(fire["prelaunch_hit_probability_scale"])
    time_discount = max(float(fire["time_to_fire_discount"]), 1.0)
    pressed_discount = _clamp(float(fire.get("pressed_threat_discount", 0.0)), 0.0, 1.0)
    weapon_by_id = {unit.platform_id: _has_weapon(unit) for unit in observation.own_units}
    track_status = {target_id: track.status for target_id, track in memory_snapshot.tracks.items()}
    assignments = {
        platform_id: target_id
        for platform_id, target_id in plan.target_assignments.items()
        if target_id is not None
    }
    own_target_pairs = set(assignments.items())
    assigned_targets = set(assignments.values())
    best = {}
    for frame in frames:
        if frame.time_offset_s > horizon:
            continue
        own_by_id = {state.aircraft_id: state for state in frame.own_states}
        enemy_by_id = {state.aircraft_id: state for state in frame.enemy_states}
        for enemy in frame.enemy_states:
            pressed_probability = _enemy_pressed_probability(enemy, frame.own_states, assigned_targets, params)
            for own in frame.own_states:
                detail = _fire_window_detail(
                    enemy,
                    own,
                    bool(fire.get("enemy_assumed_ammo", True)),
                    own.observed,
                    False,
                    params,
                )
                probability = detail["probability"]
                time_to_fire = detail["time_to_fire_s"]
                if probability <= 0.0 or not math.isfinite(time_to_fire):
                    continue
                launch_probability = _prelaunch_probability(
                    probability,
                    frame.time_offset_s,
                    time_to_fire,
                    time_discount,
                )
                launch_probability *= 1.0 - pressed_discount * pressed_probability
                if launch_probability < threshold:
                    continue
                hit_risk = _clamp(launch_probability * hit_scale, 0.0, 1.0)
                _keep_best_threat_chain(
                    best,
                    PredictedThreatChain(
                        enemy.aircraft_id,
                        own.aircraft_id,
                        "enemy",
                        frame.time_offset_s,
                        time_to_fire,
                        _clamp(launch_probability, 0.0, 1.0),
                        hit_risk,
                        pressed_probability,
                        detail["dynamic_launch_range_m"],
                        detail["aspect_class"],
                        detail["heading_error_deg"],
                        probability,
                    ),
                )
        for platform_id, target_id in own_target_pairs:
            if platform_id not in own_by_id or target_id not in enemy_by_id:
                continue
            target_status = track_status.get(target_id)
            detail = _fire_window_detail(
                own_by_id[platform_id],
                enemy_by_id[target_id],
                bool(weapon_by_id.get(platform_id, False)),
                target_status == OBSERVED,
                target_status == COASTING,
                params,
            )
            probability = detail["probability"]
            time_to_fire = detail["time_to_fire_s"]
            if probability <= 0.0 or not math.isfinite(time_to_fire):
                continue
            launch_probability = _prelaunch_probability(probability, frame.time_offset_s, time_to_fire, time_discount)
            if launch_probability < threshold:
                continue
            hit_risk = _clamp(launch_probability * hit_scale, 0.0, 1.0)
            _keep_best_threat_chain(
                best,
                PredictedThreatChain(
                    platform_id,
                    target_id,
                    "own",
                    frame.time_offset_s,
                    time_to_fire,
                    _clamp(launch_probability, 0.0, 1.0),
                    hit_risk,
                    0.0,
                    detail["dynamic_launch_range_m"],
                    detail["aspect_class"],
                    detail["heading_error_deg"],
                    probability,
                ),
            )
    return list(best.values())


def _enemy_pressed_probability(enemy, own_states, assigned_targets, params):
    if enemy.aircraft_id not in assigned_targets:
        return 0.0
    values = [
        _fire_window_estimate(own, enemy, True, enemy.observed, not enemy.observed, params)[0]
        for own in own_states
    ]
    return _clamp(max(values) if values else 0.0, 0.0, 1.0)


def _prelaunch_probability(window_probability, time_offset_s, time_to_fire_s, time_discount):
    time_factor = math.exp(-max(time_offset_s + time_to_fire_s, 0.0) / time_discount)
    return _clamp(window_probability * time_factor, 0.0, 1.0)


def _keep_best_threat_chain(best, chain):
    key = (chain.shooter_side, chain.shooter_id, chain.target_id)
    previous = best.get(key)
    if previous is None or chain.hit_risk > previous.hit_risk:
        best[key] = chain


def _pending_shot(shooter_id, target_id, shooter_side, launch_time_s, hit_probability, shooter, target, params):
    distance = _distance_local(shooter, target)
    impact = distance / max(float(params["fire_window"]["time_to_impact_scale_s"]) * 100.0, 1.0)
    return PendingShot(
        shooter_id=shooter_id,
        target_id=target_id,
        shooter_side=shooter_side,
        launch_time_s=launch_time_s,
        estimated_time_to_impact_s=impact,
        hit_probability=_clamp(hit_probability * float(params["fire_window"]["hit_probability_scale"]), 0.0, 1.0),
        active=True,
    )


def _expected_losses_from_pending(rollout, params):
    horizon = rollout.frames[-1].time_offset_s if rollout.frames else 0.0
    own_loss_by_id = {}
    enemy_loss_by_id = {}
    for shot in rollout.pending_shots:
        if not shot.active:
            continue
        if shot.launch_time_s + shot.estimated_time_to_impact_s > horizon + 1.0:
            continue
        target = own_loss_by_id if shot.shooter_side == "enemy" else enemy_loss_by_id
        previous = target.get(shot.target_id, 0.0)
        target[shot.target_id] = 1.0 - (1.0 - previous) * (1.0 - shot.hit_probability)
    return sum(own_loss_by_id.values()), sum(enemy_loss_by_id.values())


def _expected_losses_from_engagement(rollout, params):
    horizon = rollout.frames[-1].time_offset_s if rollout.frames else 0.0
    own_loss_by_id = {}
    enemy_loss_by_id = {}
    for shot in rollout.pending_shots:
        if not shot.active:
            continue
        if shot.launch_time_s + shot.estimated_time_to_impact_s > horizon + 1.0:
            continue
        target = own_loss_by_id if shot.shooter_side == "enemy" else enemy_loss_by_id
        previous = target.get(shot.target_id, 0.0)
        target[shot.target_id] = 1.0 - (1.0 - previous) * (1.0 - shot.hit_probability)
    for chain in getattr(rollout, "threat_chains", ()) or ():
        target = own_loss_by_id if chain.shooter_side == "enemy" else enemy_loss_by_id
        previous = target.get(chain.target_id, 0.0)
        target[chain.target_id] = 1.0 - (1.0 - previous) * (1.0 - chain.hit_risk)
    return sum(own_loss_by_id.values()), sum(enemy_loss_by_id.values())


def _fire_window_estimate(attacker, target, has_ammo, observed, coasting, params):
    if not attacker.alive or not target.alive or not has_ammo:
        return 0.0, float("inf")
    fire = params["fire_window"]
    geometry = _engagement_geometry_local(attacker, target)
    distance = geometry.distance_m
    launch_range = max(float(geometry.dynamic_launch_range_m), 1.0)
    near = min(float(fire["near_range_m"]), launch_range * 0.65)
    if distance > launch_range * 1.35:
        return 0.0, float("inf")
    distance_score = _clamp((launch_range * 1.15 - distance) / max(launch_range * 1.15 - near, 1.0), 0.0, 1.0)
    closing_score = _clamp(_closing_speed(attacker, target) / max(float(fire["closing_scale_mps"]), 1.0), 0.0, 1.0)
    heading_error_limit = float(fire.get("heading_error_max_deg", 28.0))
    heading_score = _clamp((heading_error_limit * 1.5 - geometry.shooter_heading_error_deg) / max(heading_error_limit * 1.5, 1.0), 0.0, 1.0)
    if geometry.shooter_heading_error_deg > heading_error_limit * 2.5:
        heading_score = 0.0
    visibility = 1.0 if observed else (float(fire["coasting_multiplier"]) if coasting else 0.0)
    in_range_bonus = 0.15 if geometry.within_dynamic_range else 0.0
    probability = _clamp((0.50 * distance_score + 0.20 * closing_score + 0.30 * heading_score + in_range_bonus) * visibility, 0.0, 1.0)
    time_to_fire = (1.0 - probability) * float(fire["time_to_fire_scale_s"])
    return probability, time_to_fire


def _fire_window_detail(attacker, target, has_ammo, observed, coasting, params):
    probability, time_to_fire = _fire_window_estimate(attacker, target, has_ammo, observed, coasting, params)
    geometry = _engagement_geometry_local(attacker, target)
    return {
        "probability": probability,
        "time_to_fire_s": time_to_fire,
        "dynamic_launch_range_m": geometry.dynamic_launch_range_m,
        "aspect_class": geometry.aspect_class.value,
        "heading_error_deg": geometry.shooter_heading_error_deg,
        "distance_m": geometry.distance_m,
        "within_launch_envelope": bool(
            geometry.within_dynamic_range
            and geometry.shooter_heading_error_deg <= float(params["fire_window"].get("heading_error_max_deg", 28.0))
        ),
    }


def _engagement_geometry_local(attacker, target):
    return evaluate_engagement_geometry(
        _local_position(attacker),
        attacker.heading_deg,
        _local_position(target),
        target.heading_deg,
    )


def _risk_score(rollout, params):
    if not rollout.frames:
        return 0.0
    thresholds = params["risk_thresholds"]
    risks = []
    for frame in rollout.frames:
        for own in frame.own_states:
            simultaneous = 0
            for enemy in frame.enemy_states:
                distance = _distance_local(enemy, own)
                closing = _closing_speed(enemy, own)
                alignment = _alignment_to(enemy, own)
                risk = 0.0
                if distance < float(thresholds["dangerous_distance_m"]):
                    risk += (float(thresholds["dangerous_distance_m"]) - distance) / float(thresholds["dangerous_distance_m"])
                if closing > float(thresholds["dangerous_closure_mps"]):
                    risk += closing / max(float(thresholds["dangerous_closure_mps"]) * 2.0, 1.0)
                if alignment > float(thresholds["strong_alignment"]):
                    risk += alignment
                if risk > 0.0:
                    simultaneous += 1
                risks.append(_clamp(risk / 3.0, 0.0, 1.0))
            if simultaneous > 1:
                risks.append(_clamp((simultaneous - 1.0) / 2.0, 0.0, 1.0))
    return _clamp(_average(risks, 0.0), 0.0, 1.0)


def _role_tactic_score(plan):
    roles = set(plan.roles.values())
    if plan.tactic == Tactic.DEFEND_COUNTER:
        return 1.0 if {Role.DEFENDER, Role.PRESSER}.issubset(roles) else -1.0
    if plan.tactic == Tactic.FOCUS_FIRE:
        return 1.0 if roles & {Role.PRESSER, Role.SHOOTER, Role.SUPPORTER, Role.TRACK_HOLDER} else -0.5
    if plan.tactic == Tactic.MUTUAL_SUPPORT:
        return 1.0 if Role.SUPPORTER in roles else -0.5
    return 0.5


def _is_initial_baseline_hold(plan):
    metadata = getattr(plan, "metadata", {}) or {}
    return (
        getattr(plan, "source", None) == PlanSource.BASELINE
        and (
            bool(metadata.get("legacy_no_target"))
            or getattr(plan, "plan_id", "") == "manager_initial_safe_hold"
        )
    )


def _target_assignment_score(plan):
    targets = [target for target in plan.target_assignments.values() if target is not None]
    if plan.tactic == Tactic.FOCUS_FIRE:
        return 1.0 if targets and len(set(targets)) == 1 else -1.0
    if plan.tactic == Tactic.SEPARATE_ATTACK:
        return 1.0 if len(set(targets)) == len(targets) else -0.5
    return 0.5


def _support_position_score(plan, frame, params):
    supporters = [pid for pid, role in plan.roles.items() if role in {Role.SUPPORTER, Role.TRACK_HOLDER}]
    if not supporters:
        return 0.0
    own_by_id = {state.aircraft_id: state for state in frame.own_states}
    threshold = float(params["risk_thresholds"]["formation_max_distance_m"])
    values = []
    for supporter in supporters:
        if supporter not in own_by_id:
            continue
        distances = [
            _distance_local(own_by_id[supporter], other)
            for pid, other in own_by_id.items()
            if pid != supporter
        ]
        if distances:
            values.append(1.0 - _clamp(min(distances) / threshold, 0.0, 1.0))
    return _average(values, 0.0)


def _belief_probabilities(belief):
    posterior = getattr(belief, "posterior", {}) or {}
    if not posterior:
        value = 1.0 / len(STATE_NAMES)
        return {state: value for state in STATE_NAMES}
    total = sum(max(float(posterior.get(state, 0.0)), 0.0) for state in STATE_NAMES)
    if total <= 0.0:
        value = 1.0 / len(STATE_NAMES)
        return {state: value for state in STATE_NAMES}
    return {state: max(float(posterior.get(state, 0.0)), 0.0) / total for state in STATE_NAMES}


def _rollout_diagnostics(rollout):
    own_shots = [shot for shot in rollout.pending_shots if shot.shooter_side == "own"]
    enemy_shots = [shot for shot in rollout.pending_shots if shot.shooter_side == "enemy"]
    own_loss, enemy_loss = _expected_losses_from_pending(rollout, {"fire_window": {"high_pending_hit_probability": 0.55}})
    engagement_own_loss, engagement_enemy_loss = _expected_losses_from_engagement(rollout, {"fire_window": {"high_pending_hit_probability": 0.55}})
    own_chains = [chain for chain in getattr(rollout, "threat_chains", ()) if chain.shooter_side == "own"]
    enemy_chains = [chain for chain in getattr(rollout, "threat_chains", ()) if chain.shooter_side == "enemy"]
    return {
        "own_pending_shots": len(own_shots),
        "enemy_pending_shots": len(enemy_shots),
        "max_own_hit_probability": max((shot.hit_probability for shot in own_shots), default=0.0),
        "max_enemy_hit_probability": max((shot.hit_probability for shot in enemy_shots), default=0.0),
        "expected_own_losses": own_loss,
        "expected_enemy_losses": enemy_loss,
        "prelaunch_own_expected_losses": max(0.0, engagement_own_loss - own_loss),
        "prelaunch_enemy_expected_losses": max(0.0, engagement_enemy_loss - enemy_loss),
        "engagement_own_expected_losses": engagement_own_loss,
        "engagement_enemy_expected_losses": engagement_enemy_loss,
        "enemy_threat_chain_count": len(enemy_chains),
        "own_threat_chain_count": len(own_chains),
        "max_enemy_launch_probability": max((chain.launch_probability for chain in enemy_chains), default=0.0),
        "max_own_launch_probability": max((chain.launch_probability for chain in own_chains), default=0.0),
        "max_enemy_fire_window_probability": max((chain.window_probability for chain in enemy_chains), default=0.0),
        "max_own_fire_window_probability": max((chain.window_probability for chain in own_chains), default=0.0),
        "enemy_threat_chains": _top_threat_chains(enemy_chains),
        "own_threat_chains": _top_threat_chains(own_chains),
    }


def _top_threat_chains(chains, limit=6):
    return [
        {
            "shooter_id": chain.shooter_id,
            "target_id": chain.target_id,
            "time_offset_s": _finite(chain.time_offset_s),
            "time_to_fire_s": _finite(chain.time_to_fire_s),
            "launch_probability": _finite(chain.launch_probability),
            "hit_risk": _finite(chain.hit_risk),
            "pressed_probability": _finite(chain.pressed_probability),
            "predicted_fire_window_probability": _finite(chain.window_probability),
            "dynamic_launch_range_m": _finite(chain.dynamic_launch_range_m),
            "aspect_class": chain.aspect_class,
            "heading_error_deg": _finite(chain.heading_error_deg),
        }
        for chain in sorted(chains, key=lambda item: (-item.hit_risk, item.shooter_id, item.target_id))[:limit]
    ]


def _rollout_steps(params):
    prediction = params["prediction"]
    dt = float(prediction["dt_s"])
    horizon = max(
        float(prediction["horizon_s"]),
        float(params.get("fire_window", {}).get("threat_prediction_horizon_s", 0.0)),
    )
    return int(round(horizon / dt))


def _step_state(state, guidance, prediction, dt):
    if guidance is None:
        return state
    heading = _step_heading(
        state.heading_deg,
        guidance.heading_deg,
        float(prediction["max_turn_rate_deg_s"]) * dt,
    )
    altitude_delta = guidance.altitude_m - state.altitude_m
    if altitude_delta >= 0.0:
        altitude_delta = min(altitude_delta, float(prediction["max_climb_rate_mps"]) * dt)
    else:
        altitude_delta = max(altitude_delta, -float(prediction["max_descent_rate_mps"]) * dt)
    altitude = state.altitude_m + altitude_delta
    target_speed = _clamp(
        guidance.mach * float(prediction["mach_to_mps"]),
        float(prediction["min_planning_speed_mps"]),
        float(prediction["max_planning_speed_mps"]),
    )
    speed_delta = _clamp(
        target_speed - state.speed_mps,
        -float(prediction["max_acceleration_mps2"]) * dt,
        float(prediction["max_acceleration_mps2"]) * dt,
    )
    speed = state.speed_mps + speed_delta
    heading_rad = math.radians(heading)
    north = state.north_m + math.cos(heading_rad) * speed * dt
    east = state.east_m + math.sin(heading_rad) * speed * dt
    return PredictedAircraftState(
        aircraft_id=state.aircraft_id,
        north_m=north,
        east_m=east,
        altitude_m=altitude,
        speed_mps=speed,
        heading_deg=heading,
        alive=state.alive,
        observed=state.observed,
        target_id=guidance.target_id,
    )


def _step_heading(current, target, max_delta):
    delta = _signed_angle_delta_deg(current, target)
    delta = _clamp(delta, -max_delta, max_delta)
    return (current + delta) % 360.0


def _own_initial_states(observation, origin):
    controlled = set(observation.controlled_platform_ids)
    states = []
    for unit in observation.own_units:
        if unit.platform_id not in controlled:
            continue
        north, east = local_from_position(unit.position, origin)
        states.append(
            PredictedAircraftState(
                unit.platform_id,
                north,
                east,
                unit.position.altitude_m,
                _speed_mps(unit.velocity),
                unit.attitude.heading_deg % 360.0,
                True,
                True,
                None,
            )
        )
    return states


def _enemy_initial_states(memory_snapshot, origin):
    states = []
    for target_id, track in sorted(memory_snapshot.tracks.items()):
        if track.status not in {OBSERVED, COASTING}:
            continue
        north, east = local_from_position(track.position, origin)
        states.append(
            PredictedAircraftState(
                target_id,
                north,
                east,
                track.position.altitude_m,
                _speed_mps(track.velocity),
                track.attitude.heading_deg % 360.0,
                True,
                track.status == OBSERVED,
                None,
            )
        )
    return states


def _own_centroid_position(observation):
    units = [
        unit
        for unit in observation.own_units
        if unit.platform_id in set(observation.controlled_platform_ids)
    ]
    if not units:
        return None
    return _Position(
        sum(unit.position.latitude for unit in units) / len(units),
        sum(unit.position.longitude for unit in units) / len(units),
        sum(unit.position.altitude_m for unit in units) / len(units),
    )


def local_from_position(position, origin):
    lat_avg = math.radians((position.latitude + origin.latitude) / 2.0)
    north_m = math.radians(position.latitude - origin.latitude) * EARTH_RADIUS_M
    east_m = math.radians(position.longitude - origin.longitude) * EARTH_RADIUS_M * math.cos(lat_avg)
    return north_m, east_m


def position_from_local(north_m, east_m, altitude_m, origin):
    latitude = origin.latitude + math.degrees(north_m / EARTH_RADIUS_M)
    cos_lat = max(1.0e-9, math.cos(math.radians(origin.latitude)))
    longitude = origin.longitude + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return _Position(latitude, longitude, altitude_m)


def _speed_mps(velocity):
    return math.sqrt(
        velocity.north_mps * velocity.north_mps
        + velocity.east_mps * velocity.east_mps
        + velocity.up_mps * velocity.up_mps
    )


def _intent_to_state(source, target, mach):
    return _GuidanceIntent(
        target.aircraft_id,
        _bearing_local(source, target),
        target.altitude_m,
        mach,
    )


def _centroid_state(states):
    return PredictedAircraftState(
        "centroid",
        sum(state.north_m for state in states) / len(states),
        sum(state.east_m for state in states) / len(states),
        sum(state.altitude_m for state in states) / len(states),
        sum(state.speed_mps for state in states) / len(states),
        0.0,
        True,
        all(state.observed for state in states),
        None,
    )


def _bearing_local(a, b):
    return (math.degrees(math.atan2(b.east_m - a.east_m, b.north_m - a.north_m)) + 360.0) % 360.0


def _distance_local(a, b):
    return math.sqrt((b.north_m - a.north_m) ** 2 + (b.east_m - a.east_m) ** 2 + (b.altitude_m - a.altitude_m) ** 2)


def _closing_speed(source, target):
    distance = _distance_local(source, target)
    if distance <= 0.0:
        return 0.0
    rel_north = math.cos(math.radians(source.heading_deg)) * source.speed_mps
    rel_east = math.sin(math.radians(source.heading_deg)) * source.speed_mps
    target_north = math.cos(math.radians(target.heading_deg)) * target.speed_mps
    target_east = math.sin(math.radians(target.heading_deg)) * target.speed_mps
    unit_north = (target.north_m - source.north_m) / distance
    unit_east = (target.east_m - source.east_m) / distance
    closing = (
        (rel_north - target_north) * unit_north
        + (rel_east - target_east) * unit_east
    )
    return closing


def _alignment_to(source, target):
    bearing = _bearing_local(source, target)
    delta = _angle_delta_abs(source.heading_deg, bearing)
    return math.cos(math.radians(delta))


def _angle_delta_abs(a, b):
    return abs(_signed_angle_delta_deg(a, b))


def _signed_angle_delta_deg(current, target):
    return (target - current + 180.0) % 360.0 - 180.0


def _has_bad_numbers(frame):
    for state in frame.own_states + frame.enemy_states:
        values = (state.north_m, state.east_m, state.altitude_m, state.speed_mps, state.heading_deg)
        if any(not math.isfinite(value) for value in values):
            return True
    return False


def _clamp(value, low, high):
    return max(low, min(high, value))


def _average(values, default):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return default
    return sum(values) / len(values)


def _has_weapon(unit):
    for weapon in unit.weapons:
        if weapon.name == "aam_medium":
            return bool(weapon.enabled) and int(weapon.count) > 0
    return False


def _finite(value):
    if not math.isfinite(value):
        return 0.0
    return value


class _Position:
    def __init__(self, latitude, longitude, altitude_m):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_m = altitude_m


def _local_position(state):
    return _Position(state.north_m / EARTH_RADIUS_M * 180.0 / math.pi, state.east_m / EARTH_RADIUS_M * 180.0 / math.pi, state.altitude_m)
