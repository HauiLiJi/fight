import math
from dataclasses import dataclass

from air_combat_challenge.competition.models import ActionBatchV1

from .config import (
    BRACKET_HEADING_OFFSET_DEG,
    CHASE_DISTANCE_THRESHOLD_M,
    CHASE_FAR_MACH,
    CHASE_NEAR_MACH,
    CRUISE_MACH,
    DEFEND_HEADING_OFFSET_DEG,
    DEFEND_MACH,
    DISENGAGE_MACH,
    EARTH_RADIUS_M,
    FIRE_COOLDOWN_S,
    ENGAGEMENT_COLD_RANGE_M,
    ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG,
    ENGAGEMENT_FLANK_RANGE_M,
    ENGAGEMENT_HOT_RANGE_M,
    ENGAGEMENT_MAX_PENDING_SHOTS_PER_SHOOTER,
    ENGAGEMENT_MAX_TEAM_PENDING_SHOTS_PER_TARGET,
    ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG,
    ENGAGEMENT_PENDING_SHOT_TIMEOUT_S,
    FORMATION_MAX_DISTANCE_M,
    HIGH_LOW_ALTITUDE_OFFSET_M,
    NO_TARGET_MAX_ALTITUDE_M,
    NO_TARGET_MIN_ALTITUDE_M,
    PRESS_MACH,
    SUPPORT_ALTITUDE_OFFSET_M,
    SUPPORT_HEADING_OFFSET_DEG,
    TACTICAL_THREAT_ALIGNMENT,
    TACTICAL_THREAT_CLOSING_MPS,
    TACTICAL_THREAT_DISTANCE_M,
    TARGET_ALTITUDE_OFFSET_M,
    TARGET_MAX_ALTITUDE_M,
    TARGET_MIN_ALTITUDE_M,
)
from .engagement_envelope import FireEligibility, evaluate_engagement_geometry
from .chain_logger import log_event
from .strategy import Role, Tactic
from .team_memory import OBSERVED


POST_SHOT_CRANK_S = 25.0
DEFENSIVE_S = 40.0
RECOMMIT_S = 8.0
CRANK_OFFSET_DEG = 60.0
DEFENSIVE_OFFSET_DEG = 165.0
POST_SHOT_MACH = 1.0
DEFENSIVE_MACH_OVERRIDE = 1.15
RECOMMIT_MACH = 0.95


@dataclass(frozen=True)
class FlightGuidance:
    platform_id: str
    target_id: object
    heading_deg: float
    altitude_m: float
    mach: float
    role: object
    tactic: object


@dataclass(frozen=True)
class _PendingShot:
    shot_id: str
    shooter_id: str
    target_id: str
    created_sim_time: float


@dataclass
class _PlatformEngagementState:
    phase: str = "APPROACH"
    phase_since: float = 0.0
    last_target_id: object = None
    last_threat_shooter: object = None
    crank_side: int = 1


class Executor:
    def __init__(self):
        self._last_launch_time = {}
        self._pending_shots = []
        self._platform_states = {}

    def reset(self) -> None:
        self._last_launch_time.clear()
        self._pending_shots.clear()
        self._platform_states.clear()

    def build_actions(self, observation, memory_snapshot, situation, plan) -> ActionBatchV1:
        self._refresh_pending_shots(observation, memory_snapshot)
        ownships = _live_ownships(observation)
        if not ownships:
            return ActionBatchV1()

        guidance_by_platform = self.preview_flight_guidance(
            observation,
            memory_snapshot,
            situation,
            plan,
        )
        actions = []
        for ownship in ownships:
            target_id = plan.target_assignments.get(ownship.platform_id)
            target = memory_snapshot.tracks.get(target_id) if target_id else None
            if target is None or getattr(target, "status", None) != OBSERVED:
                target = _nearest_observed_target(ownship, memory_snapshot)
            guidance = guidance_by_platform.get(ownship.platform_id)
            flight_action = (
                _flight_action(
                    guidance.platform_id,
                    guidance.heading_deg,
                    guidance.altitude_m,
                    guidance.mach,
                )
                if guidance is not None
                else None
            )
            weapon_action = self.build_weapon_action(
                observation,
                memory_snapshot,
                ownship,
                target,
                plan,
            )
            if weapon_action is not None:
                actions.append(weapon_action)
                self._last_launch_time[ownship.platform_id] = observation.sim_time
                self._record_pending_shot(observation, ownship.platform_id, target.target_id)
                self._set_platform_phase(
                    ownship.platform_id,
                    "POST_SHOT_CRANK",
                    observation.sim_time,
                    target_id=target.target_id,
                )
            if flight_action is not None:
                actions.append(flight_action)
                log_event(
                    "executor_decision",
                    {
                        "platform": ownship.platform_id,
                        "role": getattr(getattr(guidance, "role", None), "value", getattr(guidance, "role", None)),
                        "tactic": getattr(getattr(guidance, "tactic", None), "value", getattr(guidance, "tactic", None)),
                        "target": getattr(target, "target_id", None),
                        "phase": self._platform_phase(ownship.platform_id),
                        "flight_action": dict(flight_action),
                    },
                )
        return ActionBatchV1.model_validate({"actions": actions})

    def preview_flight_guidance(
        self,
        observation,
        memory_snapshot,
        situation,
        plan,
    ) -> dict:
        ownships = _live_ownships(observation)
        guidance = {}
        for index, ownship in enumerate(ownships):
            target_id = plan.target_assignments.get(ownship.platform_id)
            target = memory_snapshot.tracks.get(target_id) if target_id else None
            action = self.build_flight_action(
                ownship,
                index,
                ownships,
                target,
                observation,
                memory_snapshot,
                situation,
                plan,
            )
            if action is None:
                continue
            guidance[ownship.platform_id] = FlightGuidance(
                platform_id=ownship.platform_id,
                target_id=target_id,
                heading_deg=action["heading_deg"],
                altitude_m=action["altitude_m"],
                mach=action["mach"],
                role=plan.roles.get(ownship.platform_id),
                tactic=plan.tactic,
            )
        return guidance

    def can_fire(self, observation, memory_snapshot, ownship, target):
        del memory_snapshot
        return self.evaluate_fire_eligibility(
            observation,
            ownship,
            target,
            ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG,
            require_direct_detection=True,
        ).eligible

    def evaluate_fire_eligibility(
        self,
        observation,
        ownship,
        target,
        heading_error_limit_deg,
        require_direct_detection=True,
    ):
        platform_id = getattr(ownship, "platform_id", None)
        target_id = getattr(target, "target_id", None)
        detected_by = tuple(getattr(target, "detected_by", ()) or ()) if target is not None else ()
        target_observed = bool(target is not None and getattr(target, "status", None) == OBSERVED)
        target_live = bool(target is not None and getattr(target, "alive", True))
        detected_by_self = bool(platform_id in detected_by)
        has_weapon = _weapon_available(ownship, "aam_medium")
        elapsed = float(getattr(observation, "sim_time", 0.0)) - float(self._last_launch_time.get(platform_id, -9999.0))
        cooldown_remaining = max(0.0, FIRE_COOLDOWN_S - elapsed)
        cooldown_ready = cooldown_remaining <= 0.0
        pending_ids = tuple(
            shot.shot_id
            for shot in self._pending_shots
            if shot.shooter_id == platform_id or (target_id is not None and shot.target_id == target_id)
        )
        shooter_pending = sum(1 for shot in self._pending_shots if shot.shooter_id == platform_id)
        target_pending = sum(1 for shot in self._pending_shots if target_id is not None and shot.target_id == target_id)
        pending_shot_clear = (
            shooter_pending < ENGAGEMENT_MAX_PENDING_SHOTS_PER_SHOOTER
            and target_pending < ENGAGEMENT_MAX_TEAM_PENDING_SHOTS_PER_TARGET
        )
        geometry = None
        if target is not None and hasattr(ownship, "position") and hasattr(target, "position"):
            geometry = evaluate_engagement_geometry(
                ownship.position,
                ownship.attitude.heading_deg,
                target.position,
                target.attitude.heading_deg,
            )

        reasons = []
        if target is None:
            reasons.append("no_target")
        if not target_observed:
            reasons.append("target_not_observed")
        if not target_live:
            reasons.append("target_not_live")
        if require_direct_detection and not detected_by_self:
            reasons.append("not_detected_by_self")
        if not has_weapon:
            reasons.append("no_aam_medium")
        if not cooldown_ready:
            reasons.append("cooldown")
        if shooter_pending >= ENGAGEMENT_MAX_PENDING_SHOTS_PER_SHOOTER:
            reasons.append("pending_shooter_limit")
        if target_pending >= ENGAGEMENT_MAX_TEAM_PENDING_SHOTS_PER_TARGET:
            reasons.append("pending_target_limit")
        if geometry is None:
            reasons.append("missing_geometry")
        else:
            if not geometry.within_dynamic_range:
                reasons.append("outside_dynamic_range")
            if geometry.shooter_heading_error_deg > float(heading_error_limit_deg):
                reasons.append("heading_error")

        return FireEligibility(
            eligible=not reasons,
            platform_id=str(platform_id),
            target_id=target_id,
            distance_m=None if geometry is None else geometry.distance_m,
            dynamic_launch_range_m=None if geometry is None else geometry.dynamic_launch_range_m,
            aspect_deg=None if geometry is None else geometry.target_aspect_deg,
            aspect_class=None if geometry is None else geometry.aspect_class.value,
            heading_error_deg=None if geometry is None else geometry.shooter_heading_error_deg,
            heading_error_limit_deg=float(heading_error_limit_deg),
            target_observed=target_observed,
            detected_by_self=detected_by_self,
            has_weapon=has_weapon,
            cooldown_ready=cooldown_ready,
            cooldown_remaining_s=cooldown_remaining,
            pending_shot_clear=pending_shot_clear,
            ineligible_reasons=tuple(reasons),
            detected_by=detected_by,
            pending_shot_ids=pending_ids,
        )

    def select_guider(self, observation, target, shooter_id):
        if target is None or target.status != OBSERVED:
            return None
        live_ids = {unit.platform_id for unit in _live_ownships(observation)}
        for platform_id in observation.controlled_platform_ids:
            if platform_id == shooter_id:
                continue
            if platform_id in live_ids and platform_id in target.detected_by:
                return platform_id
        return None

    def build_weapon_action(self, observation, memory_snapshot, ownship, target, plan=None):
        del memory_snapshot, plan
        phase = self._platform_phase(ownship.platform_id)
        target_id = getattr(target, "target_id", None)
        if phase in {"POST_SHOT_CRANK", "DEFENSIVE"}:
            self._log_fire_decision(observation, ownship, target, phase, "hold_fire", ["phase_blocks_fire"])
            return None
        if target is None or target.status != OBSERVED:
            self._log_fire_decision(observation, ownship, target, phase, "hold_fire", ["target_not_observed"])
            return None
        if not _weapon_available(ownship, "aam_medium"):
            self._log_fire_decision(observation, ownship, target, phase, "hold_fire", ["no_aam_medium"])
            return None
        geometry = evaluate_engagement_geometry(
            ownship.position,
            ownship.attitude.heading_deg,
            target.position,
            target.attitude.heading_deg,
        )
        if not _within_launch_stage(geometry):
            self._log_fire_decision(
                observation,
                ownship,
                target,
                phase,
                "hold_fire",
                ["outside_dynamic_launch_stage"],
                geometry=geometry,
            )
            return None
        if not self._cooldown_ready_for_distance(
            observation,
            ownship.platform_id,
            geometry.distance_m,
        ):
            self._log_fire_decision(
                observation,
                ownship,
                target,
                phase,
                "hold_fire",
                ["cooldown_not_ready"],
                geometry=geometry,
            )
            return None
        if ownship.platform_id in target.detected_by:
            self._log_fire_decision(
                observation,
                ownship,
                target,
                phase,
                "fire",
                [],
                geometry=geometry,
            )
            return {
                "type": "fire",
                "platform_id": ownship.platform_id,
                "weapon_name": "aam_medium",
                "target_id": target_id,
            }
        guider_id = self.select_guider(observation, target, ownship.platform_id)
        if guider_id is None:
            self._log_fire_decision(
                observation,
                ownship,
                target,
                phase,
                "hold_fire",
                ["no_direct_or_team_detection"],
                geometry=geometry,
            )
            return None
        self._log_fire_decision(
            observation,
            ownship,
            target,
            phase,
            "co_fire",
            [],
            geometry=geometry,
            guider_id=guider_id,
        )
        return {
            "type": "co_fire",
            "platform_id": ownship.platform_id,
            "weapon_name": "aam_medium",
            "target_id": target_id,
            "guider_id": guider_id,
        }

    def _log_fire_decision(
        self,
        observation,
        ownship,
        target,
        phase,
        decision,
        blocked_reasons,
        geometry=None,
        guider_id=None,
    ):
        detected_by = tuple(getattr(target, "detected_by", ()) or ()) if target is not None else ()
        distance_m = None
        dynamic_launch_range_m = None
        if geometry is not None:
            distance_m = round(float(geometry.distance_m), 3)
            dynamic_launch_range_m = round(float(geometry.dynamic_launch_range_m), 3)
        cooldown_required_s = _cooldown_s_for_distance(distance_m) if distance_m is not None else None
        elapsed = float(getattr(observation, "sim_time", 0.0)) - float(self._last_launch_time.get(ownship.platform_id, -9999.0))
        log_event(
            "fire_decision",
            {
                "platform": ownship.platform_id,
                "target": getattr(target, "target_id", None),
                "phase": phase,
                "decision": decision,
                "weapon": "aam_medium",
                "distance_m": distance_m,
                "dynamic_launch_range_m": dynamic_launch_range_m,
                "detected_by_self": ownship.platform_id in detected_by,
                "detected_by_teammate": any(platform_id != ownship.platform_id for platform_id in detected_by),
                "detected_by": list(detected_by),
                "cooldown_ready": cooldown_required_s is None or elapsed >= cooldown_required_s,
                "cooldown_elapsed_s": round(float(elapsed), 3),
                "cooldown_required_s": cooldown_required_s,
                "launch_stage_ok": geometry is not None and _within_launch_stage(geometry),
                "guider_id": guider_id,
                "blocked_reasons": list(blocked_reasons),
            },
        )

    def _cooldown_ready_for_distance(self, observation, platform_id, distance_m):
        elapsed = float(getattr(observation, "sim_time", 0.0)) - float(self._last_launch_time.get(platform_id, -9999.0))
        return elapsed >= _cooldown_s_for_distance(distance_m)

    def fire_eligibility_summary(self, observation, memory_snapshot, plan):
        self._refresh_pending_shots(observation, memory_snapshot)
        if plan is None:
            return []
        tracks = getattr(memory_snapshot, "tracks", {}) or {}
        output = []
        for ownship in _live_ownships(observation):
            target_id = plan.target_assignments.get(ownship.platform_id)
            target = tracks.get(target_id) if target_id else None
            eligibility = self.evaluate_fire_eligibility(
                observation,
                ownship,
                target,
                _heading_error_limit_for_plan(plan, ownship.platform_id),
                require_direct_detection=True,
            )
            output.append(_fire_eligibility_to_dict(eligibility))
        return output

    def _record_pending_shot(self, observation, shooter_id, target_id):
        shot_id = f"{shooter_id}->{target_id}@{float(getattr(observation, 'sim_time', 0.0)):.1f}"
        self._pending_shots.append(
            _PendingShot(
                shot_id=shot_id,
                shooter_id=str(shooter_id),
                target_id=str(target_id),
                created_sim_time=float(getattr(observation, "sim_time", 0.0)),
            )
        )

    def _refresh_pending_shots(self, observation, memory_snapshot):
        sim_time = float(getattr(observation, "sim_time", 0.0))
        tracks = getattr(memory_snapshot, "tracks", {}) or {}
        finished = _finished_shots_from_events(getattr(memory_snapshot, "events_history", ()) or ())
        retained = []
        for shot in self._pending_shots:
            if sim_time - shot.created_sim_time > ENGAGEMENT_PENDING_SHOT_TIMEOUT_S:
                continue
            if shot.shot_id in finished or (shot.shooter_id, shot.target_id) in finished:
                continue
            track = tracks.get(shot.target_id)
            if track is None or getattr(track, "status", None) != OBSERVED:
                continue
            retained.append(shot)
        self._pending_shots = retained

    def build_flight_action(
        self,
        ownship,
        index,
        ownships,
        target,
        observation,
        memory_snapshot,
        situation,
        plan,
    ):
        target = target or _nearest_observed_target(ownship, memory_snapshot)
        phase_action = self._phase_flight_action(
            ownship,
            target,
            observation,
            memory_snapshot,
        )
        if phase_action is not None:
            return phase_action

        if plan.metadata.get("legacy_no_target"):
            return _safe_hold_action(ownship)
        if plan.metadata.get("legacy_baseline"):
            return _press_action(ownship, target)

        if plan.tactic == Tactic.DISENGAGE:
            threat = _nearest_observed_threat(ownship, memory_snapshot)
            if threat is None:
                return _safe_hold_action(ownship, mach=DISENGAGE_MACH)
            heading = (_bearing_deg(ownship.position, threat.position) + 180.0) % 360.0
            return _flight_action(
                ownship.platform_id,
                _formation_limited_heading(heading, ownship, ownships),
                _clamp(ownship.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                DISENGAGE_MACH,
            )
        if plan.tactic == Tactic.BRACKET:
            reference = target or _enemy_centroid_track(situation)
            if reference is None:
                return _safe_hold_action(ownship)
            threat = _high_threat_for_platform(ownship.platform_id, situation, memory_snapshot, exclude_target_id=getattr(reference, "target_id", None))
            if threat is not None:
                heading = _defensive_heading(ownship, threat)
                return _flight_action(
                    ownship.platform_id,
                    _formation_limited_heading(heading, ownship, ownships),
                    _clamp(ownship.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                    DEFEND_MACH,
                )
            side = _bracket_side(plan, ownship.platform_id, index)
            heading = _bearing_deg(ownship.position, reference.position) + side * BRACKET_HEADING_OFFSET_DEG
            return _flight_action(ownship.platform_id, heading, _target_altitude(reference), PRESS_MACH)
        if plan.tactic == Tactic.HIGH_LOW:
            if target is None:
                return _safe_hold_action(ownship)
            offset = -HIGH_LOW_ALTITUDE_OFFSET_M if index % 2 == 0 else HIGH_LOW_ALTITUDE_OFFSET_M
            return _flight_action(
                ownship.platform_id,
                _bearing_deg(ownship.position, target.position),
                _clamp(target.position.altitude_m + offset, TARGET_MIN_ALTITUDE_M, TARGET_MAX_ALTITUDE_M),
                PRESS_MACH,
            )
        if plan.tactic == Tactic.MUTUAL_SUPPORT:
            if target is None:
                return _safe_hold_action(ownship)
            if plan.roles.get(ownship.platform_id) == Role.SUPPORTER:
                teammate_threat = _teammate_high_threat(ownship.platform_id, ownships, situation, memory_snapshot)
                if teammate_threat is not None:
                    return _press_action(ownship, teammate_threat, mach=PRESS_MACH)
                heading = _bearing_deg(ownship.position, target.position) + SUPPORT_HEADING_OFFSET_DEG
                if _max_formation_spacing(ownship, ownships) > FORMATION_MAX_DISTANCE_M:
                    heading = _heading_to_team_centroid(ownship, ownships)
                altitude = _clamp(
                    target.position.altitude_m + SUPPORT_ALTITUDE_OFFSET_M,
                    TARGET_MIN_ALTITUDE_M,
                    TARGET_MAX_ALTITUDE_M,
                )
                return _flight_action(ownship.platform_id, heading, altitude, CRUISE_MACH)
            return _press_action(ownship, target, mach=PRESS_MACH)
        if plan.tactic == Tactic.DEFEND_COUNTER:
            if plan.roles.get(ownship.platform_id) == Role.DEFENDER:
                threat = target or _nearest_observed_threat(ownship, memory_snapshot)
                if threat is None:
                    return _safe_hold_action(ownship, mach=DEFEND_MACH)
                heading = _defensive_heading(ownship, threat)
                return _flight_action(
                    ownship.platform_id,
                    heading,
                    _clamp(ownship.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                    DEFEND_MACH,
                )
            return _press_action(ownship, target, mach=PRESS_MACH)
        if target is None:
            return _safe_hold_action(ownship)
        return _press_action(ownship, target, mach=PRESS_MACH)

    def _phase_flight_action(self, ownship, target, observation, memory_snapshot):
        state = self._update_platform_phase(ownship, target, observation, memory_snapshot)
        if state.phase == "DEFENSIVE":
            threat = _track_by_id(memory_snapshot, state.last_threat_shooter) or target or _nearest_observed_threat(ownship, memory_snapshot)
            if threat is None:
                return _safe_hold_action(ownship, mach=DEFENSIVE_MACH_OVERRIDE)
            heading = (_bearing_deg(ownship.position, threat.position) + DEFENSIVE_OFFSET_DEG) % 360.0
            return _flight_action(
                ownship.platform_id,
                heading,
                _defensive_altitude(ownship),
                DEFENSIVE_MACH_OVERRIDE,
            )
        if state.phase == "POST_SHOT_CRANK":
            reference = target or _track_by_id(memory_snapshot, state.last_target_id) or _nearest_observed_threat(ownship, memory_snapshot)
            if reference is None:
                return _safe_hold_action(ownship, mach=POST_SHOT_MACH)
            heading = _bearing_deg(ownship.position, reference.position) + state.crank_side * CRANK_OFFSET_DEG
            return _flight_action(
                ownship.platform_id,
                heading,
                _clamp(ownship.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                POST_SHOT_MACH,
            )
        if state.phase == "RECOMMIT":
            reference = target or _track_by_id(memory_snapshot, state.last_target_id)
            if reference is None:
                return None
            return _press_action(ownship, reference, mach=RECOMMIT_MACH)
        return None

    def _update_platform_phase(self, ownship, target, observation, memory_snapshot):
        del memory_snapshot
        sim_time = float(getattr(observation, "sim_time", 0.0))
        state = self._platform_state(ownship.platform_id, sim_time)
        incoming = _recent_incoming_shooter(observation, ownship.platform_id, DEFENSIVE_S)
        if incoming is not None:
            if state.phase != "DEFENSIVE" or state.last_threat_shooter != incoming:
                self._set_platform_phase(
                    ownship.platform_id,
                    "DEFENSIVE",
                    sim_time,
                    threat_shooter=incoming,
                    target_id=getattr(target, "target_id", state.last_target_id),
                )
            return self._platform_state(ownship.platform_id, sim_time)

        if state.phase == "DEFENSIVE":
            if sim_time - state.phase_since < DEFENSIVE_S:
                return state
            self._set_platform_phase(ownship.platform_id, "RECOMMIT", sim_time, target_id=state.last_target_id)
            return self._platform_state(ownship.platform_id, sim_time)

        if state.phase == "POST_SHOT_CRANK":
            if sim_time - state.phase_since < POST_SHOT_CRANK_S:
                return state
            self._set_platform_phase(ownship.platform_id, "RECOMMIT", sim_time, target_id=state.last_target_id)
            return self._platform_state(ownship.platform_id, sim_time)

        if state.phase == "RECOMMIT" and sim_time - state.phase_since >= RECOMMIT_S:
            self._set_platform_phase(ownship.platform_id, "APPROACH", sim_time, target_id=state.last_target_id)
            return self._platform_state(ownship.platform_id, sim_time)

        if target is not None and state.last_target_id is None:
            state.last_target_id = target.target_id
        return state

    def _platform_state(self, platform_id, sim_time):
        state = self._platform_states.get(platform_id)
        if state is None:
            state = _PlatformEngagementState(
                phase="APPROACH",
                phase_since=float(sim_time),
                crank_side=_crank_side(platform_id),
            )
            self._platform_states[platform_id] = state
        return state

    def _platform_phase(self, platform_id):
        state = self._platform_states.get(platform_id)
        return state.phase if state is not None else "APPROACH"

    def _set_platform_phase(self, platform_id, phase, sim_time, target_id=None, threat_shooter=None):
        state = self._platform_state(platform_id, sim_time)
        previous = state.phase
        state.phase = phase
        state.phase_since = float(sim_time)
        if target_id is not None:
            state.last_target_id = target_id
        if threat_shooter is not None:
            state.last_threat_shooter = threat_shooter
        if previous != phase:
            log_event(
                "phase_transition",
                {
                    "platform": platform_id,
                    "from": previous,
                    "to": phase,
                    "target": state.last_target_id,
                    "threat_shooter": state.last_threat_shooter,
                    "reason": _phase_transition_reason(previous, phase),
                },
            )


def _live_ownships(observation):
    controlled_ids = set(observation.controlled_platform_ids)
    return [
        unit
        for unit in observation.own_units
        if unit.platform_id in controlled_ids
    ]


def _heading_error_limit_for_plan(plan, platform_id):
    if (
        plan is not None
        and getattr(plan, "tactic", None) == Tactic.DEFEND_COUNTER
        and getattr(plan, "roles", {}).get(platform_id) == Role.PRESSER
    ):
        return ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG
    return ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG


def _fire_eligibility_to_dict(eligibility):
    return {
        "executor_fire_eligible_now": bool(eligibility.eligible),
        "platform_id": eligibility.platform_id,
        "target_id": eligibility.target_id,
        "distance_m": _round_or_none(eligibility.distance_m),
        "dynamic_launch_range_m": _round_or_none(eligibility.dynamic_launch_range_m),
        "aspect_deg": _round_or_none(eligibility.aspect_deg),
        "aspect_class": eligibility.aspect_class,
        "heading_error_deg": _round_or_none(eligibility.heading_error_deg),
        "heading_error_limit_deg": _round_or_none(eligibility.heading_error_limit_deg),
        "target_observed": bool(eligibility.target_observed),
        "detected_by_self": bool(eligibility.detected_by_self),
        "detected_by": list(eligibility.detected_by),
        "has_weapon": bool(eligibility.has_weapon),
        "cooldown_ready": bool(eligibility.cooldown_ready),
        "cooldown_remaining_s": _round_or_none(eligibility.cooldown_remaining_s),
        "pending_shot_clear": bool(eligibility.pending_shot_clear),
        "pending_shot_ids": list(eligibility.pending_shot_ids),
        "ineligible_reasons": list(eligibility.ineligible_reasons),
    }


def _round_or_none(value):
    if value is None:
        return None
    return round(float(value), 3)


def _within_launch_stage(geometry):
    return float(geometry.distance_m) <= float(geometry.dynamic_launch_range_m)


def _cooldown_s_for_distance(distance_m):
    distance = float(distance_m)
    if distance <= 90000.0:
        return 3.0
    if distance <= 120000.0:
        return 8.0
    return 15.0


def _recent_incoming_shooter(observation, platform_id, window_s):
    sim_time = float(getattr(observation, "sim_time", 0.0))
    best = None
    for event in getattr(observation, "events", ()) or ():
        if getattr(event, "event_type", None) != "WeaponFired":
            continue
        if getattr(event, "target", None) != platform_id:
            continue
        event_time = float(getattr(event, "sim_time", sim_time))
        if sim_time - event_time > float(window_s):
            continue
        shooter = getattr(event, "shooter", None)
        if shooter is not None:
            best = shooter
    return best


def _track_by_id(memory_snapshot, target_id):
    if target_id is None:
        return None
    return (getattr(memory_snapshot, "tracks", {}) or {}).get(target_id)


def _defensive_altitude(ownship):
    return _clamp(
        ownship.position.altitude_m - 500.0,
        NO_TARGET_MIN_ALTITUDE_M,
        NO_TARGET_MAX_ALTITUDE_M,
    )


def _crank_side(platform_id):
    text = str(platform_id)
    if text.endswith("_01") or text.endswith("1"):
        return -1
    return 1


def _phase_transition_reason(previous, phase):
    if phase == "POST_SHOT_CRANK":
        return "weapon_fired"
    if phase == "DEFENSIVE":
        return "incoming_missile_detected"
    if previous == "POST_SHOT_CRANK" and phase == "RECOMMIT":
        return "post_shot_crank_timeout"
    if previous == "DEFENSIVE" and phase == "RECOMMIT":
        return "defensive_timeout"
    if previous == "RECOMMIT" and phase == "APPROACH":
        return "recommit_timeout"
    return "phase_update"


def _finished_shots_from_events(events_history):
    finished = set()
    for event in events_history:
        events = event if isinstance(event, (list, tuple)) else (event,)
        for item in events:
            event_type = str(getattr(item, "type", getattr(item, "event_type", ""))).lower()
            if not any(token in event_type for token in ("hit", "miss", "kill", "destroy", "end", "timeout")):
                continue
            shooter_id = getattr(item, "shooter_id", getattr(item, "platform_id", None))
            target_id = getattr(item, "target_id", None)
            weapon_id = getattr(item, "weapon_id", getattr(item, "munition_id", None))
            if weapon_id is not None:
                finished.add(str(weapon_id))
            if shooter_id is not None and target_id is not None:
                finished.add((str(shooter_id), str(target_id)))
    return finished


def _weapon_available(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return bool(weapon.enabled) and int(weapon.count) > 0
    return False


def _safe_hold_action(ownship, mach=CRUISE_MACH):
    return _flight_action(
        ownship.platform_id,
        ownship.attitude.heading_deg,
        _clamp(ownship.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
        mach,
    )


def _press_action(ownship, target, mach=None):
    if target is None:
        return _safe_hold_action(ownship)
    distance = _distance_m(ownship.position, target.position)
    selected_mach = mach
    if selected_mach is None:
        selected_mach = CHASE_FAR_MACH if distance > CHASE_DISTANCE_THRESHOLD_M else CHASE_NEAR_MACH
    return _flight_action(
        ownship.platform_id,
        _bearing_deg(ownship.position, target.position),
        _target_altitude(target),
        selected_mach,
    )


def _target_altitude(target):
    return _clamp(
        target.position.altitude_m + TARGET_ALTITUDE_OFFSET_M,
        TARGET_MIN_ALTITUDE_M,
        TARGET_MAX_ALTITUDE_M,
    )


def _nearest_observed_threat(ownship, memory_snapshot):
    observed = [
        track
        for track in memory_snapshot.tracks.values()
        if track.status == OBSERVED
    ]
    if not observed:
        return None
    return min(observed, key=lambda track: _distance_m(ownship.position, track.position))


def _nearest_observed_target(ownship, memory_snapshot):
    observed = [
        track
        for track in memory_snapshot.tracks.values()
        if track.status == OBSERVED
    ]
    if not observed:
        return None
    return min(observed, key=lambda track: _distance_m(ownship.position, track.position))


def _high_threat_for_platform(platform_id, situation, memory_snapshot, exclude_target_id=None):
    best = None
    for item in situation.tracks:
        if item.ownship_id != platform_id:
            continue
        if exclude_target_id is not None and item.target_id == exclude_target_id:
            continue
        if not item.is_observed:
            continue
        if (
            item.pair.distance_3d_m <= TACTICAL_THREAT_DISTANCE_M
            and item.pair.closing_speed_mps >= TACTICAL_THREAT_CLOSING_MPS
            and item.pair.alignment >= TACTICAL_THREAT_ALIGNMENT
        ):
            score = (
                item.pair.distance_3d_m
                - 100.0 * item.pair.closing_speed_mps
                - 5000.0 * item.pair.alignment
            )
            candidate = (score, item.target_id)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return memory_snapshot.tracks.get(best[1])


def _teammate_high_threat(platform_id, ownships, situation, memory_snapshot):
    for ownship in ownships:
        if ownship.platform_id == platform_id:
            continue
        threat = _high_threat_for_platform(ownship.platform_id, situation, memory_snapshot)
        if threat is not None:
            return threat
    return None


def _bracket_side(plan, platform_id, index):
    sides = plan.metadata.get("bracket_sides", {}) if isinstance(plan.metadata, dict) else {}
    side = sides.get(platform_id)
    if side in {-1, 1, -1.0, 1.0}:
        return float(side)
    return -1.0 if index % 2 == 0 else 1.0


def _defensive_heading(ownship, threat):
    return (_bearing_deg(ownship.position, threat.position) + DEFEND_HEADING_OFFSET_DEG) % 360.0


def _enemy_centroid_track(situation):
    if situation.enemy_centroid is None:
        return None
    return _CentroidTrack(situation.enemy_centroid)


def _max_formation_spacing(ownship, ownships):
    distances = [
        _distance_m(ownship.position, other.position)
        for other in ownships
        if other.platform_id != ownship.platform_id
    ]
    return max(distances) if distances else 0.0


def _formation_limited_heading(heading, ownship, ownships):
    if _max_formation_spacing(ownship, ownships) <= FORMATION_MAX_DISTANCE_M:
        return heading
    return _heading_to_team_centroid(ownship, ownships)


def _heading_to_team_centroid(ownship, ownships):
    others = [
        unit
        for unit in ownships
        if unit.platform_id != ownship.platform_id
    ]
    if not others:
        return ownship.attitude.heading_deg
    latitude = sum(unit.position.latitude for unit in others) / len(others)
    longitude = sum(unit.position.longitude for unit in others) / len(others)
    altitude_m = sum(unit.position.altitude_m for unit in others) / len(others)
    return _bearing_deg(ownship.position, _Position(latitude, longitude, altitude_m))


def _flight_action(platform_id, heading, altitude, mach):
    return {
        "type": "set_flight",
        "platform_id": platform_id,
        "heading_deg": heading % 360.0,
        "altitude_m": _clamp(altitude, 1000.0, 20000.0),
        "mach": _clamp(mach, 0.2, 2.0),
    }


def _clamp(value, low, high):
    return max(low, min(high, value))


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
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


class _Position:
    def __init__(self, latitude, longitude, altitude_m):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude_m = altitude_m


class _CentroidTrack:
    def __init__(self, position):
        self.target_id = None
        self.position = position
