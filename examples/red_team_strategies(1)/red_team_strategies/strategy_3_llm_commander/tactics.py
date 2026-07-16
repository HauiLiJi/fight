from air_combat_challenge.competition.models import ActionBatchV1

from .bvr import fighter_tracks, get_bvr_config, in_launch_envelope
from .geometry import (
    bearing_deg,
    clamp,
    distance_m,
    formation_slots,
    heading_offset_deg,
    midpoint_position,
    select_nearest_track,
)


TACTICS = {
    "split_targets",
    "focus_fire",
    "bait_left_shoot_right",
    "bait_right_shoot_left",
    "defensive_regroup",
    "search_and_reacquire",
    "finish_damaged_enemy",
    "disengage_and_reset",
}
PLAN_KEYS = {
    "survival": "survival_plan",
    "attack": "attack_plan",
}
LAUNCH_COOLDOWN_S = 30.0
REGROUP_TARGET_SEPARATION_M = 12000.0
REGROUP_MIN_SEPARATION_M = 10000.0
REGROUP_MAX_SEPARATION_M = 14000.0
REGROUP_ANCHOR_LEAD_M = 4000.0
CAUTIOUS_APPROACH_DISTANCE_M = 60000.0
REACQUIRE_ANCHOR_LEAD_M = 6000.0
SEARCH_SWEEP_DEG = 35.0


def validate_tactic_payload(payload):
    if not isinstance(payload, dict):
        return False, "payload_not_dict"
    tactic = payload.get("tactic")
    if tactic not in TACTICS:
        return False, "unknown_tactic"
    duration_steps = payload.get("duration_steps")
    if not isinstance(duration_steps, int) or duration_steps < 3 or duration_steps > 5:
        return False, "invalid_duration_steps"
    if payload.get("risk_bias") not in {"low", "balanced", "high"}:
        return False, "invalid_risk_bias"
    brief = payload.get("brief")
    if not isinstance(brief, str) or not brief.strip() or len(brief) > 80:
        return False, "invalid_brief"
    return True, ""


def validate_plan_bundle(payload):
    if not isinstance(payload, dict):
        return False, "plan_bundle_not_dict"
    for selection, key in PLAN_KEYS.items():
        is_valid, reason = validate_tactic_payload(payload.get(key))
        if not is_valid:
            return False, f"{selection}_{reason}"
    if payload.get("recommended_plan") not in PLAN_KEYS:
        return False, "invalid_recommended_plan"
    recommendation_brief = payload.get("recommendation_brief")
    if (
        not isinstance(recommendation_brief, str)
        or not recommendation_brief.strip()
        or len(recommendation_brief) > 80
    ):
        return False, "invalid_recommendation_brief"
    return True, ""


def translate_tactic_to_actions(
    observation,
    tactic_payload,
    last_launch_time,
    enemy_contact_state=None,
    bvr_controller=None,
):
    red_units = sorted(observation.own_units, key=lambda unit: unit.platform_id)
    blue_tracks = fighter_tracks(observation)
    if not red_units:
        return ActionBatchV1()

    tactic = tactic_payload["tactic"]
    if not blue_tracks:
        if tactic == "disengage_and_reset":
            actions = _disengage_and_reset(red_units, blue_tracks)
        else:
            actions = _search_and_reacquire(
                observation.sim_time,
                red_units,
                enemy_contact_state,
            )
        return ActionBatchV1.model_validate({"actions": actions})

    track_by_id = {track.target_id: track for track in blue_tracks}
    primary_target = track_by_id.get(tactic_payload.get("primary_target", ""))
    secondary_target = track_by_id.get(tactic_payload.get("secondary_target", ""))
    if primary_target is None and blue_tracks:
        primary_target = blue_tracks[0]
    if secondary_target is None and len(blue_tracks) > 1:
        secondary_target = blue_tracks[-1]

    actions = []
    if tactic == "split_targets":
        assignments = _split_assignments(red_units, blue_tracks, primary_target, secondary_target)
        actions.extend(_attack_assignments(observation.sim_time, red_units, assignments, last_launch_time, bvr_controller=bvr_controller))
    elif tactic == "focus_fire":
        assignments = {unit.platform_id: primary_target for unit in red_units if primary_target is not None}
        actions.extend(_attack_assignments(
            observation.sim_time,
            red_units,
            assignments,
            last_launch_time,
            shared_focus=True,
            bvr_controller=bvr_controller,
        ))
    elif tactic == "bait_left_shoot_right":
        actions.extend(_bait_and_shoot(observation.sim_time, red_units, primary_target, secondary_target, last_launch_time, bait_left=True, bvr_controller=bvr_controller))
    elif tactic == "bait_right_shoot_left":
        actions.extend(_bait_and_shoot(observation.sim_time, red_units, primary_target, secondary_target, last_launch_time, bait_left=False, bvr_controller=bvr_controller))
    elif tactic == "defensive_regroup":
        if bvr_controller is not None and _has_bvr_engagement_window(
            observation.sim_time,
            red_units,
            blue_tracks,
            bvr_controller,
        ):
            assignments = _split_assignments(
                red_units,
                blue_tracks,
                primary_target,
                secondary_target,
            )
            actions.extend(
                _attack_assignments(
                    observation.sim_time,
                    red_units,
                    assignments,
                    last_launch_time,
                    bvr_controller=bvr_controller,
                )
            )
        else:
            actions.extend(_defensive_regroup(red_units, blue_tracks, enemy_contact_state))
    elif tactic == "search_and_reacquire":
        actions.extend(
            _search_and_reacquire(
                observation.sim_time,
                red_units,
                enemy_contact_state,
                blue_tracks,
            )
        )
    elif tactic == "finish_damaged_enemy":
        actions.extend(_finish_damaged_enemy(observation.sim_time, red_units, primary_target, secondary_target, blue_tracks, last_launch_time, bvr_controller=bvr_controller))
    elif tactic == "disengage_and_reset":
        actions.extend(_disengage_and_reset(red_units, blue_tracks))
    return ActionBatchV1.model_validate({"actions": actions})


def _split_assignments(red_units, blue_tracks, primary_target, secondary_target):
    assignments = {}
    if len(red_units) == 1:
        assignments[red_units[0].platform_id] = primary_target
        return assignments
    if primary_target is None and blue_tracks:
        primary_target = blue_tracks[0]
    if secondary_target is None:
        secondary_target = primary_target if len(blue_tracks) <= 1 else blue_tracks[-1]
    assignments[red_units[0].platform_id] = primary_target
    assignments[red_units[1].platform_id] = secondary_target
    return assignments


def _has_bvr_engagement_window(sim_time, red_units, blue_tracks, bvr_controller):
    config = get_bvr_config()
    return any(
        bvr_controller.can_fire(sim_time, unit, track, config)
        for unit in red_units
        for track in blue_tracks
    )


def _attack_assignments(sim_time, red_units, assignments, last_launch_time, shared_focus=False, bvr_controller=None):
    actions = []
    for index, unit in enumerate(red_units):
        target = assignments.get(unit.platform_id)
        if target is None:
            continue
        direct_heading = bearing_deg(unit.position, target.position)
        heading = heading_offset_deg(
            direct_heading,
            (-15.0 if index == 0 else 15.0) if shared_focus else (-10.0 if index == 0 else 10.0),
        )
        altitude = clamp(target.position.altitude_m + (200.0 if index == 0 else 350.0), 6500.0, 11000.0)
        distance = _distance_to(unit, target)
        mach = 0.9 if distance > 30000.0 else 0.86
        if _can_fire(sim_time, distance, unit, target, last_launch_time, bvr_controller):
            actions.append(_fire_action(unit.platform_id, target.target_id))
            last_launch_time[unit.platform_id] = sim_time
            heading = heading_offset_deg(direct_heading, -28.0 if index == 0 else 28.0)
            mach = 0.95
        actions.append(_flight_action(unit.platform_id, heading, altitude, mach))
    return actions


def _bait_and_shoot(sim_time, red_units, primary_target, secondary_target, last_launch_time, bait_left, bvr_controller=None):
    actions = []
    if not red_units:
        return actions
    bait_index = 0 if bait_left else min(1, len(red_units) - 1)
    shooter_index = min(1, len(red_units) - 1) if bait_left else 0
    for index, unit in enumerate(red_units):
        target = primary_target if index == bait_index else secondary_target or primary_target
        if target is None:
            continue
        direct_heading = bearing_deg(unit.position, target.position)
        if index == bait_index:
            heading = heading_offset_deg(direct_heading, -40.0 if bait_left else 40.0)
            altitude = clamp(target.position.altitude_m + 100.0, 6500.0, 10500.0)
            mach = 0.96
        else:
            heading = heading_offset_deg(direct_heading, 12.0 if bait_left else -12.0)
            altitude = clamp(target.position.altitude_m + 450.0, 7000.0, 11000.0)
            mach = 0.88
            distance = _distance_to(unit, target)
            if _can_fire(sim_time, distance, unit, target, last_launch_time, bvr_controller):
                actions.append(_fire_action(unit.platform_id, target.target_id))
                last_launch_time[unit.platform_id] = sim_time
                heading = heading_offset_deg(direct_heading, 26.0 if bait_left else -26.0)
                mach = 0.94
        actions.append(_flight_action(unit.platform_id, heading, altitude, mach))
    return actions


def _defensive_regroup(red_units, blue_tracks, enemy_contact_state=None):
    if len(red_units) == 1:
        return _single_defensive_regroup(red_units[0], blue_tracks)

    pair_centre = midpoint_position(red_units[0].position, red_units[1].position)
    nearest_track, nearest_distance = select_nearest_track_from_position(
        pair_centre,
        blue_tracks,
    )
    if nearest_track is None:
        return _search_and_reacquire(0.0, red_units, enemy_contact_state)
    if nearest_distance > CAUTIOUS_APPROACH_DISTANCE_M:
        return _cautious_approach(red_units, pair_centre, nearest_track)

    safe_heading = _safe_regroup_heading(pair_centre, red_units, blue_tracks)
    regroup_altitude = clamp(
        pair_centre.altitude_m + 300.0,
        7000.0,
        11000.0,
    )
    slots = formation_slots(
        red_units,
        safe_heading,
        REGROUP_TARGET_SEPARATION_M,
        REGROUP_ANCHOR_LEAD_M,
        regroup_altitude,
    )
    pair_distance = distance_m(red_units[0].position, red_units[1].position)
    actions = []
    for unit, slot in zip(red_units, slots):
        slot_distance = distance_m(unit.position, slot)
        heading = bearing_deg(unit.position, slot)
        actions.append(
            _flight_action(
                unit.platform_id,
                heading,
                regroup_altitude,
                _regroup_mach(pair_distance, slot_distance),
            )
        )
    return actions


def _cautious_approach(red_units, pair_centre, target):
    heading = bearing_deg(pair_centre, target.position)
    altitude = clamp(pair_centre.altitude_m + 250.0, 7000.0, 11000.0)
    return _formation_actions(
        red_units,
        heading,
        REGROUP_ANCHOR_LEAD_M,
        altitude,
        mach=0.88,
    )


def _search_and_reacquire(sim_time, red_units, enemy_contact_state, blue_tracks=None):
    if not red_units:
        return []
    pair_centre = midpoint_position(red_units[0].position, red_units[-1].position)
    search_target = None
    if blue_tracks:
        search_target, _ = select_nearest_track_from_position(pair_centre, blue_tracks)
        search_target = None if search_target is None else search_target.position
    if search_target is None:
        lost_contacts = (enemy_contact_state or {}).get("lost_contacts", [])
        if lost_contacts:
            search_target = min(
                lost_contacts,
                key=lambda item: distance_m(pair_centre, item["predicted_position"]),
            )["predicted_position"]

    if search_target is None:
        heading = red_units[0].attitude.heading_deg
    else:
        heading = bearing_deg(pair_centre, search_target)
        if any(item.get("prediction_capped") for item in (enemy_contact_state or {}).get("lost_contacts", [])):
            heading = heading_offset_deg(
                heading,
                SEARCH_SWEEP_DEG if int(sim_time // 20.0) % 2 else -SEARCH_SWEEP_DEG,
            )
    altitude = clamp(pair_centre.altitude_m + 250.0, 7000.0, 11000.0)
    return _formation_actions(
        red_units,
        heading,
        REACQUIRE_ANCHOR_LEAD_M,
        altitude,
        mach=0.90,
    )


def _formation_actions(red_units, heading, anchor_lead_m, altitude, mach):
    if len(red_units) == 1:
        return [_flight_action(red_units[0].platform_id, heading, altitude, mach)]
    slots = formation_slots(
        red_units,
        heading,
        REGROUP_TARGET_SEPARATION_M,
        anchor_lead_m,
        altitude,
    )
    return [
        _flight_action(unit.platform_id, bearing_deg(unit.position, slot), altitude, mach)
        for unit, slot in zip(red_units, slots)
    ]


def select_nearest_track_from_position(position, tracks):
    if not tracks:
        return None, float("inf")
    return min(
        ((track, distance_m(position, track.position)) for track in tracks),
        key=lambda item: item[1],
    )


def contact_mode_for_tactic(observation, tactic_payload, enemy_contact_state):
    blue_tracks = fighter_tracks(observation)
    if blue_tracks:
        return "visible_contact"
    if tactic_payload.get("tactic") == "disengage_and_reset":
        return "disengage_no_contact"
    if (enemy_contact_state or {}).get("lost_contact_ids"):
        return "reacquire"
    return "search_no_contact"


def _single_defensive_regroup(unit, blue_tracks):
    fallback_heading = 90.0
    if blue_tracks:
        fallback_heading = bearing_deg(blue_tracks[0].position, unit.position)
    altitude = clamp(unit.position.altitude_m + 400.0, 7000.0, 11000.0)
    return [_flight_action(unit.platform_id, fallback_heading, altitude, 0.92)]


def _safe_regroup_heading(pair_centre, red_units, blue_tracks):
    if not blue_tracks:
        return red_units[0].attitude.heading_deg
    nearest_track = min(
        blue_tracks,
        key=lambda track: distance_m(pair_centre, track.position),
    )
    return bearing_deg(nearest_track.position, pair_centre)


def _regroup_mach(pair_distance, slot_distance):
    if pair_distance > REGROUP_MAX_SEPARATION_M:
        return 0.96
    if pair_distance < REGROUP_MIN_SEPARATION_M:
        return 0.82
    if slot_distance > 8000.0:
        return 0.96
    if slot_distance < 2000.0:
        return 0.82
    return 0.90


def _finish_damaged_enemy(sim_time, red_units, primary_target, secondary_target, blue_tracks, last_launch_time, bvr_controller=None):
    if primary_target is None and blue_tracks:
        primary_target = blue_tracks[0]
    assignments = {}
    if red_units and primary_target is not None:
        assignments[red_units[0].platform_id] = primary_target
    if len(red_units) > 1:
        assignments[red_units[1].platform_id] = secondary_target or primary_target
    return _attack_assignments(
        sim_time,
        red_units,
        assignments,
        last_launch_time,
        shared_focus=secondary_target is None,
        bvr_controller=bvr_controller,
    )


def _disengage_and_reset(red_units, blue_tracks):
    actions = []
    for index, unit in enumerate(red_units):
        nearest_track, _ = select_nearest_track(unit, blue_tracks)
        if nearest_track is None:
            heading = unit.attitude.heading_deg
        else:
            heading = heading_offset_deg(bearing_deg(nearest_track.position, unit.position), -12.0 if index == 0 else 12.0)
        altitude = clamp(unit.position.altitude_m + 500.0, 7500.0, 11000.0)
        actions.append(_flight_action(unit.platform_id, heading, altitude, 0.97))
    return actions


def _flight_action(platform_id, heading, altitude, mach):
    return {
        "type": "set_flight",
        "platform_id": platform_id,
        "heading_deg": heading % 360.0,
        "altitude_m": altitude,
        "mach": clamp(mach, 0.2, 2.0),
    }


def _fire_action(platform_id, target_id):
    return {
        "type": "fire",
        "platform_id": platform_id,
        "weapon_name": "aam_medium",
        "target_id": target_id,
    }


def _can_fire(sim_time, distance, unit, target, last_launch_time, bvr_controller=None):
    if _weapon_remaining(unit, "aam_medium") <= 0:
        return False
    if sim_time - last_launch_time.get(unit.platform_id, -9999.0) <= LAUNCH_COOLDOWN_S:
        return False
    config = get_bvr_config()
    if bvr_controller is not None:
        return bvr_controller.can_fire(sim_time, unit, target, config)
    return in_launch_envelope(unit, target, config, heading_error_deg=28.0)


def _weapon_remaining(unit, weapon_name):
    for weapon in getattr(unit, "weapons", []):
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0


def _distance_to(unit, target):
    return distance_m(unit.position, target.position)
