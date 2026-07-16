from air_combat_challenge.competition.models import ActionBatchV1

from .bvr import fighter_tracks, get_bvr_config, in_launch_envelope
from .geometry import (
    bearing_deg,
    clamp,
    heading_offset_deg,
    select_nearest_track,
    unit_risk_score,
)


LAUNCH_COOLDOWN_S = 35.0


def fallback_action_batch(observation, last_launch_time, bvr_controller=None):
    red_units = sorted(
        observation.own_units,
        key=lambda unit: unit.platform_id,
    )
    blue_tracks = fighter_tracks(observation)

    if not red_units:
        return ActionBatchV1()
    if not blue_tracks:
        return ActionBatchV1.model_validate(
            {
                "actions": [
                    _flight_action(
                        unit.platform_id,
                        unit.attitude.heading_deg,
                        clamp(unit.position.altitude_m, 7000.0, 9500.0),
                        0.72,
                    )
                    for unit in red_units
                ]
            }
        )

    assignments = _assign_targets(red_units, blue_tracks)
    actions = []
    for index, unit in enumerate(red_units):
        target = assignments.get(unit.platform_id)
        if target is None:
            continue
        risk = unit_risk_score(unit, blue_tracks, red_units, index)
        if risk["score"] >= 5.0:
            heading = heading_offset_deg(
                bearing_deg(target.position, unit.position),
                -30.0 if index == 0 else 30.0,
            )
            altitude = clamp(unit.position.altitude_m + 300.0, 6500.0, 10500.0)
            mach = 0.92
        else:
            direct_heading = bearing_deg(unit.position, target.position)
            heading = heading_offset_deg(direct_heading, -18.0 if index == 0 else 18.0)
            altitude = clamp(target.position.altitude_m + 250.0, 6500.0, 11000.0)
            mach = 0.9 if risk["score"] < 3.0 else 0.86
            distance = _distance_to(unit, target)
            if _can_fire(observation.sim_time, distance, unit, target, last_launch_time, bvr_controller):
                actions.append(
                    {
                        "type": "fire",
                        "platform_id": unit.platform_id,
                        "weapon_name": "aam_medium",
                        "target_id": target.target_id,
                    }
                )
                last_launch_time[unit.platform_id] = observation.sim_time
                heading = heading_offset_deg(direct_heading, -32.0 if index == 0 else 32.0)
                mach = 0.93

        actions.append(
            _flight_action(
                unit.platform_id,
                heading,
                altitude,
                mach,
            )
        )

    return ActionBatchV1.model_validate({"actions": actions})


def _assign_targets(red_units, blue_tracks):
    assignments = {}
    remaining_tracks = list(blue_tracks)
    for unit in red_units:
        if remaining_tracks:
            nearest_track, _ = select_nearest_track(unit, remaining_tracks)
            assignments[unit.platform_id] = nearest_track
            remaining_tracks.remove(nearest_track)
        else:
            nearest_track, _ = select_nearest_track(unit, blue_tracks)
            assignments[unit.platform_id] = nearest_track
    if len(blue_tracks) == 1 and len(red_units) == 2:
        assignments[red_units[1].platform_id] = blue_tracks[0]
    return assignments


def _flight_action(platform_id, heading, altitude, mach):
    return {
        "type": "set_flight",
        "platform_id": platform_id,
        "heading_deg": heading % 360.0,
        "altitude_m": altitude,
        "mach": clamp(mach, 0.2, 2.0),
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
    from .geometry import distance_m

    return distance_m(unit.position, target.position)
