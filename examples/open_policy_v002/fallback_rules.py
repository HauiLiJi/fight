from air_combat_challenge.competition.models import ActionBatchV1

from . import settings
from .bvr import fighter_tracks, in_launch_envelope
from .geometry import bearing_deg, clamp, heading_offset_deg, select_nearest_track, unit_risk_score


def fallback_action_batch(observation, last_launch_time, bvr_controller=None, profile="standard"):
    units = sorted(observation.own_units, key=lambda unit: unit.platform_id)
    tracks = fighter_tracks(observation)
    if not units:
        return ActionBatchV1()
    if not tracks:
        return ActionBatchV1.model_validate(
            {"actions": [_flight(unit.platform_id, unit.attitude.heading_deg, clamp(unit.position.altitude_m, 7000.0, 9500.0), 0.72) for unit in units]}
        )

    assignments = assign_targets(units, tracks)
    actions = []
    for index, unit in enumerate(units):
        target = assignments.get(unit.platform_id)
        if target is None:
            continue
        risk = unit_risk_score(unit, tracks, units, index)["score"]
        defensive = profile == "defensive" or (
            profile == "standard" and risk >= settings.HIGH_RISK_SCORE
        )
        if defensive:
            heading = heading_offset_deg(
                bearing_deg(target.position, unit.position),
                -settings.HIGH_RISK_ESCAPE_OFFSET_DEG
                if index == 0
                else settings.HIGH_RISK_ESCAPE_OFFSET_DEG,
            )
            altitude = clamp(unit.position.altitude_m + 300.0, 6500.0, 10500.0)
            mach = 0.92
        else:
            direct_heading = bearing_deg(unit.position, target.position)
            press_offset = 0.0 if profile == "press" else settings.PRESS_OFFSET_DEG
            heading = heading_offset_deg(
                direct_heading, -press_offset if index == 0 else press_offset
            )
            altitude = clamp(target.position.altitude_m + 250.0, 6500.0, 11000.0)
            mach = (
                settings.CONTACT_MACH_LOW_RISK
                if risk < settings.MEDIUM_RISK_SCORE
                else settings.CONTACT_MACH_MED_RISK
            )
            if _can_fire(
                observation.sim_time, unit, target, last_launch_time, bvr_controller
            ):
                actions.append(
                    {"type": "fire", "platform_id": unit.platform_id, "weapon_name": "aam_medium", "target_id": target.target_id}
                )
                last_launch_time[unit.platform_id] = observation.sim_time
                heading = heading_offset_deg(
                    direct_heading,
                    -settings.POST_LAUNCH_OFFSET_DEG
                    if index == 0
                    else settings.POST_LAUNCH_OFFSET_DEG,
                )
                mach = settings.POST_LAUNCH_MACH
        actions.append(_flight(unit.platform_id, heading, altitude, mach))
    return ActionBatchV1.model_validate({"actions": actions})


def assign_targets(units, tracks):
    assignments = {}
    remaining = list(tracks)
    for unit in units:
        pool = remaining or tracks
        target, _ = select_nearest_track(unit, pool)
        assignments[unit.platform_id] = target
        if target in remaining:
            remaining.remove(target)
    if len(tracks) == 1 and len(units) == 2:
        assignments[units[1].platform_id] = tracks[0]
    return assignments


def _can_fire(sim_time, unit, target, last_launch_time, bvr_controller):
    if sim_time - last_launch_time.get(unit.platform_id, -9999.0) <= settings.LAUNCH_COOLDOWN_S:
        return False
    if bvr_controller is not None:
        return bvr_controller.can_fire(sim_time, unit, target)
    return in_launch_envelope(unit, target, settings.NORMAL_HEADING_ERROR_DEG)


def _flight(platform_id, heading, altitude, mach):
    return {"type": "set_flight", "platform_id": platform_id, "heading_deg": heading % 360.0, "altitude_m": altitude, "mach": clamp(mach, 0.2, 2.0)}
