from .geometry import (
    altitude_delta_m,
    bearing_deg,
    clamp,
    distance_m,
    pair_distance_m,
    unit_risk_score,
)
from .bvr import fighter_tracks, get_bvr_config, in_launch_envelope, launch_range_m

def build_state_summary(
    observation,
    red_force_status=None,
    enemy_contact_state=None,
    missile_threat_state=None,
):
    red_units = sorted(
        observation.own_units,
        key=lambda unit: unit.platform_id,
    )
    blue_tracks = fighter_tracks(observation)
    bvr_config = get_bvr_config()

    red_unit_summaries = []
    for index, unit in enumerate(red_units):
        nearest_track = None
        nearest_distance = None
        if blue_tracks:
            nearest_track = min(
                blue_tracks,
                key=lambda track: distance_m(unit.position, track.position),
            )
            nearest_distance = distance_m(unit.position, nearest_track.position)
        risk = unit_risk_score(
            unit,
            blue_tracks,
            red_units,
            index,
            missile_threat_state,
        )
        red_unit_summaries.append(
            {
                "platform_id": unit.platform_id,
                "alive": True,
                "altitude_m": round(unit.position.altitude_m, 1),
                "speed_mps": round(
                    (unit.velocity.north_mps ** 2 + unit.velocity.east_mps ** 2 + unit.velocity.up_mps ** 2) ** 0.5,
                    1,
                ),
                "medium_missiles": _weapon_remaining(unit, "aam_medium"),
                "short_missiles": _weapon_remaining(unit, "aam_short"),
                "weapons": _weapon_inventory(unit),
                "nearest_enemy_distance": None if nearest_distance is None else round(nearest_distance, 1),
                "nearest_enemy": None if nearest_track is None else nearest_track.target_id,
                "danger": _danger_label(risk["score"]),
                "risk_score": risk["score"],
                "incoming_missile_count": risk["incoming_missile_count"],
            }
        )

    blue_track_summaries = []
    for track in blue_tracks:
        nearest_red = None
        nearest_distance = None
        if red_units:
            nearest_red = min(
                red_units,
                key=lambda unit: distance_m(unit.position, track.position),
            )
            nearest_distance = distance_m(nearest_red.position, track.position)
        blue_track_summaries.append(
            {
                "target_id": track.target_id,
                "distance_to_nearest_red_m": None if nearest_distance is None else round(nearest_distance, 1),
                "visible_to": list(track.detected_by),
                "altitude_m": round(track.position.altitude_m, 1),
                "is_attackable": bool(
                    nearest_red is not None
                    and in_launch_envelope(nearest_red, track, bvr_config)
                ),
                "bvr_launch_range_m": None
                if nearest_red is None
                else round(launch_range_m(nearest_red, track, bvr_config), 1),
                "relative_altitude_to_nearest_red_m": None
                if nearest_red is None
                else round(altitude_delta_m(nearest_red, track), 1),
                "bearing_from_nearest_red_deg": None
                if nearest_red is None
                else round(bearing_deg(nearest_red.position, track.position), 1),
            }
        )

    pair_distance = pair_distance_m(red_units)
    threat_eval = _build_threat_eval(red_units, blue_tracks, missile_threat_state)
    return {
        "sim_time": round(observation.sim_time, 1),
        "red_units": red_unit_summaries,
        "red_force_status": red_force_status or _default_force_status(red_units),
        "red_air_to_air_inventory": _air_to_air_inventory(red_units),
        "blue_contact_status": _contact_status(enemy_contact_state),
        "incoming_missile_threats": _missile_threat_summary(missile_threat_state),
        "bvr_engagement": {
            "known_enemy_medium_missile_launch_range_m": bvr_config.hot_launch_range_m,
            "hot_launch_range_m": bvr_config.hot_launch_range_m,
            "flank_launch_range_m": bvr_config.flank_launch_range_m,
            "cold_launch_range_m": bvr_config.cold_launch_range_m,
        },
        "blue_tracks": blue_track_summaries,
        "pair_geometry": {
            "pair_distance_m": round(pair_distance, 1),
            "too_spread": pair_distance > 30000.0,
            "mutual_support": pair_distance <= 25000.0 if len(red_units) >= 2 else False,
        },
        "threat_eval": threat_eval,
        "engagement_phase": _engagement_phase(red_units, blue_tracks, threat_eval),
    }


def _contact_status(contact_state):
    contact_state = contact_state or {}
    return {
        "known_alive_ids": list(contact_state.get("known_alive_ids", [])),
        "visible_ids": list(contact_state.get("visible_ids", [])),
        "lost_contact_ids": list(contact_state.get("lost_contact_ids", [])),
        "destroyed_ids": list(contact_state.get("destroyed_ids", [])),
        "last_contact_age_s": contact_state.get("last_contact_age_s"),
        "lost_contacts": [
            {
                "target_id": item.get("target_id"),
                "last_seen_sim_time": item.get("last_seen_sim_time"),
                "age_s": item.get("age_s"),
                "prediction_capped": item.get("prediction_capped"),
            }
            for item in contact_state.get("lost_contacts", [])
        ],
    }


def _build_threat_eval(red_units, blue_tracks, missile_threat_state=None):
    if not red_units:
        return {
            "most_threatened_red": "",
            "high_risk": False,
            "double_pressure": False,
        }
    risks = [
        (
            unit.platform_id,
            unit_risk_score(
                unit,
                blue_tracks,
                red_units,
                index,
                missile_threat_state,
            ),
        )
        for index, unit in enumerate(red_units)
    ]
    most_threatened = max(risks, key=lambda item: item[1]["score"])
    return {
        "most_threatened_red": most_threatened[0],
        "high_risk": most_threatened[1]["score"] >= 4.0,
        "double_pressure": any(item[1]["enemy_count_nearby"] >= 2 for item in risks),
        "incoming_missile_count": sum(
            item[1]["incoming_missile_count"] for item in risks
        ),
    }


def _engagement_phase(red_units, blue_tracks, threat_eval):
    if not red_units or not blue_tracks:
        return "regroup"
    nearest_distances = []
    for unit in red_units:
        nearest_distance = min(
            distance_m(unit.position, track.position)
            for track in blue_tracks
        )
        nearest_distances.append(nearest_distance)
    best_distance = min(nearest_distances)
    if threat_eval["high_risk"] or threat_eval.get("incoming_missile_count", 0):
        return "disengage"
    if best_distance > 60000.0:
        return "approach"
    if best_distance > 30000.0:
        return "attack"
    if len(blue_tracks) == 1:
        return "endgame"
    return "attack"


def _missile_threat_summary(missile_threat_state):
    missile_threat_state = missile_threat_state or {}
    return {
        "active_threats": list(missile_threat_state.get("active_threats", [])),
        "threatened_platform_ids": list(
            missile_threat_state.get("threatened_platform_ids", [])
        ),
        "visible_enemy_missile_ids": list(
            missile_threat_state.get("visible_enemy_missile_ids", [])
        ),
    }


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0


def _weapon_inventory(unit):
    return [
        {
            "name": weapon.name,
            "weapon_type": weapon.weapon_type,
            "enabled": weapon.enabled,
            "count": int(weapon.count),
            "time_since_last_fired_s": round(weapon.time_since_last_fired_s, 1),
            "is_air_to_air": weapon.weapon_type.startswith("AA_"),
        }
        for weapon in unit.weapons
    ]


def _air_to_air_inventory(red_units):
    by_type = {}
    attack_capable_unit_ids = []
    for unit in red_units:
        has_available_missile = False
        for weapon in unit.weapons:
            if not weapon.weapon_type.startswith("AA_"):
                continue
            item = by_type.setdefault(
                weapon.weapon_type,
                {"weapon_type": weapon.weapon_type, "total_count": 0, "carriers": []},
            )
            item["total_count"] += int(weapon.count)
            item["carriers"].append(
                {
                    "platform_id": unit.platform_id,
                    "weapon_name": weapon.name,
                    "enabled": weapon.enabled,
                    "count": int(weapon.count),
                }
            )
            has_available_missile |= weapon.enabled and weapon.count > 0
        if has_available_missile:
            attack_capable_unit_ids.append(unit.platform_id)
    inventory = [by_type[key] for key in sorted(by_type)]
    return {
        "total_count": sum(item["total_count"] for item in inventory),
        "attack_capable_unit_ids": attack_capable_unit_ids,
        "by_type": inventory,
    }


def _default_force_status(red_units):
    return {
        "initial_count": len(red_units),
        "alive_count": len(red_units),
        "destroyed_count": 0,
        "units": [
            {
                "platform_id": unit.platform_id,
                "status": "alive",
                "destroyed_at_sim_time": None,
            }
            for unit in red_units
        ],
        "newly_destroyed_ids": (),
    }


def _danger_label(score):
    if score >= 5.0:
        return "high"
    if score >= 3.0:
        return "medium"
    return "low"
