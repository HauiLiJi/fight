"""Deterministic BVR launch, crank, and missile-defense control."""

import json
from dataclasses import dataclass
from pathlib import Path

from .geometry import bearing_deg, clamp, heading_offset_deg


CONFIG_PATH = Path(__file__).with_name("llm_config.json")
SHOT_MEMORY_S = 85.0
NORMAL_HEADING_ERROR_DEG = 28.0
COUNTER_HEADING_ERROR_DEG = 36.0


@dataclass(frozen=True)
class BvrConfig:
    hot_launch_range_m: float = 150000.0
    flank_launch_range_m: float = 115000.0
    cold_launch_range_m: float = 85000.0
    crank_angle_deg: float = 47.0
    crank_duration_s: float = 58.0
    threat_timeout_s: float = 110.0
    defense_duration_s: float = 72.0


@dataclass
class FlightLeg:
    until_sim_time: float
    heading_deg: float
    altitude_m: float
    mach: float


def get_bvr_config():
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = {}
    return BvrConfig(
        hot_launch_range_m=_number(config, "bvr_hot_launch_range_m", 150000.0, 20000.0),
        flank_launch_range_m=_number(config, "bvr_flank_launch_range_m", 115000.0, 20000.0),
        cold_launch_range_m=_number(config, "bvr_cold_launch_range_m", 85000.0, 20000.0),
        crank_angle_deg=_number(config, "bvr_crank_angle_deg", 47.0, 5.0, 85.0),
        crank_duration_s=_number(config, "bvr_crank_duration_sim_s", 58.0, 1.0),
        threat_timeout_s=_number(config, "missile_threat_timeout_sim_s", 110.0, 1.0),
        defense_duration_s=_number(config, "missile_defense_duration_sim_s", 72.0, 1.0),
    )


def is_missile_track(track):
    model = str(getattr(track, "model", "")).upper()
    target_id = str(getattr(track, "target_id", "")).upper()
    return "MISSILE" in model or "_AAM_" in target_id


def fighter_tracks(observation):
    return sorted(
        (
            track
            for track in observation.tracks
            if track.target_side != observation.side and not is_missile_track(track)
        ),
        key=lambda track: track.target_id,
    )


def heading_difference_deg(first, second):
    return abs(((first - second + 180.0) % 360.0) - 180.0)


def target_aspect_deg(unit, target):
    attitude = getattr(target, "attitude", None)
    if attitude is None:
        return 0.0
    return heading_difference_deg(
        bearing_deg(target.position, unit.position),
        attitude.heading_deg,
    )


def launch_range_m(unit, target, config=None):
    config = config or get_bvr_config()
    aspect = target_aspect_deg(unit, target)
    if aspect <= 60.0:
        return config.hot_launch_range_m
    if aspect <= 120.0:
        return config.flank_launch_range_m
    return config.cold_launch_range_m


def in_launch_envelope(unit, target, config=None, heading_error_deg=None):
    from .geometry import distance_m

    config = config or get_bvr_config()
    if distance_m(unit.position, target.position) > launch_range_m(unit, target, config):
        return False
    if heading_error_deg is None:
        return True
    target_bearing = bearing_deg(unit.position, target.position)
    return heading_difference_deg(unit.attitude.heading_deg, target_bearing) <= heading_error_deg


class BvrController:
    def __init__(self):
        self._launch_legs = {}
        self._defense_legs = {}
        self._shot_until = {}
        self._last_metadata = self.empty_metadata()

    def reset(self):
        self._launch_legs.clear()
        self._defense_legs.clear()
        self._shot_until.clear()
        self._last_metadata = self.empty_metadata()

    def can_fire(self, sim_time, unit, target, config=None, heading_error_deg=NORMAL_HEADING_ERROR_DEG):
        if not _weapon_available(unit, "aam_medium"):
            return False
        if self._shot_until.get((unit.platform_id, target.target_id), -1.0) > sim_time:
            return False
        return in_launch_envelope(unit, target, config, heading_error_deg)

    def apply(self, observation, actions, last_launch_time, missile_threat_state):
        config = get_bvr_config()
        now = observation.sim_time
        units = sorted(observation.own_units, key=lambda item: item.platform_id)
        unit_by_id = {unit.platform_id: unit for unit in units}
        fighter_by_id = {track.target_id: track for track in fighter_tracks(observation)}
        index_by_id = {unit.platform_id: index for index, unit in enumerate(units)}
        self._prune(now)

        weapons = {}
        flights = {}
        fired = []
        for action in actions:
            if action.get("type") == "fire":
                unit = unit_by_id.get(action.get("platform_id"))
                target = fighter_by_id.get(action.get("target_id"))
                if unit is None or target is None:
                    continue
                if not self.can_fire(now, unit, target, config):
                    continue
                weapons[unit.platform_id] = action
                self._record_launch(unit, target, index_by_id[unit.platform_id], now, config)
                last_launch_time[unit.platform_id] = now
                fired.append({"platform_id": unit.platform_id, "target_id": target.target_id, "reason": "bvr_launch"})
            elif action.get("type") == "set_flight" and action.get("platform_id") in unit_by_id:
                flights[action["platform_id"]] = action

        active_threats = list((missile_threat_state or {}).get("active_threats", []))
        threats_by_target = {}
        for threat in active_threats:
            threats_by_target.setdefault(threat.get("target_id"), threat)

        modes = {}
        for unit in units:
            threat = threats_by_target.get(unit.platform_id)
            if threat is not None:
                shooter = fighter_by_id.get(threat.get("shooter_id"))
                if shooter is not None and unit.platform_id not in weapons:
                    if self.can_fire(now, unit, shooter, config, COUNTER_HEADING_ERROR_DEG):
                        weapons[unit.platform_id] = _fire_action(unit.platform_id, shooter.target_id)
                        self._record_launch(unit, shooter, index_by_id[unit.platform_id], now, config)
                        last_launch_time[unit.platform_id] = now
                        fired.append({"platform_id": unit.platform_id, "target_id": shooter.target_id, "reason": "missile_counterfire"})
                flights[unit.platform_id] = self._defense_action(
                    unit,
                    shooter,
                    index_by_id[unit.platform_id],
                    threat,
                    now,
                    config,
                )
                modes[unit.platform_id] = "missile_defense"
                continue

            leg = self._launch_legs.get(unit.platform_id)
            if leg is not None and now < leg.until_sim_time:
                flights[unit.platform_id] = _flight_action(
                    unit.platform_id,
                    leg.heading_deg,
                    leg.altitude_m,
                    leg.mach,
                )
                modes[unit.platform_id] = "bvr_crank"
            else:
                modes[unit.platform_id] = "tactic"

        normalized = []
        for unit in units:
            if unit.platform_id in weapons:
                normalized.append(weapons[unit.platform_id])
            if unit.platform_id in flights:
                normalized.append(flights[unit.platform_id])
        self._last_metadata = {
            "bvr_mode": modes,
            "active_missile_threats": active_threats,
            "threatened_platform_ids": sorted(threats_by_target),
            "bvr_fired_actions": fired,
            "bvr_launch_envelopes": self._envelopes(units, fighter_by_id, config),
        }
        return normalized

    def metadata(self):
        return self._last_metadata

    @staticmethod
    def empty_metadata():
        return {
            "bvr_mode": {},
            "active_missile_threats": [],
            "threatened_platform_ids": [],
            "bvr_fired_actions": [],
            "bvr_launch_envelopes": [],
        }

    def _record_launch(self, unit, target, index, now, config):
        sign = -1.0 if index % 2 == 0 else 1.0
        direct_heading = bearing_deg(unit.position, target.position)
        self._launch_legs[unit.platform_id] = FlightLeg(
            until_sim_time=now + config.crank_duration_s,
            heading_deg=heading_offset_deg(direct_heading, sign * config.crank_angle_deg),
            altitude_m=clamp(target.position.altitude_m + (850.0 if sign < 0 else -850.0), 5200.0, 10800.0),
            mach=1.18,
        )
        self._shot_until[(unit.platform_id, target.target_id)] = now + SHOT_MEMORY_S

    def _defense_action(self, unit, shooter, index, threat, now, config):
        threat_key = (unit.platform_id, threat.get("shooter_id", ""))
        leg = self._defense_legs.get(threat_key)
        if leg is None or now >= leg.until_sim_time:
            if shooter is None:
                escape_heading = heading_offset_deg(unit.attitude.heading_deg, 180.0)
            else:
                escape_heading = bearing_deg(shooter.position, unit.position)
            escape_heading = heading_offset_deg(
                escape_heading,
                -32.0 if index % 2 == 0 else 32.0,
            )
            leg = FlightLeg(
                until_sim_time=now + config.defense_duration_s,
                heading_deg=escape_heading,
                altitude_m=clamp(unit.position.altitude_m + (700.0 if index % 2 == 0 else -700.0), 5000.0, 11000.0),
                mach=1.4,
            )
            self._defense_legs[threat_key] = leg
        return _flight_action(unit.platform_id, leg.heading_deg, leg.altitude_m, leg.mach)

    def _prune(self, now):
        self._launch_legs = {
            key: leg for key, leg in self._launch_legs.items() if now < leg.until_sim_time
        }
        self._defense_legs = {
            key: leg for key, leg in self._defense_legs.items() if now < leg.until_sim_time
        }
        self._shot_until = {
            key: until for key, until in self._shot_until.items() if now < until
        }

    @staticmethod
    def _envelopes(units, fighter_by_id, config):
        from .geometry import distance_m

        rows = []
        for unit in units:
            for target in fighter_by_id.values():
                rows.append(
                    {
                        "platform_id": unit.platform_id,
                        "target_id": target.target_id,
                        "range_m": round(distance_m(unit.position, target.position), 1),
                        "launch_range_m": round(launch_range_m(unit, target, config), 1),
                        "in_range": in_launch_envelope(unit, target, config),
                    }
                )
        return rows


def _number(config, key, default, minimum, maximum=None):
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        return default
    if value < minimum or (maximum is not None and value > maximum):
        return default
    return value


def _weapon_available(unit, weapon_name):
    return any(
        weapon.name == weapon_name and weapon.enabled and weapon.count > 0
        for weapon in getattr(unit, "weapons", [])
    )


def _fire_action(platform_id, target_id):
    return {
        "type": "fire",
        "platform_id": platform_id,
        "weapon_name": "aam_medium",
        "target_id": target_id,
    }


def _flight_action(platform_id, heading, altitude, mach):
    return {
        "type": "set_flight",
        "platform_id": platform_id,
        "heading_deg": heading % 360.0,
        "altitude_m": clamp(altitude, 1000.0, 20000.0),
        "mach": clamp(mach, 0.2, 2.0),
    }
