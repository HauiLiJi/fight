"""Deterministic launch, post-launch crank, and missile-defense state machine."""

from dataclasses import dataclass

from . import settings
from .geometry import bearing_deg, clamp, distance_m, heading_offset_deg


@dataclass
class FlightLeg:
    until_sim_time: float
    heading_deg: float
    altitude_m: float
    mach: float


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
        bearing_deg(target.position, unit.position), attitude.heading_deg
    )


def launch_range_m(unit, target):
    aspect = target_aspect_deg(unit, target)
    if aspect <= 60.0:
        return settings.HOT_RANGE_M
    if aspect <= 120.0:
        return settings.FLANK_RANGE_M
    return settings.COLD_RANGE_M


def in_launch_envelope(unit, target, heading_error_deg=None):
    if distance_m(unit.position, target.position) > launch_range_m(unit, target):
        return False
    if heading_error_deg is None:
        return True
    target_bearing = bearing_deg(unit.position, target.position)
    return (
        heading_difference_deg(unit.attitude.heading_deg, target_bearing)
        <= heading_error_deg
    )


class BvrController:
    def __init__(self):
        self._launch_legs = {}
        self._defense_legs = {}
        self._shot_until = {}

    def reset(self):
        self._launch_legs.clear()
        self._defense_legs.clear()
        self._shot_until.clear()

    def has_launch_leg(self, platform_id, sim_time):
        leg = self._launch_legs.get(platform_id)
        return leg is not None and sim_time < leg.until_sim_time

    def can_fire(self, sim_time, unit, target, heading_error_deg=None):
        heading_error_deg = (
            settings.NORMAL_HEADING_ERROR_DEG
            if heading_error_deg is None
            else heading_error_deg
        )
        if not _weapon_available(unit, "aam_medium"):
            return False
        if self._shot_until.get((unit.platform_id, target.target_id), -1.0) > sim_time:
            return False
        return in_launch_envelope(unit, target, heading_error_deg)

    def apply(self, observation, actions, last_launch_time, missile_threat_state):
        now = observation.sim_time
        units = sorted(observation.own_units, key=lambda item: item.platform_id)
        unit_by_id = {unit.platform_id: unit for unit in units}
        fighter_by_id = {track.target_id: track for track in fighter_tracks(observation)}
        index_by_id = {unit.platform_id: index for index, unit in enumerate(units)}
        self._prune(now)
        weapons = {}
        flights = {}
        for action in actions:
            if action.get("type") == "fire":
                unit = unit_by_id.get(action.get("platform_id"))
                target = fighter_by_id.get(action.get("target_id"))
                if unit is None or target is None or not self.can_fire(now, unit, target):
                    continue
                weapons[unit.platform_id] = action
                self._record_launch(unit, target, index_by_id[unit.platform_id], now)
                last_launch_time[unit.platform_id] = now
            elif action.get("type") == "set_flight" and action.get("platform_id") in unit_by_id:
                flights[action["platform_id"]] = action

        threats_by_target = {}
        for threat in (missile_threat_state or {}).get("active_threats", []):
            threats_by_target.setdefault(threat.get("target_id"), threat)
        for unit in units:
            threat = threats_by_target.get(unit.platform_id)
            if threat is not None:
                shooter = fighter_by_id.get(threat.get("shooter_id"))
                if shooter is not None and unit.platform_id not in weapons:
                    if self.can_fire(
                        now, unit, shooter, settings.COUNTER_HEADING_ERROR_DEG
                    ):
                        weapons[unit.platform_id] = _fire_action(
                            unit.platform_id, shooter.target_id
                        )
                        self._record_launch(unit, shooter, index_by_id[unit.platform_id], now)
                        last_launch_time[unit.platform_id] = now
                flights[unit.platform_id] = self._defense_action(
                    unit, shooter, index_by_id[unit.platform_id], threat, now
                )
                continue
            leg = self._launch_legs.get(unit.platform_id)
            if leg is not None and now < leg.until_sim_time:
                flights[unit.platform_id] = _flight_action(
                    unit.platform_id, leg.heading_deg, leg.altitude_m, leg.mach
                )

        normalized = []
        for unit in units:
            if unit.platform_id in weapons:
                normalized.append(weapons[unit.platform_id])
            if unit.platform_id in flights:
                normalized.append(flights[unit.platform_id])
        return normalized

    def _record_launch(self, unit, target, index, now):
        sign = -1.0 if index % 2 == 0 else 1.0
        direct_heading = bearing_deg(unit.position, target.position)
        self._launch_legs[unit.platform_id] = FlightLeg(
            now + settings.CRANK_DURATION_S,
            heading_offset_deg(direct_heading, sign * settings.CRANK_ANGLE_DEG),
            clamp(
                target.position.altitude_m
                + (settings.ALTITUDE_SPLIT_M if sign < 0 else -settings.ALTITUDE_SPLIT_M),
                5200.0,
                10800.0,
            ),
            settings.CRANK_MACH,
        )
        self._shot_until[(unit.platform_id, target.target_id)] = (
            now + settings.SHOT_MEMORY_S
        )

    def _defense_action(self, unit, shooter, index, threat, now):
        threat_key = (unit.platform_id, threat.get("shooter_id", ""))
        leg = self._defense_legs.get(threat_key)
        if leg is None or now >= leg.until_sim_time:
            escape_heading = (
                heading_offset_deg(unit.attitude.heading_deg, 180.0)
                if shooter is None
                else bearing_deg(shooter.position, unit.position)
            )
            escape_heading = heading_offset_deg(
                escape_heading,
                -settings.DEFENSE_OFFSET_DEG
                if index % 2 == 0
                else settings.DEFENSE_OFFSET_DEG,
            )
            leg = FlightLeg(
                now + settings.DEFENSE_DURATION_S,
                escape_heading,
                clamp(
                    unit.position.altitude_m
                    + (
                        settings.DEFENSE_ALTITUDE_SPLIT_M
                        if index % 2 == 0
                        else -settings.DEFENSE_ALTITUDE_SPLIT_M
                    ),
                    5000.0,
                    11000.0,
                ),
                settings.DEFENSE_MACH,
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
