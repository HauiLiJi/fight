"""Editable Strategy 3 rule baseline – modified constants for improved Pk and survival. v3: faster cooldown, longer hot range."""

import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6371000.0

# Increased launch ranges for more engagement opportunities
HOT_RANGE_M = 165000.0
FLANK_RANGE_M = 130000.0
COLD_RANGE_M = 100000.0
# More moderate crank angle, longer duration to maintain missile support
CRANK_ANGLE_DEG = 45.0
CRANK_DURATION_S = 65.0
THREAT_TIMEOUT_S = 110.0
DEFENSE_DURATION_S = 80.0
SHOT_MEMORY_S = 120.0
LAUNCH_COOLDOWN_S = 20.0  # reduced from 30.0 for faster re-engagement
# Relaxed heading errors to allow more shots
NORMAL_HEADING_ERROR_DEG = 30.0
COUNTER_HEADING_ERROR_DEG = 38.0
HIGH_RISK_SCORE = 5.0
MEDIUM_RISK_SCORE = 3.0
HIGH_RISK_ESCAPE_OFFSET_DEG = 30.0
PRESS_OFFSET_DEG = 18.0
POST_LAUNCH_OFFSET_DEG = 25.0
DEFENSE_OFFSET_DEG = 35.0
CONTACT_MACH_LOW_RISK = 0.9
CONTACT_MACH_MED_RISK = 0.86
POST_LAUNCH_MACH = 0.98
CRANK_MACH = 1.18
DEFENSE_MACH = 1.45
ALTITUDE_SPLIT_M = 850.0
DEFENSE_ALTITUDE_SPLIT_M = 700.0


@dataclass
class FlightLeg:
    until_sim_time: float
    heading_deg: float
    altitude_m: float
    mach: float


def clamp(value, low, high):
    return max(low, min(high, value))


def distance_m(first, second):
    lat1 = math.radians(first.latitude)
    lon1 = math.radians(first.longitude)
    lat2 = math.radians(second.latitude)
    lon2 = math.radians(second.longitude)
    haversine = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    horizontal = EARTH_RADIUS_M * 2.0 * math.asin(math.sqrt(haversine))
    return math.hypot(horizontal, first.altitude_m - second.altitude_m)


def bearing_deg(first, second):
    lat1 = math.radians(first.latitude)
    lon1 = math.radians(first.longitude)
    lat2 = math.radians(second.latitude)
    lon2 = math.radians(second.longitude)
    delta_lon = lon2 - lon1
    x_value = math.sin(delta_lon) * math.cos(lat2)
    y_value = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(x_value, y_value)) % 360.0


def heading_offset_deg(base_heading, offset_deg):
    return (base_heading + offset_deg) % 360.0


def heading_difference_deg(first, second):
    return abs(((first - second + 180.0) % 360.0) - 180.0)


def is_missile_track(track):
    model = str(track.model).upper()
    target_id = str(track.target_id).upper()
    return "MISSILE" in model or "_AAM_" in target_id


def fighter_tracks(observation):
    return sorted(
        [
            track
            for track in observation.tracks
            if track.target_side != observation.side and not is_missile_track(track)
        ],
        key=lambda track: track.target_id,
    )


def select_nearest_track(unit, tracks):
    if not tracks:
        return None, float("inf")
    return min(
        [(track, distance_m(unit.position, track.position)) for track in tracks],
        key=lambda item: item[1],
    )


def support_available(units, index, max_support_distance_m=25000.0):
    if len(units) < 2:
        return False
    return (
        distance_m(units[index].position, units[1 - index].position)
        <= max_support_distance_m
    )


def unit_risk_score(unit, tracks, ally_units, unit_index):
    nearest_track, nearest_distance = select_nearest_track(unit, tracks)
    enemy_count = sum(
        distance_m(unit.position, track.position) <= 35000.0 for track in tracks
    )
    isolated = not support_available(ally_units, unit_index)
    score = 0.0
    if nearest_track is not None:
        if nearest_distance < 18000.0:
            score += 3.0
        elif nearest_distance < 30000.0:
            score += 2.0
        elif nearest_distance < 50000.0:
            score += 1.0
    score += max(0, enemy_count - 1) * 1.5
    if isolated:
        score += 2.0
    score += max(0.0, (6000.0 - unit.position.altitude_m) / 2500.0)
    return score


def target_aspect_deg(unit, target):
    return heading_difference_deg(
        bearing_deg(target.position, unit.position),
        target.attitude.heading_deg,
    )


def launch_range_m(unit, target):
    aspect = target_aspect_deg(unit, target)
    if aspect <= 60.0:
        return HOT_RANGE_M
    if aspect <= 120.0:
        return FLANK_RANGE_M
    return COLD_RANGE_M


def in_launch_envelope(unit, target, heading_error_deg):
    if distance_m(unit.position, target.position) > launch_range_m(unit, target):
        return False
    target_bearing = bearing_deg(unit.position, target.position)
    return (
        heading_difference_deg(unit.attitude.heading_deg, target_bearing)
        <= heading_error_deg
    )


def weapon_available(unit, weapon_name):
    return any(
        weapon.name == weapon_name and weapon.enabled and weapon.count > 0
        for weapon in unit.weapons
    )


def flight_action(platform_id, heading, altitude, mach):
    return {
        "type": "set_flight",
        "platform_id": platform_id,
        "heading_deg": heading % 360.0,
        "altitude_m": clamp(altitude, 1000.0, 20000.0),
        "mach": clamp(mach, 0.2, 2.0),
    }


def fire_action(platform_id, target_id):
    return {
        "type": "fire",
        "platform_id": platform_id,
        "weapon_name": "aam_medium",
        "target_id": target_id,
    }


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


class MissileThreatTracker:
    def __init__(self):
        self._seen_event_ids = set()
        self._threats = {}

    def reset(self):
        self._seen_event_ids.clear()
        self._threats.clear()

    def update(self, observation):
        own_ids = {unit.platform_id for unit in observation.own_units}
        for event in observation.events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)
            if event.event_type == "WeaponFired":
                if (
                    event.target in own_ids
                    and event.shooter
                    and event.shooter not in own_ids
                ):
                    started = float(event.sim_time)
                    self._threats[(event.shooter, event.target)] = {
                        "shooter_id": event.shooter,
                        "target_id": event.target,
                        "expires_sim_time": started + THREAT_TIMEOUT_S,
                    }
            elif event.event_type in {"WeaponHit", "WeaponMissed"}:
                self._threats.pop((event.shooter, event.target), None)
        self._threats = {
            key: threat
            for key, threat in self._threats.items()
            if threat["target_id"] in own_ids
            and observation.sim_time < threat["expires_sim_time"]
        }
        return {
            "active_threats": [
                {
                    "shooter_id": threat["shooter_id"],
                    "target_id": threat["target_id"],
                }
                for threat in sorted(
                    self._threats.values(),
                    key=lambda item: (item["target_id"], item["shooter_id"]),
                )
            ]
        }


class BvrController:
    def __init__(self):
        self._launch_legs = {}
        self._defense_legs = {}
        self._shot_until = {}

    def reset(self):
        self._launch_legs.clear()
        self._defense_legs.clear()
        self._shot_until.clear()

    def can_fire(self, sim_time, unit, target, heading_error_deg):
        if not weapon_available(unit, "aam_medium"):
            return False
        if self._shot_until.get((unit.platform_id, target.target_id), -1.0) > sim_time:
            return False
        # New: enforce visibility: shooter must be in target.detected_by
        if unit.platform_id not in target.detected_by:
            return False
        return in_launch_envelope(unit, target, heading_error_deg)

    def apply(self, observation, actions, last_launch_time, threat_state):
        now = observation.sim_time
        units = sorted(observation.own_units, key=lambda item: item.platform_id)
        unit_by_id = {unit.platform_id: unit for unit in units}
        fighter_by_id = {
            track.target_id: track for track in fighter_tracks(observation)
        }
        index_by_id = {unit.platform_id: index for index, unit in enumerate(units)}
        self._prune(now)

        weapons = {}
        flights = {}
        for action in actions:
            if action["type"] == "fire":
                unit = unit_by_id.get(action["platform_id"])
                target = fighter_by_id.get(action["target_id"])
                if unit is None or target is None:
                    continue
                if not self.can_fire(now, unit, target, NORMAL_HEADING_ERROR_DEG):
                    continue
                weapons[unit.platform_id] = action
                self._record_launch(
                    unit,
                    target,
                    index_by_id[unit.platform_id],
                    now,
                )
                last_launch_time[unit.platform_id] = now
            elif action["type"] == "set_flight" and action["platform_id"] in unit_by_id:
                flights[action["platform_id"]] = action

        threats_by_target = {}
        for threat in threat_state.get("active_threats", []):
            threats_by_target.setdefault(threat["target_id"], threat)

        for unit in units:
            threat = threats_by_target.get(unit.platform_id)
            if threat is not None:
                shooter = fighter_by_id.get(threat["shooter_id"])
                if shooter is not None and unit.platform_id not in weapons:
                    if self.can_fire(
                        now,
                        unit,
                        shooter,
                        COUNTER_HEADING_ERROR_DEG,
                    ):
                        weapons[unit.platform_id] = fire_action(
                            unit.platform_id,
                            shooter.target_id,
                        )
                        self._record_launch(
                            unit,
                            shooter,
                            index_by_id[unit.platform_id],
                            now,
                        )
                        last_launch_time[unit.platform_id] = now
                flights[unit.platform_id] = self._defense_action(
                    unit,
                    shooter,
                    index_by_id[unit.platform_id],
                    threat,
                    now,
                )
                continue

            leg = self._launch_legs.get(unit.platform_id)
            if leg is not None and now < leg.until_sim_time:
                flights[unit.platform_id] = flight_action(
                    unit.platform_id,
                    leg.heading_deg,
                    leg.altitude_m,
                    leg.mach,
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
            now + CRANK_DURATION_S,
            heading_offset_deg(direct_heading, sign * CRANK_ANGLE_DEG),
            clamp(
                target.position.altitude_m
                + (ALTITUDE_SPLIT_M if sign < 0 else -ALTITUDE_SPLIT_M),
                4000.0,  # widened from 5200
                12000.0, # widened from 10800
            ),
            CRANK_MACH,
        )
        self._shot_until[(unit.platform_id, target.target_id)] = now + SHOT_MEMORY_S

    def _defense_action(self, unit, shooter, index, threat, now):
        threat_key = (unit.platform_id, threat["shooter_id"])
        leg = self._defense_legs.get(threat_key)
        if leg is None or now >= leg.until_sim_time:
            if shooter is None:
                escape_heading = heading_offset_deg(
                    unit.attitude.heading_deg,
                    180.0,
                )
            else:
                escape_heading = bearing_deg(shooter.position, unit.position)
            escape_heading = heading_offset_deg(
                escape_heading,
                -DEFENSE_OFFSET_DEG if index % 2 == 0 else DEFENSE_OFFSET_DEG,
            )
            leg = FlightLeg(
                now + DEFENSE_DURATION_S,
                escape_heading,
                clamp(
                    unit.position.altitude_m
                    + (
                        DEFENSE_ALTITUDE_SPLIT_M
                        if index % 2 == 0
                        else -DEFENSE_ALTITUDE_SPLIT_M
                    ),
                    4000.0,  # widened from 5000
                    12000.0, # widened from 11000
                ),
                DEFENSE_MACH,
            )
            self._defense_legs[threat_key] = leg
        return flight_action(
            unit.platform_id,
            leg.heading_deg,
            leg.altitude_m,
            leg.mach,
        )

    def _prune(self, now):
        self._launch_legs = {
            key: leg
            for key, leg in self._launch_legs.items()
            if now < leg.until_sim_time
        }
        self._defense_legs = {
            key: leg
            for key, leg in self._defense_legs.items()
            if now < leg.until_sim_time
        }
        self._shot_until = {
            key: until for key, until in self._shot_until.items() if now < until
        }


def fallback_actions(observation, last_launch_time, bvr_controller):
    units = sorted(observation.own_units, key=lambda unit: unit.platform_id)
    tracks = fighter_tracks(observation)
    if not units:
        return []
    if not tracks:
        return [
            flight_action(
                unit.platform_id,
                unit.attitude.heading_deg,
                clamp(unit.position.altitude_m, 7000.0, 9500.0),
                0.72,
            )
            for unit in units
        ]

    assignments = assign_targets(units, tracks)
    actions = []
    for index, unit in enumerate(units):
        target = assignments.get(unit.platform_id)
        if target is None:
            continue
        risk = unit_risk_score(unit, tracks, units, index)
        if risk >= HIGH_RISK_SCORE:
            heading = heading_offset_deg(
                bearing_deg(target.position, unit.position),
                -HIGH_RISK_ESCAPE_OFFSET_DEG
                if index == 0
                else HIGH_RISK_ESCAPE_OFFSET_DEG,
            )
            altitude = clamp(
                unit.position.altitude_m + 300.0,
                6500.0,
                10500.0,
            )
            mach = 0.92
        else:
            direct_heading = bearing_deg(unit.position, target.position)
            heading = heading_offset_deg(
                direct_heading,
                -PRESS_OFFSET_DEG if index == 0 else PRESS_OFFSET_DEG,
            )
            altitude = clamp(
                target.position.altitude_m + 250.0,
                6500.0,
                11000.0,
            )
            mach = (
                CONTACT_MACH_LOW_RISK
                if risk < MEDIUM_RISK_SCORE
                else CONTACT_MACH_MED_RISK
            )
            since_launch = observation.sim_time - last_launch_time.get(
                unit.platform_id,
                -9999.0,
            )
            if (
                since_launch > LAUNCH_COOLDOWN_S
                and bvr_controller.can_fire(
                    observation.sim_time,
                    unit,
                    target,
                    NORMAL_HEADING_ERROR_DEG,
                )
            ):
                actions.append(fire_action(unit.platform_id, target.target_id))
                last_launch_time[unit.platform_id] = observation.sim_time
                heading = heading_offset_deg(
                    direct_heading,
                    -POST_LAUNCH_OFFSET_DEG
                    if index == 0
                    else POST_LAUNCH_OFFSET_DEG,
                )
                mach = POST_LAUNCH_MACH
        actions.append(
            flight_action(unit.platform_id, heading, altitude, mach)
        )
    return actions


class Policy:
    """Strong editable seed: Strategy 3 fallback with tuned constants and visibility check. v3: faster cooldown, longer hot range."""

    def __init__(self):
        self._last_launch_time = {}
        self._threat_tracker = MissileThreatTracker()
        self._bvr_controller = BvrController()

    def reset(self, context):
        self._last_launch_time.clear()
        self._threat_tracker.reset()
        self._bvr_controller.reset()

    def act(self, observation):
        if not observation.own_units:
            return []
        threat_state = self._threat_tracker.update(observation)
        actions = fallback_actions(
            observation,
            self._last_launch_time,
            self._bvr_controller,
        )
        return self._bvr_controller.apply(
            observation,
            actions,
            self._last_launch_time,
            threat_state,
        )
