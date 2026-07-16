"""A deterministic, team-level air-to-air rule agent.

The agent deliberately contains no side-specific platform names: the same source
can be used for blue and red.  It assigns targets across the formation, keeps a
simple altitude separation between aircraft, and controls weapon expenditure
with per-aircraft/per-target launch cooldowns.
"""

import math

from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1


EARTH_RADIUS_M = 6_371_000
MEDIUM_FIRE_RANGE_M = 51000
MEDIUM_MIN_RANGE_M = 18_000
SHORT_FIRE_RANGE_M = 14_000
MEDIUM_LAUNCH_COOLDOWN_S = 24
SHORT_LAUNCH_COOLDOWN_S = 12.0
TARGET_MEMORY_S = 20.0


class LJHA2ATeamAgent(BaseAgent):
    """Side-neutral A2A team agent for symmetric red-versus-blue simulations."""

    def __init__(self):
        self._last_launch_time = {}
        self._assigned_targets = {}
        self._last_seen_time = {}

    def reset(self, context):
        super().reset(context)
        self._last_launch_time.clear()
        self._assigned_targets.clear()
        self._last_seen_time.clear()

    def act(self, observation):
        ownships = sorted(
            (
                unit
                for unit in observation.own_units
                if unit.platform_id in observation.controlled_platform_ids
            ),
            key=lambda unit: unit.platform_id,
        )
        enemies = sorted(
            (
                track
                for track in observation.tracks
                if track.target_side != observation.side
            ),
            key=lambda track: track.target_id,
        )

        self._remember_tracks(enemies, observation.sim_time)
        self._discard_stale_state(observation.sim_time, {track.target_id for track in enemies})
        if not ownships:
            return ActionBatchV1()

        assignments = self._assign_targets(ownships, enemies)
        actions = []
        for index, ownship in enumerate(ownships):
            target = assignments.get(ownship.platform_id)
            if target is None:
                actions.append(self._patrol_action(ownship, index))
                continue

            distance = _distance_m(ownship.position, target.position)
            heading = _bearing_deg(ownship.position, target.position)
            altitude = self._tactical_altitude(ownship, target, index, distance)
            weapon_name = self._select_weapon(ownship, distance)
            if weapon_name and self._can_fire(
                observation.sim_time, ownship, target, weapon_name, distance
            ):
                actions.append(
                    {
                        "type": "fire",
                        "platform_id": ownship.platform_id,
                        "weapon_name": weapon_name,
                        "target_id": target.target_id,
                    }
                )
                self._last_launch_time[(ownship.platform_id, target.target_id, weapon_name)] = (
                    observation.sim_time
                )

            actions.append(
                self._flight_action(
                    ownship.platform_id,
                    heading,
                    altitude,
                    self._tactical_mach(distance),
                )
            )

        return ActionBatchV1.model_validate({"actions": actions})

    def _assign_targets(self, ownships, enemies):
        """Assign different targets first, then permit support/second attacks."""
        if not enemies:
            self._assigned_targets.clear()
            return {}

        enemy_by_id = {track.target_id: track for track in enemies}
        assignments = {}
        unassigned = set(enemy_by_id)

        # Preserve a valid previous assignment.  This prevents target switching
        # every step when two tracks have similar ranges.
        for ownship in ownships:
            target_id = self._assigned_targets.get(ownship.platform_id)
            if target_id in enemy_by_id and target_id in unassigned:
                assignments[ownship.platform_id] = enemy_by_id[target_id]
                unassigned.remove(target_id)

        # Give every remaining aircraft a distinct target where possible.
        for ownship in ownships:
            if ownship.platform_id in assignments:
                continue
            candidates = unassigned or set(enemy_by_id)
            target_id = min(
                candidates,
                key=lambda item: _distance_m(ownship.position, enemy_by_id[item].position),
            )
            assignments[ownship.platform_id] = enemy_by_id[target_id]
            unassigned.discard(target_id)

        self._assigned_targets = {
            platform_id: target.target_id for platform_id, target in assignments.items()
        }
        return assignments

    def _can_fire(self, sim_time, ownship, target, weapon_name, distance):
        # A fire action is valid only when this aircraft, rather than merely a
        # teammate, has detected the target.
        if ownship.platform_id not in target.detected_by:
            return False
        if _weapon_remaining(ownship, weapon_name) <= 0:
            return False

        if weapon_name == "aam_medium":
            if not MEDIUM_MIN_RANGE_M <= distance <= MEDIUM_FIRE_RANGE_M:
                return False
            cooldown = MEDIUM_LAUNCH_COOLDOWN_S
        else:
            if distance > SHORT_FIRE_RANGE_M:
                return False
            cooldown = SHORT_LAUNCH_COOLDOWN_S

        last_time = self._last_launch_time.get(
            (ownship.platform_id, target.target_id, weapon_name), -9_999.0
        )
        return sim_time - last_time >= cooldown

    @staticmethod
    def _select_weapon(ownship, distance):
        if MEDIUM_MIN_RANGE_M <= distance <= MEDIUM_FIRE_RANGE_M:
            return "aam_medium" if _weapon_remaining(ownship, "aam_medium") else None
        if distance <= SHORT_FIRE_RANGE_M:
            return "aam_short" if _weapon_remaining(ownship, "aam_short") else None
        return None

    @staticmethod
    def _tactical_mach(distance):
        if distance > 55_000:
            return 0.95
        if distance > 30_000:
            return 0.90
        return 0.84

    @staticmethod
    def _tactical_altitude(ownship, target, index, distance):
        # The leading aircraft stays above the target; wingmen alternate below
        # and above it to maintain a simple vertical deconfliction.
        offset = 800 if index % 2 == 0 else -900
        if distance > 55_000:
            offset += 500
        return _clamp(target.position.altitude_m + offset, 5_000, 11_000)

    @staticmethod
    def _patrol_action(ownship, index):
        altitude = _clamp(7_500 + (index % 2) * 1_000, 6_000, 10_000)
        return LJHA2ATeamAgent._flight_action(
            ownship.platform_id,
            ownship.attitude.heading_deg,
            altitude,
            0.78,
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

    def _remember_tracks(self, enemies, sim_time):
        for track in enemies:
            self._last_seen_time[track.target_id] = sim_time

    def _discard_stale_state(self, sim_time, visible_target_ids):
        expired = {
            target_id
            for target_id, last_seen in self._last_seen_time.items()
            if target_id not in visible_target_ids and sim_time - last_seen > TARGET_MEMORY_S
        }
        for target_id in expired:
            self._last_seen_time.pop(target_id, None)
        self._assigned_targets = {
            platform_id: target_id
            for platform_id, target_id in self._assigned_targets.items()
            if target_id not in expired
        }


def _clamp(value, low, high):
    return max(low, min(high, value))


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name and weapon.enabled:
            return int(weapon.count)
    return 0


def _distance_m(a, b):
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    horizontal = EARTH_RADIUS_M * 2 * math.asin(math.sqrt(haversine))
    return math.hypot(horizontal, a.altitude_m - b.altitude_m)


def _bearing_deg(a, b):
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360
