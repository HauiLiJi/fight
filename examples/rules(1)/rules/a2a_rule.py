import math

from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1


EARTH_RADIUS_M = 6371000
FIRE_RANGE_M = 50000


class A2ARuleAgent(BaseAgent):
    """Per-aircraft deterministic A2A baseline using the competition API."""

    def __init__(self):
        self._last_launch_time = {}

    def reset(self, context):
        super().reset(context)
        self._last_launch_time.clear()

    def act(self, observation):
        ownship = next(
            (
                unit
                for unit in observation.own_units
                if unit.platform_id in observation.controlled_platform_ids
            ),
            None,
        )
        if ownship is None:
            return ActionBatchV1()

        enemies = [
            track
            for track in observation.tracks
            if track.target_side != observation.side
        ]
        if not enemies:
            return ActionBatchV1.model_validate(
                {
                    "actions": [
                        self._flight_action(
                            ownship.platform_id,
                            ownship.attitude.heading_deg,
                            _clamp(ownship.position.altitude_m, 6000, 9000),
                            0.75,
                        )
                    ]
                }
            )

        target, distance = min(
            ((track, _distance_m(ownship.position, track.position)) for track in enemies),
            key=lambda item: item[1],
        )
        heading = _bearing_deg(ownship.position, target.position)
        altitude = _clamp(target.position.altitude_m + 500, 5000, 11000)

        actions = []
        weapon_name = "aam_medium"
        if self._can_fire(
            observation.sim_time,
            distance,
            ownship,
            weapon_name,
        ):
            actions.append(
                {
                    "type": "fire",
                    "platform_id": ownship.platform_id,
                    "weapon_name": weapon_name,
                    "target_id": target.target_id,
                }
            )
            self._last_launch_time[ownship.platform_id] = observation.sim_time

        actions.append(
            self._flight_action(
                ownship.platform_id,
                heading,
                altitude,
                0.9 if distance > 30000 else 0.85,
            )
        )
        return ActionBatchV1.model_validate({"actions": actions})

    def _can_fire(self, sim_time, distance, ownship, weapon_name):
        if _weapon_remaining(ownship, weapon_name) <= 0 or distance > FIRE_RANGE_M:
            return False
        return (
            sim_time
            - self._last_launch_time.get(ownship.platform_id, -9999)
            > 30
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


def _clamp(value, low, high):
    return max(low, min(high, value))


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0


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
    return (math.degrees(math.atan2(x, y)) + 360) % 360
