from air_combat_challenge.competition.models import ActionBatchV1

from .config import CRUISE_MACH, NO_TARGET_MAX_ALTITUDE_M, NO_TARGET_MIN_ALTITUDE_M


class FallbackPolicy:
    def build_safe_hold(self, observation) -> ActionBatchV1:
        actions = [
            _flight_action(
                unit.platform_id,
                unit.attitude.heading_deg,
                _clamp(unit.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                CRUISE_MACH,
            )
            for unit in _live_ownships(observation)
        ]
        return ActionBatchV1.model_validate({"actions": actions})

    def build_no_target(self, observation) -> ActionBatchV1:
        return self.build_safe_hold(observation)

    def build_invalid_plan(self, observation, memory_snapshot, situation, reason) -> ActionBatchV1:
        del memory_snapshot, situation, reason
        return self.build_safe_hold(observation)

    def build_emergency_evasion(
        self,
        observation,
        situation,
        threatened_platform_id,
        threat_target_id,
    ) -> ActionBatchV1:
        current_track_ids = {
            track.target_id
            for track in observation.tracks
            if track.target_side != observation.side
        }
        if threat_target_id not in current_track_ids:
            return self.build_safe_hold(observation)
        threat = None
        for track in situation.tracks:
            if (
                track.ownship_id == threatened_platform_id
                and track.target_id == threat_target_id
                and track.is_observed
            ):
                threat = track
                break
        if threat is None:
            return self.build_safe_hold(observation)
        actions = []
        for unit in _live_ownships(observation):
            if unit.platform_id != threatened_platform_id:
                actions.append(
                    _flight_action(
                        unit.platform_id,
                        unit.attitude.heading_deg,
                        _clamp(unit.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                        CRUISE_MACH,
                    )
                )
                continue
            heading = (threat.pair.bearing_deg + 180.0) % 360.0
            actions.append(
                _flight_action(
                    unit.platform_id,
                    heading,
                    _clamp(unit.position.altitude_m, NO_TARGET_MIN_ALTITUDE_M, NO_TARGET_MAX_ALTITUDE_M),
                    CRUISE_MACH,
                )
            )
        return ActionBatchV1.model_validate({"actions": actions})


def _live_ownships(observation):
    controlled_ids = set(observation.controlled_platform_ids)
    return [
        unit
        for unit in observation.own_units
        if unit.platform_id in controlled_ids
    ]


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
