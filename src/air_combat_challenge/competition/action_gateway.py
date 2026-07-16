from dataclasses import dataclass
from typing import List

from .models import ActionBatchV1, ActionReportV1, ObservationV1


FLIGHT_ACTIONS = {"set_flight", "set_heading", "set_altitude", "set_speed", "fly_path"}
WEAPON_ACTIONS = {"fire", "co_fire"}


@dataclass
class GatewayResult:
    commands: List[dict]
    reports: List[ActionReportV1]

    @property
    def has_violation(self):
        return any(report.status != "accepted" for report in self.reports)


class ActionGateway:
    def validate_and_translate(self, side: str, batch: ActionBatchV1, observation: ObservationV1):
        commands = []
        reports = []
        controlled = set(observation.controlled_platform_ids)
        units = {unit.platform_id: unit for unit in observation.own_units}
        tracks = {track.target_id: track for track in observation.tracks}
        used_flight = set()
        used_weapon = set()

        for index, action in enumerate(batch.actions):
            reason = self._validate_action(
                action,
                controlled,
                units,
                tracks,
                used_flight,
                used_weapon,
            )
            if reason:
                reports.append(self._report(index, action, "rejected", reason))
                continue

            if action.type in FLIGHT_ACTIONS:
                used_flight.add(action.platform_id)
            if action.type in WEAPON_ACTIONS:
                used_weapon.add(action.platform_id)
            commands.append(self._translate(action))
            reports.append(self._report(index, action, "accepted", "accepted"))

        return GatewayResult(commands=commands, reports=reports)

    def _validate_action(self, action, controlled, units, tracks, used_flight, used_weapon):
        if action.platform_id not in controlled or action.platform_id not in units:
            return "unauthorized_platform"
        if action.type in FLIGHT_ACTIONS and action.platform_id in used_flight:
            return "flight_conflict"
        if action.type in WEAPON_ACTIONS and action.platform_id in used_weapon:
            return "weapon_conflict"
        if action.type not in WEAPON_ACTIONS:
            return None

        if action.type == "co_fire" and action.guider_id not in controlled:
            return "guider_unauthorized"
        track = tracks.get(action.target_id)
        detector_id = action.guider_id if action.type == "co_fire" else action.platform_id
        if track is None or detector_id not in track.detected_by:
            return "target_not_visible"
        weapon = next((item for item in units[action.platform_id].weapons if item.name == action.weapon_name), None)
        if weapon is None or not weapon.enabled or weapon.count <= 0:
            return "weapon_unavailable"
        return None

    @staticmethod
    def _report(index, action, status, reason):
        return ActionReportV1(
            action_index=index,
            action_id=action.action_id,
            action_type=action.type,
            platform_id=action.platform_id,
            status=status,
            reason_code=reason,
        )

    @staticmethod
    def _translate(action):
        if action.type == "set_flight":
            sub_cmd_type = "fly_heading_speed_altitude"
            params = {
                "platform_name": action.platform_id,
                "heading": action.heading_deg,
                "altitude": action.altitude_m,
                "mach": action.mach,
            }
        elif action.type == "set_heading":
            sub_cmd_type = "fly_heading"
            params = {"platform_name": action.platform_id, "heading": action.heading_deg}
        elif action.type == "set_altitude":
            sub_cmd_type = "fly_altitude"
            params = {"platform_name": action.platform_id, "altitude": action.altitude_m}
        elif action.type == "set_speed":
            sub_cmd_type = "fly_speed"
            params = {"platform_name": action.platform_id, "speed": action.speed_mps}
        elif action.type == "fly_path":
            sub_cmd_type = "fly_in_path"
            route = ";".join(
                f"{point.latitude},{point.longitude},{point.altitude_m},{point.speed_mps}"
                for point in action.waypoints
            )
            params = {"platform_name": action.platform_id, "route": route, "is_cycle": action.cycle}
        elif action.type == "fire":
            sub_cmd_type = "attack_target"
            params = {
                "platform_name": action.platform_id,
                "weapon_name": action.weapon_name,
                "target_name": action.target_id,
            }
        else:
            sub_cmd_type = "co_attack_target"
            params = {
                "platform_name": action.platform_id,
                "weapon_name": action.weapon_name,
                "target_name": action.target_id,
                "guider_name": action.guider_id,
            }
        return {"cmd_type": "base_cmd", "sub_cmd_type": sub_cmd_type, "params": params}
