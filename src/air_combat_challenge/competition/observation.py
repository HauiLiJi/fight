from typing import Dict, Iterable, Optional

from .models import (
    AttitudeV1,
    EventV1,
    ObservationV1,
    PositionV1,
    SensorStateV1,
    TrackStateV1,
    UnitStateV1,
    VelocityV1,
    WeaponStateV1,
)


class ObservationBuilder:
    def build(
        self,
        sim_data,
        side,
        controlled_platform_ids,
        episode_id,
        step_index,
        events: Optional[Iterable[EventV1]] = None,
        mission=None,
        global_view=False,
    ):
        controlled = set(controlled_platform_ids)
        own_units = []
        own_data = sim_data.get(side, {})
        for platform_id in sorted(controlled):
            platform_data = own_data.get(platform_id)
            if self._is_alive_platform(platform_data):
                own_units.append(self._make_unit(platform_id, side, platform_data))

        restricted_tracks = self._restricted_tracks(own_data, controlled)
        tracks = (
            self._global_tracks(sim_data, side, restricted_tracks)
            if global_view
            else restricted_tracks
        )
        return ObservationV1(
            episode_id=episode_id,
            step_index=step_index,
            sim_time=float(sim_data.get("time", 0.0)),
            side=side,
            controlled_platform_ids=sorted(controlled),
            own_units=own_units,
            tracks=tracks,
            events=list(events or []),
            mission=mission or {},
        )

    @staticmethod
    def _is_alive_platform(platform_data):
        return isinstance(platform_data, dict) and bool(platform_data.get("pos")) and bool(platform_data.get("att"))

    def _make_unit(self, platform_id, side, data):
        sensor_data = data.get("sensor")
        sensor = None
        if isinstance(sensor_data, dict):
            sensor = SensorStateV1(
                enabled=bool(sensor_data.get("on", 0)),
                cue_azimuth=float(sensor_data.get("heading", 0.0)),
            )

        weapons = []
        for name, weapon in sorted((data.get("weapons") or {}).items()):
            weapons.append(
                WeaponStateV1(
                    name=name,
                    weapon_type=str(weapon.get("type", "")),
                    enabled=bool(weapon.get("on", 0)),
                    count=max(0, int(weapon.get("count", 0))),
                    time_since_last_fired_s=float(weapon.get("TimeSinceLastFired", 0.0)),
                )
            )

        return UnitStateV1(
            platform_id=platform_id,
            side=side,
            entity_type=str(data.get("entity_type", "unknown")),
            position=self._position(data.get("pos") or {}),
            velocity=self._velocity(data.get("vel") or {}),
            attitude=self._attitude(data.get("att") or {}),
            sensor=sensor,
            weapons=weapons,
        )

    def _restricted_tracks(self, own_data, controlled):
        tracks_by_target: Dict[str, dict] = {}
        for detector_id in sorted(controlled):
            platform_data = own_data.get(detector_id) or {}
            for target_id, target_data in (platform_data.get("targets") or {}).items():
                item = tracks_by_target.setdefault(
                    target_id,
                    {"data": target_data, "detected_by": []},
                )
                item["detected_by"].append(detector_id)
        return [
            self._make_track(target_id, item["data"], item["detected_by"])
            for target_id, item in sorted(tracks_by_target.items())
        ]

    def _global_tracks(self, sim_data, side, restricted_tracks):
        other_side = "red" if side == "blue" else "blue"
        restricted_by_target = {track.target_id: track for track in restricted_tracks}
        return [
            restricted_by_target.get(platform_id)
            or self._make_track_from_platform(platform_id, other_side, data)
            for platform_id, data in sorted(sim_data.get(other_side, {}).items())
            if self._is_alive_platform(data)
        ]

    def _make_track(self, target_id, data, detected_by):
        target_vel = data.get("target_vel") or {}
        return TrackStateV1(
            target_id=target_id,
            target_side=str(data.get("target_sideName", "")),
            model=str(data.get("target_model", "")),
            position=PositionV1(
                longitude=float(data.get("target_Longitude", 0.0)),
                latitude=float(data.get("target_Latitude", 0.0)),
                altitude_m=float(data.get("target_Altitude", 0.0)),
            ),
            velocity=VelocityV1(
                north_mps=float(target_vel.get("target_vel_north", 0.0)),
                east_mps=float(target_vel.get("target_vel_east", 0.0)),
                up_mps=float(target_vel.get("target_vel_up", 0.0)),
            ),
            attitude=AttitudeV1(
                heading_deg=float(data.get("target_heading", 0.0)),
                pitch_deg=float(data.get("target_pitch", 0.0)),
                roll_deg=float(data.get("target_roll", 0.0)),
            ),
            detected_by=sorted(set(detected_by)),
        )

    def _make_track_from_platform(self, platform_id, side, data):
        return TrackStateV1(
            target_id=platform_id,
            target_side=side,
            model=str(data.get("entity_type", "")),
            position=self._position(data.get("pos") or {}),
            velocity=self._velocity(data.get("vel") or {}),
            attitude=self._attitude(data.get("att") or {}),
            detected_by=[],
        )

    @staticmethod
    def _position(data):
        return PositionV1(
            longitude=float(data.get("Longitude", 0.0)),
            latitude=float(data.get("Latitude", 0.0)),
            altitude_m=float(data.get("Altitude", 0.0)),
        )

    @staticmethod
    def _velocity(data):
        return VelocityV1(
            north_mps=float(data.get("vel_north", 0.0)),
            east_mps=float(data.get("vel_east", 0.0)),
            up_mps=float(data.get("vel_up", 0.0)),
        )

    @staticmethod
    def _attitude(data):
        return AttitudeV1(
            heading_deg=float(data.get("heading", 0.0)),
            pitch_deg=float(data.get("pitch", 0.0)),
            roll_deg=float(data.get("roll", 0.0)),
        )
