import math
from dataclasses import dataclass

from .config import EARTH_RADIUS_M
from .team_memory import OBSERVED


@dataclass(frozen=True)
class PairGeometry:
    """Ownship-to-track geometry.

    altitude_delta_m is target altitude minus ownship altitude. closing_speed_mps
    is positive when range is decreasing. bearing_deg is ownship-to-target.
    alignment is enemy heading toward ownship; own_alignment is ownship heading
    toward enemy.
    """

    distance_3d_m: float
    horizontal_distance_m: float
    altitude_delta_m: float
    relative_speed_mps: float
    closing_speed_mps: float
    bearing_deg: float
    alignment: float
    own_alignment: float
    is_observed: bool


@dataclass(frozen=True)
class FormationGeometry:
    """Two-aircraft formation geometry.

    altitude_delta_m is second aircraft altitude minus first aircraft altitude.
    heading_delta_deg is the smallest absolute heading difference.
    """

    spacing_m: float
    heading_delta_deg: float
    altitude_delta_m: float
    centroid_position: object
    centroid_velocity: object


@dataclass(frozen=True)
class TrackSituation:
    """Per-ownship, per-track situation without intent or probability fields."""

    ownship_id: str
    target_id: str
    track_status: str
    is_observed: bool
    pair: PairGeometry
    track_age_s: float
    time_since_last_seen_s: float
    source_count: int
    prediction_residual_m: float


@dataclass(frozen=True)
class TrendMetrics:
    """Recent 2-3 second deltas.

    Positive distance and spacing deltas mean increasing values. Heading deltas
    are current smallest heading difference minus past smallest heading
    difference.
    """

    distance_delta_m: float
    own_spacing_delta_m: float
    enemy_spacing_delta_m: float
    own_heading_delta_deg: float
    enemy_heading_delta_deg: float


@dataclass(frozen=True)
class SituationFrame:
    """Team-level situation frame.

    centroid_closing_speed_mps is positive when formation centroid range is
    decreasing. enemy_depth_delta_m is enemy-2 projection minus enemy-1
    projection on the enemy-centroid-to-own-centroid horizontal axis.
    """

    sim_time: float
    tracks: tuple
    own_formation: FormationGeometry
    enemy_formation: FormationGeometry
    own_centroid: object
    enemy_centroid: object
    centroid_distance_m: float
    centroid_closing_speed_mps: float
    enemy_depth_delta_m: float
    enemy_turn_rates_dps: dict
    enemy_turn_rate_delta_dps: float
    trends: TrendMetrics


class SituationAnalyzer:
    def compute(self, observation, memory_snapshot):
        controlled_ids = set(observation.controlled_platform_ids)
        ownships = [
            unit
            for unit in observation.own_units
            if unit.platform_id in controlled_ids
        ]
        tracks = tuple(
            self._track_situation(ownship, track)
            for ownship in ownships
            for track in memory_snapshot.tracks.values()
            if track.target_side != observation.side and track.status != "LOST"
        )

        own_formation = _formation_geometry(ownships)
        enemy_tracks = [
            track
            for track in memory_snapshot.tracks.values()
            if track.target_side != observation.side and track.status != "LOST"
        ]
        enemy_formation = _formation_geometry(enemy_tracks)

        own_centroid = _centroid_position(ownships) if ownships else None
        enemy_centroid = _centroid_position(enemy_tracks) if enemy_tracks else None
        own_velocity = _centroid_velocity(ownships) if ownships else None
        enemy_velocity = _centroid_velocity(enemy_tracks) if enemy_tracks else None

        centroid_distance_m = None
        centroid_closing_speed_mps = None
        enemy_depth_delta_m = None
        if own_centroid is not None and enemy_centroid is not None:
            centroid_distance_m = _distance_3d_m(own_centroid, enemy_centroid)
            centroid_closing_speed_mps = _closing_speed_mps(
                own_centroid,
                own_velocity,
                enemy_centroid,
                enemy_velocity,
            )
            enemy_depth_delta_m = _enemy_depth_delta_m(enemy_tracks, enemy_centroid, own_centroid)

        enemy_turn_rates_dps = {
            track.target_id: _turn_rate_dps(
                memory_snapshot.tracks[track.target_id],
                memory_snapshot,
            )
            for track in enemy_tracks
        }
        enemy_turn_rate_delta_dps = _delta_if_two(
            list(enemy_turn_rates_dps.values())
        )

        return SituationFrame(
            sim_time=observation.sim_time,
            tracks=tracks,
            own_formation=own_formation,
            enemy_formation=enemy_formation,
            own_centroid=own_centroid,
            enemy_centroid=enemy_centroid,
            centroid_distance_m=centroid_distance_m,
            centroid_closing_speed_mps=centroid_closing_speed_mps,
            enemy_depth_delta_m=enemy_depth_delta_m,
            enemy_turn_rates_dps=enemy_turn_rates_dps,
            enemy_turn_rate_delta_dps=enemy_turn_rate_delta_dps,
            trends=_trend_metrics(memory_snapshot),
        )

    def _track_situation(self, ownship, track):
        pair = PairGeometry(
            distance_3d_m=_distance_3d_m(ownship.position, track.position),
            horizontal_distance_m=_horizontal_distance_m(
                ownship.position,
                track.position,
            ),
            altitude_delta_m=track.position.altitude_m - ownship.position.altitude_m,
            relative_speed_mps=_relative_speed_mps(ownship.velocity, track.velocity),
            closing_speed_mps=_closing_speed_mps(
                ownship.position,
                ownship.velocity,
                track.position,
                track.velocity,
            ),
            bearing_deg=_bearing_deg(ownship.position, track.position),
            alignment=_heading_alignment(
                track.attitude.heading_deg,
                _bearing_deg(track.position, ownship.position),
            ),
            own_alignment=_heading_alignment(
                ownship.attitude.heading_deg,
                _bearing_deg(ownship.position, track.position),
            ),
            is_observed=track.status == OBSERVED,
        )
        return TrackSituation(
            ownship_id=ownship.platform_id,
            target_id=track.target_id,
            track_status=track.status,
            is_observed=track.status == OBSERVED,
            pair=pair,
            track_age_s=track.track_age_s,
            time_since_last_seen_s=track.time_since_last_seen_s,
            source_count=track.source_count,
            prediction_residual_m=track.prediction_residual_m,
        )


def _formation_geometry(items):
    if len(items) < 2:
        return None
    first, second = items[0], items[1]
    return FormationGeometry(
        spacing_m=_distance_3d_m(first.position, second.position),
        heading_delta_deg=_angle_delta_deg(
            first.attitude.heading_deg,
            second.attitude.heading_deg,
        ),
        altitude_delta_m=second.position.altitude_m - first.position.altitude_m,
        centroid_position=_centroid_position(items),
        centroid_velocity=_centroid_velocity(items),
    )


def _centroid_position(items):
    count = len(items)
    latitude = sum(item.position.latitude for item in items) / count
    longitude = sum(item.position.longitude for item in items) / count
    altitude_m = sum(item.position.altitude_m for item in items) / count
    return _PositionLike(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


def _centroid_velocity(items):
    count = len(items)
    return _VelocityLike(
        north_mps=sum(item.velocity.north_mps for item in items) / count,
        east_mps=sum(item.velocity.east_mps for item in items) / count,
        up_mps=sum(item.velocity.up_mps for item in items) / count,
    )


def _enemy_depth_delta_m(enemy_tracks, enemy_centroid, own_centroid):
    if len(enemy_tracks) < 2:
        return None
    axis_north, axis_east = _unit_vector_to(enemy_centroid, own_centroid)
    if axis_north is None or axis_east is None:
        return None
    projections = [
        _north_east_delta_m(enemy_centroid, track.position)[0] * axis_north
        + _north_east_delta_m(enemy_centroid, track.position)[1] * axis_east
        for track in enemy_tracks[:2]
    ]
    return projections[1] - projections[0]


def _turn_rate_dps(track, memory_snapshot):
    history = memory_snapshot.track_history.get(track.target_id, ())
    if len(history) < 2:
        return None
    latest = history[-1]
    previous = _history_entry_at_age(history, memory_snapshot.sim_time, 2.0, 3.0)
    if previous is None:
        previous = history[-2]
    dt = latest.track_age_s - previous.track_age_s
    if dt <= 0:
        return None
    return _signed_angle_delta_deg(
        previous.attitude.heading_deg,
        latest.attitude.heading_deg,
    ) / dt


def _trend_metrics(memory_snapshot):
    return TrendMetrics(
        distance_delta_m=_distance_delta_m(memory_snapshot),
        own_spacing_delta_m=_own_spacing_delta_m(memory_snapshot),
        enemy_spacing_delta_m=_enemy_spacing_delta_m(memory_snapshot),
        own_heading_delta_deg=_own_heading_delta_deg(memory_snapshot),
        enemy_heading_delta_deg=_enemy_heading_delta_deg(memory_snapshot),
    )


def _distance_delta_m(memory_snapshot):
    own_histories = list(memory_snapshot.ownship_history.values())
    if not own_histories:
        return None
    observed_tracks = [
        track
        for track in memory_snapshot.tracks.values()
        if track.status == OBSERVED
    ]
    if not observed_tracks:
        return None
    current_own = own_histories[0][-1]
    current_track = min(
        observed_tracks,
        key=lambda track: _distance_3d_m(current_own.position, track.position),
    )
    current = _distance_3d_m(current_own.position, current_track.position)
    past_own = _history_entry_at_age(
        own_histories[0],
        memory_snapshot.sim_time,
        2.0,
        3.0,
    )
    past_track = _history_entry_at_age(
        memory_snapshot.track_history.get(current_track.target_id, ()),
        memory_snapshot.sim_time,
        2.0,
        3.0,
    )
    if past_own is None or past_track is None:
        return None
    return current - _distance_3d_m(past_own.position, past_track.position)


def _own_spacing_delta_m(memory_snapshot):
    histories = list(memory_snapshot.ownship_history.values())
    if len(histories) < 2 or not histories[0] or not histories[1]:
        return None
    current = _distance_3d_m(histories[0][-1].position, histories[1][-1].position)
    previous_pair = _history_pair_at_age(histories[0], histories[1], 2.0, 3.0)
    if previous_pair is None:
        return None
    previous = _distance_3d_m(previous_pair[0].position, previous_pair[1].position)
    return current - previous


def _own_heading_delta_deg(memory_snapshot):
    histories = list(memory_snapshot.ownship_history.values())
    if len(histories) < 2 or not histories[0] or not histories[1]:
        return None
    current = _angle_delta_deg(
        histories[0][-1].attitude.heading_deg,
        histories[1][-1].attitude.heading_deg,
    )
    previous_pair = _history_pair_at_age(histories[0], histories[1], 2.0, 3.0)
    if previous_pair is None:
        return None
    previous = _angle_delta_deg(
        previous_pair[0].attitude.heading_deg,
        previous_pair[1].attitude.heading_deg,
    )
    return current - previous


def _enemy_spacing_delta_m(memory_snapshot):
    histories = list(memory_snapshot.track_history.values())
    active_histories = [
        history
        for history in histories
        if history and history[-1].status != "LOST"
    ]
    if len(active_histories) < 2:
        return None
    current = _distance_3d_m(
        active_histories[0][-1].position,
        active_histories[1][-1].position,
    )
    previous_pair = _history_pair_at_age(
        active_histories[0],
        active_histories[1],
        2.0,
        3.0,
    )
    if previous_pair is None:
        return None
    previous = _distance_3d_m(previous_pair[0].position, previous_pair[1].position)
    return current - previous


def _enemy_heading_delta_deg(memory_snapshot):
    histories = list(memory_snapshot.track_history.values())
    active_histories = [
        history
        for history in histories
        if history and history[-1].status != "LOST"
    ]
    if len(active_histories) < 2:
        return None
    current = _angle_delta_deg(
        active_histories[0][-1].attitude.heading_deg,
        active_histories[1][-1].attitude.heading_deg,
    )
    previous_pair = _history_pair_at_age(
        active_histories[0],
        active_histories[1],
        2.0,
        3.0,
    )
    if previous_pair is None:
        return None
    previous = _angle_delta_deg(
        previous_pair[0].attitude.heading_deg,
        previous_pair[1].attitude.heading_deg,
    )
    return current - previous


def _history_pair_at_age(history_a, history_b, min_age_s, max_age_s):
    current_time = min(_entry_time(history_a[-1]), _entry_time(history_b[-1]))
    candidates_a = [
        entry
        for entry in history_a
        if min_age_s <= current_time - _entry_time(entry) <= max_age_s
    ]
    candidates_b = [
        entry
        for entry in history_b
        if min_age_s <= current_time - _entry_time(entry) <= max_age_s
    ]
    if not candidates_a or not candidates_b:
        return None
    return candidates_a[-1], candidates_b[-1]


def _history_entry_at_age(history, current_time, min_age_s, max_age_s):
    candidates = [
        entry
        for entry in history
        if min_age_s <= current_time - _entry_time(entry) <= max_age_s
    ]
    if not candidates:
        return None
    return candidates[-1]


def _entry_time(entry):
    if hasattr(entry, "sim_time"):
        return entry.sim_time
    return entry.last_prediction_time or entry.last_seen_time


def _horizontal_distance_m(a, b):
    return _horizontal_and_components_m(a, b)[0]


def _distance_3d_m(a, b):
    horizontal = _horizontal_distance_m(a, b)
    vertical = b.altitude_m - a.altitude_m
    return math.hypot(horizontal, vertical)


def _relative_speed_mps(a, b):
    return math.sqrt(
        (b.north_mps - a.north_mps) ** 2
        + (b.east_mps - a.east_mps) ** 2
        + (b.up_mps - a.up_mps) ** 2
    )


def _closing_speed_mps(pos_a, vel_a, pos_b, vel_b):
    north_m, east_m = _north_east_delta_m(pos_a, pos_b)
    up_m = pos_b.altitude_m - pos_a.altitude_m
    distance = math.sqrt(north_m * north_m + east_m * east_m + up_m * up_m)
    if distance == 0:
        return None
    unit_north = north_m / distance
    unit_east = east_m / distance
    unit_up = up_m / distance
    relative_north = vel_b.north_mps - vel_a.north_mps
    relative_east = vel_b.east_mps - vel_a.east_mps
    relative_up = vel_b.up_mps - vel_a.up_mps
    return -(
        relative_north * unit_north
        + relative_east * unit_east
        + relative_up * unit_up
    )


def _bearing_deg(a, b):
    lat1, lon1 = math.radians(a.latitude), math.radians(a.longitude)
    lat2, lon2 = math.radians(b.latitude), math.radians(b.longitude)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _heading_alignment(heading_deg, bearing_deg):
    return math.cos(math.radians(_angle_delta_deg(heading_deg, bearing_deg)))


def _angle_delta_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _signed_angle_delta_deg(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def _delta_if_two(values):
    actual_values = [value for value in values if value is not None]
    if len(actual_values) != 2:
        return None
    return actual_values[1] - actual_values[0]


def _unit_vector_to(a, b):
    north_m, east_m = _north_east_delta_m(a, b)
    length = math.hypot(north_m, east_m)
    if length == 0:
        return None, None
    return north_m / length, east_m / length


def _north_east_delta_m(a, b):
    lat_avg = math.radians((a.latitude + b.latitude) / 2.0)
    north_m = math.radians(b.latitude - a.latitude) * EARTH_RADIUS_M
    east_m = (
        math.radians(b.longitude - a.longitude)
        * EARTH_RADIUS_M
        * math.cos(lat_avg)
    )
    return north_m, east_m


def _horizontal_and_components_m(a, b):
    north_m, east_m = _north_east_delta_m(a, b)
    return math.hypot(north_m, east_m), north_m, east_m


@dataclass(frozen=True)
class _PositionLike:
    latitude: float
    longitude: float
    altitude_m: float


@dataclass(frozen=True)
class _VelocityLike:
    north_mps: float
    east_mps: float
    up_mps: float
