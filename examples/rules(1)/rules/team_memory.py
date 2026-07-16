import math
from collections import deque
from dataclasses import dataclass

from .config import EARTH_RADIUS_M, HISTORY_WINDOW_S, TRACK_COASTING_TIMEOUT_S


OBSERVED = "OBSERVED"
COASTING = "COASTING"
LOST = "LOST"


@dataclass(frozen=True)
class OwnshipHistoryEntry:
    platform_id: str
    sim_time: float
    position: object
    velocity: object
    attitude: object
    weapons: tuple
    sensor: object


@dataclass(frozen=True)
class TrackMemoryEntry:
    target_id: str
    sim_time: float
    target_side: str
    model: str
    position: object
    velocity: object
    attitude: object
    detected_by: tuple
    first_seen_time: float
    last_seen_time: float
    consecutive_visible_steps: int
    status: str
    track_age_s: float
    time_since_last_seen_s: float
    source_count: int
    prediction_residual_m: float = None
    predicted_position: object = None
    last_prediction_time: float = None


@dataclass(frozen=True)
class MemorySnapshot:
    sim_time: float
    ownship_history: dict
    track_history: dict
    tracks: dict
    events_history: tuple
    visible_target_ids: frozenset
    last_known_tracks: dict


class TeamMemory:
    def __init__(self):
        self._ownship_history = {}
        self._track_history = {}
        self._tracks = {}
        self._events_history = deque()
        self._last_known_tracks = {}
        self._plan_context = {}

    def reset(self):
        self._ownship_history.clear()
        self._track_history.clear()
        self._tracks.clear()
        self._events_history.clear()
        self._last_known_tracks.clear()
        self._plan_context.clear()

    def update(self, observation):
        sim_time = float(observation.sim_time)
        self._append_ownships(observation, sim_time)
        self._append_events(observation, sim_time)

        visible_target_ids = {
            track.target_id
            for track in observation.tracks
            if track.target_side != observation.side
        }
        observed_tracks = {
            track.target_id: track
            for track in observation.tracks
            if track.target_side != observation.side
        }

        for target_id, track in observed_tracks.items():
            self._update_observed_track(target_id, track, sim_time)

        for target_id in list(self._tracks):
            if target_id not in observed_tracks:
                self._update_unobserved_track(target_id, sim_time)

        self._prune_histories(sim_time)
        return self._snapshot(sim_time, visible_target_ids)

    def get_track(self, target_id):
        return self._tracks.get(target_id)

    def set_plan_context(self, **kwargs):
        self._plan_context.update(kwargs)

    def get_plan_context(self):
        return dict(self._plan_context)

    def _append_ownships(self, observation, sim_time):
        controlled_ids = set(observation.controlled_platform_ids)
        for unit in observation.own_units:
            if unit.platform_id not in controlled_ids:
                continue
            history = self._ownship_history.setdefault(unit.platform_id, deque())
            history.append(
                OwnshipHistoryEntry(
                    platform_id=unit.platform_id,
                    sim_time=sim_time,
                    position=unit.position,
                    velocity=unit.velocity,
                    attitude=unit.attitude,
                    weapons=tuple(unit.weapons),
                    sensor=unit.sensor,
                )
            )

    def _append_events(self, observation, sim_time):
        for event in observation.events:
            self._events_history.append(event)
        self._prune_deque(self._events_history, sim_time, lambda event: event.sim_time)

    def _update_observed_track(self, target_id, track, sim_time):
        previous = self._tracks.get(target_id)
        first_seen_time = previous.first_seen_time if previous else sim_time
        consecutive_visible_steps = (
            previous.consecutive_visible_steps + 1
            if previous and previous.status == OBSERVED
            else 1
        )
        prediction_residual_m = None
        if previous and previous.predicted_position is not None:
            prediction_residual_m = _distance_m(
                previous.predicted_position,
                track.position,
            )

        entry = TrackMemoryEntry(
            target_id=track.target_id,
            sim_time=sim_time,
            target_side=track.target_side,
            model=track.model,
            position=track.position,
            velocity=track.velocity,
            attitude=track.attitude,
            detected_by=tuple(track.detected_by),
            first_seen_time=first_seen_time,
            last_seen_time=sim_time,
            consecutive_visible_steps=consecutive_visible_steps,
            status=OBSERVED,
            track_age_s=sim_time - first_seen_time,
            time_since_last_seen_s=0.0,
            source_count=len(track.detected_by),
            prediction_residual_m=prediction_residual_m,
        )
        self._tracks[target_id] = entry
        self._last_known_tracks[target_id] = entry
        self._track_history.setdefault(target_id, deque()).append(entry)

    def _update_unobserved_track(self, target_id, sim_time):
        previous = self._tracks[target_id]
        time_since_last_seen_s = sim_time - previous.last_seen_time
        status = (
            COASTING
            if time_since_last_seen_s <= TRACK_COASTING_TIMEOUT_S
            else LOST
        )
        predicted_position = None
        position = previous.position
        if status == COASTING:
            dt = max(0.0, sim_time - self._last_history_time(target_id, previous))
            predicted_position = _predict_position(
                previous.position,
                previous.velocity,
                dt,
            )
            position = predicted_position

        entry = TrackMemoryEntry(
            target_id=previous.target_id,
            sim_time=sim_time,
            target_side=previous.target_side,
            model=previous.model,
            position=position,
            velocity=previous.velocity,
            attitude=previous.attitude,
            detected_by=previous.detected_by,
            first_seen_time=previous.first_seen_time,
            last_seen_time=previous.last_seen_time,
            consecutive_visible_steps=0,
            status=status,
            track_age_s=sim_time - previous.first_seen_time,
            time_since_last_seen_s=time_since_last_seen_s,
            source_count=len(previous.detected_by),
            prediction_residual_m=previous.prediction_residual_m,
            predicted_position=(
                predicted_position
                if predicted_position is not None
                else previous.predicted_position
            ),
            last_prediction_time=(
                sim_time
                if predicted_position is not None
                else previous.last_prediction_time
            ),
        )
        self._tracks[target_id] = entry
        self._last_known_tracks[target_id] = entry
        self._track_history.setdefault(target_id, deque()).append(entry)

    def _last_history_time(self, target_id, fallback):
        history = self._track_history.get(target_id)
        if history:
            return history[-1].last_prediction_time or history[-1].last_seen_time
        return fallback.last_prediction_time or fallback.last_seen_time

    def _prune_histories(self, sim_time):
        for history in self._ownship_history.values():
            self._prune_deque(history, sim_time, lambda entry: entry.sim_time)
        for history in self._track_history.values():
            self._prune_deque(
                history,
                sim_time,
                lambda entry: entry.sim_time,
            )
        self._prune_deque(self._events_history, sim_time, lambda event: event.sim_time)

    @staticmethod
    def _prune_deque(items, sim_time, time_getter):
        while items and sim_time - time_getter(items[0]) > HISTORY_WINDOW_S:
            items.popleft()

    def _snapshot(self, sim_time, visible_target_ids):
        return MemorySnapshot(
            sim_time=sim_time,
            ownship_history={
                platform_id: tuple(history)
                for platform_id, history in self._ownship_history.items()
            },
            track_history={
                target_id: tuple(history)
                for target_id, history in self._track_history.items()
            },
            tracks=dict(self._tracks),
            events_history=tuple(self._events_history),
            visible_target_ids=frozenset(visible_target_ids),
            last_known_tracks=dict(self._last_known_tracks),
        )


def _predict_position(position, velocity, dt):
    d_north_m = velocity.north_mps * dt
    d_east_m = velocity.east_mps * dt
    latitude_rad = math.radians(position.latitude)
    new_latitude = position.latitude + math.degrees(d_north_m / EARTH_RADIUS_M)
    cos_lat = max(1.0e-9, math.cos(latitude_rad))
    new_longitude = position.longitude + math.degrees(
        d_east_m / (EARTH_RADIUS_M * cos_lat)
    )
    return _PositionLike(
        latitude=new_latitude,
        longitude=_normalize_longitude(new_longitude),
        altitude_m=position.altitude_m + velocity.up_mps * dt,
    )


def _normalize_longitude(longitude):
    return ((longitude + 180.0) % 360.0) - 180.0


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


@dataclass(frozen=True)
class _PositionLike:
    latitude: float
    longitude: float
    altitude_m: float
