"""Remember previously detected enemies while the radar temporarily loses track."""

from dataclasses import dataclass

from .bvr import is_missile_track
from .geometry import GeoPoint, predict_position


CONTACT_PREDICTION_LIMIT_S = 120.0


@dataclass
class EnemyContact:
    target_id: str
    position: GeoPoint
    north_mps: float
    east_mps: float
    up_mps: float
    last_seen_sim_time: float
    destroyed: bool = False


class EnemyContactTracker:
    def __init__(self):
        self._contacts = {}

    def reset(self):
        self._contacts.clear()

    def update(self, observation):
        visible_ids = set()
        for track in observation.tracks:
            if track.target_side == observation.side or is_missile_track(track):
                continue
            visible_ids.add(track.target_id)
            self._contacts[track.target_id] = EnemyContact(
                target_id=track.target_id,
                position=GeoPoint(
                    latitude=track.position.latitude,
                    longitude=track.position.longitude,
                    altitude_m=track.position.altitude_m,
                ),
                north_mps=track.velocity.north_mps,
                east_mps=track.velocity.east_mps,
                up_mps=track.velocity.up_mps,
                last_seen_sim_time=observation.sim_time,
            )

        for event in observation.events:
            if event.event_type != "PlatformBroken":
                continue
            for platform_id in (event.platform, event.target):
                if platform_id in self._contacts:
                    self._contacts[platform_id].destroyed = True

        return self.snapshot(observation.sim_time, visible_ids)

    def snapshot(self, sim_time, visible_ids=()):
        visible_ids = set(visible_ids)
        known_alive_ids = []
        destroyed_ids = []
        lost_contacts = []
        for target_id, contact in sorted(self._contacts.items()):
            if contact.destroyed:
                destroyed_ids.append(target_id)
                continue
            known_alive_ids.append(target_id)
            if target_id in visible_ids:
                continue
            age_s = max(0.0, sim_time - contact.last_seen_sim_time)
            prediction_elapsed_s = min(age_s, CONTACT_PREDICTION_LIMIT_S)
            lost_contacts.append(
                {
                    "target_id": target_id,
                    "last_seen_sim_time": round(contact.last_seen_sim_time, 1),
                    "age_s": round(age_s, 1),
                    "prediction_capped": age_s > CONTACT_PREDICTION_LIMIT_S,
                    "predicted_position": predict_position(
                        contact.position,
                        contact.north_mps,
                        contact.east_mps,
                        contact.up_mps,
                        prediction_elapsed_s,
                    ),
                }
            )
        return {
            "known_alive_ids": known_alive_ids,
            "visible_ids": sorted(visible_ids),
            "destroyed_ids": destroyed_ids,
            "lost_contact_ids": [item["target_id"] for item in lost_contacts],
            "lost_contacts": lost_contacts,
            "last_contact_age_s": (
                min((item["age_s"] for item in lost_contacts), default=None)
            ),
        }
