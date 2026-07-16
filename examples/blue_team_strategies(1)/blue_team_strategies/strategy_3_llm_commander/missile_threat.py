"""Track hostile missile launches from cumulative simulator events."""

from dataclasses import dataclass

from .bvr import is_missile_track


@dataclass
class MissileThreat:
    shooter_id: str
    target_id: str
    started_sim_time: float
    expires_sim_time: float


class MissileThreatTracker:
    def __init__(self, timeout_s=110.0):
        self._timeout_s = timeout_s
        self._seen_event_ids = set()
        self._threats = {}

    def reset(self):
        self._seen_event_ids.clear()
        self._threats.clear()

    def update(self, observation, timeout_s=None):
        if timeout_s is not None:
            self._timeout_s = timeout_s
        own_ids = {unit.platform_id for unit in observation.own_units}
        for event in observation.events:
            key = self._event_key(event)
            if key in self._seen_event_ids:
                continue
            self._seen_event_ids.add(key)
            if event.event_type == "WeaponFired":
                if event.target in own_ids and event.shooter and event.shooter not in own_ids:
                    started = float(getattr(event, "sim_time", observation.sim_time))
                    self._threats[(event.shooter, event.target)] = MissileThreat(
                        shooter_id=event.shooter,
                        target_id=event.target,
                        started_sim_time=started,
                        expires_sim_time=started + self._timeout_s,
                    )
            elif event.event_type in {"WeaponHit", "WeaponMissed"}:
                self._threats.pop((event.shooter, event.target), None)

        self._threats = {
            key: threat
            for key, threat in self._threats.items()
            if threat.target_id in own_ids and observation.sim_time < threat.expires_sim_time
        }
        visible_missiles = sorted(
            track.target_id
            for track in observation.tracks
            if track.target_side != observation.side and is_missile_track(track)
        )
        active = [
            {
                "shooter_id": threat.shooter_id,
                "target_id": threat.target_id,
                "age_s": round(max(0.0, observation.sim_time - threat.started_sim_time), 1),
                "remaining_s": round(max(0.0, threat.expires_sim_time - observation.sim_time), 1),
            }
            for threat in sorted(
                self._threats.values(), key=lambda item: (item.target_id, item.shooter_id)
            )
        ]
        return {
            "active_threats": active,
            "threatened_platform_ids": sorted({item["target_id"] for item in active}),
            "visible_enemy_missile_ids": visible_missiles,
        }

    @staticmethod
    def _event_key(event):
        event_id = getattr(event, "event_id", None)
        if event_id is not None:
            return ("id", event_id)
        return (
            "event",
            getattr(event, "event_type", None),
            getattr(event, "sim_time", None),
            getattr(event, "shooter", None),
            getattr(event, "target", None),
            getattr(event, "weapon", None),
            getattr(event, "platform", None),
        )
