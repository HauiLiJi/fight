"""Track the red formation roster so destroyed aircraft remain visible to the LLM."""


class RedForceStatusTracker:
    def __init__(self):
        self._platform_ids = ()
        self._destroyed_at = {}

    def reset(self, controlled_platform_ids):
        self._platform_ids = tuple(sorted(controlled_platform_ids))
        self._destroyed_at.clear()

    def update(self, observation):
        alive_ids = {unit.platform_id for unit in observation.own_units}
        break_times = {
            event.platform: event.sim_time
            for event in observation.events
            if event.event_type == "PlatformBroken" and event.platform in self._platform_ids
        }
        newly_destroyed = []
        for platform_id in self._platform_ids:
            if platform_id not in alive_ids and platform_id not in self._destroyed_at:
                self._destroyed_at[platform_id] = break_times.get(
                    platform_id,
                    observation.sim_time,
                )
                newly_destroyed.append(platform_id)

        units = []
        for platform_id in self._platform_ids:
            alive = platform_id in alive_ids
            units.append(
                {
                    "platform_id": platform_id,
                    "status": "alive" if alive else "destroyed",
                    "destroyed_at_sim_time": None if alive else self._destroyed_at[platform_id],
                }
            )
        return {
            "initial_count": len(self._platform_ids),
            "alive_count": len(alive_ids.intersection(self._platform_ids)),
            "destroyed_count": len(self._destroyed_at),
            "units": units,
            "newly_destroyed_ids": tuple(sorted(newly_destroyed)),
        }
