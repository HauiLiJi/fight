from .models import ScoreOutcomeV1


class EliminationScorer:
    def __init__(self, max_steps=2000, max_sim_time=None):
        self.max_steps = max_steps
        self.max_sim_time = max_sim_time

    def evaluate(self, sim_data, step_index):
        blue_alive = self._alive_fighters(sim_data.get("blue", {}))
        red_alive = self._alive_fighters(sim_data.get("red", {}))
        if not blue_alive and not red_alive:
            return self._terminal("draw", "simultaneous_elimination")
        if not blue_alive:
            return self._terminal("red", "blue_eliminated")
        if not red_alive:
            return self._terminal("blue", "red_eliminated")
        if step_index >= self.max_steps:
            return self._draw("max_steps")
        if self.max_sim_time is not None and float(sim_data.get("time", 0.0)) >= self.max_sim_time:
            return self._draw("max_sim_time")
        return ScoreOutcomeV1(
            rewards={"blue": 0.0, "red": 0.0},
            terminated=False,
            truncated=False,
        )

    @staticmethod
    def forfeit(side, reason="agent_forfeit"):
        winner = "red" if side == "blue" else "blue"
        return EliminationScorer._terminal(winner, reason)

    @staticmethod
    def _alive_fighters(side_data):
        return [
            platform_id
            for platform_id, platform in side_data.items()
            if isinstance(platform, dict) and platform.get("entity_type") == "fighter"
        ]

    @staticmethod
    def _terminal(winner, reason):
        if winner == "draw":
            rewards = {"blue": 0.0, "red": 0.0}
        else:
            loser = "red" if winner == "blue" else "blue"
            rewards = {winner: 1.0, loser: -1.0}
        return ScoreOutcomeV1(
            rewards=rewards,
            terminated=True,
            truncated=False,
            winner=winner,
            reason=reason,
        )

    @staticmethod
    def _draw(reason):
        return ScoreOutcomeV1(
            rewards={"blue": 0.0, "red": 0.0},
            terminated=False,
            truncated=True,
            winner="draw",
            reason=reason,
        )
