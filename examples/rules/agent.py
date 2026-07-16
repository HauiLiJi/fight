from air_combat_challenge.competition.agents import BaseAgent

from .tactics import CrankCounterTree


class CrankCounterAgent(BaseAgent):
    """Two-ship defensive doctrine using fixed crank legs and counterfire."""

    def __init__(self):
        self._tree = CrankCounterTree()

    def reset(self, context):
        super().reset(context)
        self._tree.reset()

    def act(self, observation):
        return self._tree.act(observation)
