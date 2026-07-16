from air_combat_challenge.competition.agents import BaseAgent
from air_combat_challenge.competition.models import ActionBatchV1

from .policy import Policy


class EvolvedPolicyAgent(BaseAgent):
    """Compiled open-policy candidate; policy.py is generated from behavior_tree.json."""

    def __init__(self):
        self._policy = Policy()

    def reset(self, context):
        super().reset(context)
        self._policy.reset(context)

    def act(self, observation):
        actions = self._policy.act(observation)
        return ActionBatchV1.model_validate({"actions": actions})
