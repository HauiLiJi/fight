from dataclasses import dataclass, field

from air_combat_challenge.competition.models import ActionBatchV1

from . import settings
from .bvr import BvrController, fighter_tracks
from .fallback_rules import assign_targets, fallback_action_batch
from .geometry import unit_risk_score
from .missile_threat import MissileThreatTracker


HOT_RANGE_M = 150000.0
FLANK_RANGE_M = 115000.0
COLD_RANGE_M = 85000.0
CRANK_ANGLE_DEG = 47.0
CRANK_DURATION_S = 58.0
THREAT_TIMEOUT_S = 110.0
DEFENSE_DURATION_S = 72.0
SHOT_MEMORY_S = 85.0
LAUNCH_COOLDOWN_S = 35.0
NORMAL_HEADING_ERROR_DEG = 28.0
COUNTER_HEADING_ERROR_DEG = 36.0
HIGH_RISK_SCORE = 5.0
MEDIUM_RISK_SCORE = 3.0
HIGH_RISK_ESCAPE_OFFSET_DEG = 30.0
PRESS_OFFSET_DEG = 18.0
POST_LAUNCH_OFFSET_DEG = 32.0
DEFENSE_OFFSET_DEG = 32.0
CONTACT_MACH_LOW_RISK = 0.9
CONTACT_MACH_MED_RISK = 0.86
POST_LAUNCH_MACH = 0.93
CRANK_MACH = 1.18
DEFENSE_MACH = 1.4
ALTITUDE_SPLIT_M = 850.0
DEFENSE_ALTITUDE_SPLIT_M = 700.0


@dataclass
class Context:
    observation: object
    threats: dict = field(default_factory=dict)
    result: ActionBatchV1 = field(default_factory=ActionBatchV1)


class Strategy3BehaviorTree:
    """Behavior-tree shell around the LLM-disabled Strategy 3 rule controller."""

    def __init__(self):
        self._last_launch_time = {}
        self._missile_threat_tracker = MissileThreatTracker()
        self._bvr_controller = BvrController()
        self.tree = None

    def reset(self):
        self._last_launch_time.clear()
        self._missile_threat_tracker.reset()
        self._bvr_controller.reset()

    def act(self, observation):
        _sync_parameters()
        context = Context(observation=observation)
        if observation.own_units:
            context.threats = self._missile_threat_tracker.update(
                observation, settings.THREAT_TIMEOUT_S
            )
        self.tree.tick(context)
        return context.result

    @staticmethod
    def _no_own_units(context):
        return not context.observation.own_units

    @staticmethod
    def _no_enemy_tracks(context):
        return not fighter_tracks(context.observation)

    @staticmethod
    def _has_active_threat(context):
        return bool(context.threats.get("active_threats"))

    def _has_launch_opportunity(self, context):
        observation = context.observation
        units = sorted(observation.own_units, key=lambda unit: unit.platform_id)
        tracks = fighter_tracks(observation)
        for unit in units:
            target = assign_targets(units, tracks).get(unit.platform_id) if tracks else None
            if target is not None and self._bvr_controller.can_fire(
                observation.sim_time, unit, target
            ):
                return True
        return False

    @staticmethod
    def _risk_scores(context):
        observation = context.observation
        units = sorted(observation.own_units, key=lambda unit: unit.platform_id)
        tracks = fighter_tracks(observation)
        return [
            unit_risk_score(unit, tracks, units, index, context.threats)["score"]
            for index, unit in enumerate(units)
        ]

    def _any_high_risk(self, context):
        return any(score >= settings.HIGH_RISK_SCORE for score in self._risk_scores(context))

    def _any_medium_risk(self, context):
        return any(score >= settings.MEDIUM_RISK_SCORE for score in self._risk_scores(context))

    @staticmethod
    def _idle(context):
        context.result = ActionBatchV1()

    @staticmethod
    def _hold_no_contact(context):
        actions = [
            {
                "type": "set_flight",
                "platform_id": unit.platform_id,
                "heading_deg": unit.attitude.heading_deg,
                "altitude_m": max(7000.0, min(9500.0, unit.position.altitude_m)),
                "mach": 0.72,
            }
            for unit in sorted(context.observation.own_units, key=lambda item: item.platform_id)
        ]
        context.result = ActionBatchV1.model_validate({"actions": actions})

    def _strategy3_rule_pipeline(self, context):
        self._run_pipeline(context, "standard")

    def _press_pipeline(self, context):
        self._run_pipeline(context, "press")

    def _defensive_pipeline(self, context):
        self._run_pipeline(context, "defensive")

    def _run_pipeline(self, context, profile):
        batch = fallback_action_batch(
            context.observation,
            self._last_launch_time,
            self._bvr_controller,
            profile=profile,
        )
        actions = [
            action.model_dump() if hasattr(action, "model_dump") else dict(action)
            for action in batch.actions
        ]
        context.result = ActionBatchV1.model_validate(
            {
                "actions": self._bvr_controller.apply(
                    context.observation,
                    actions,
                    self._last_launch_time,
                    context.threats,
                )
            }
        )


def _sync_parameters():
    for name in vars(settings):
        if name.isupper() and name in globals():
            setattr(settings, name, globals()[name])
