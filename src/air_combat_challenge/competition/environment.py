import random
import uuid

from .action_gateway import ActionGateway
from .models import ActionBatchV1, EpisodeStartV1, EventV1, StepResultV1
from .observation import ObservationBuilder
from .scoring import EliminationScorer


class EventCursor:
    def __init__(self):
        self.last_event_id = -1

    def reset(self):
        self.last_event_id = -1

    def parse_new(self, raw_events):
        if isinstance(raw_events, dict):
            values = list(raw_events.values())
        elif isinstance(raw_events, list):
            values = raw_events
        else:
            values = []

        events = []
        allowed_fields = set(EventV1.model_fields)
        for raw_event in values:
            if not isinstance(raw_event, dict):
                continue
            event_id = int(raw_event.get("event_id", -1))
            if event_id <= self.last_event_id:
                continue
            payload = {key: value for key, value in raw_event.items() if key in allowed_fields}
            payload["event_id"] = event_id
            events.append(EventV1.model_validate(payload))
        events.sort(key=lambda event: event.event_id)
        if events:
            self.last_event_id = events[-1].event_id
        return events


class CompetitionEnv:
    def __init__(
        self,
        raw_env,
        max_steps=2000,
        max_sim_time=None,
        global_view=False,
        observation_builder=None,
        action_gateway=None,
        scorer=None,
    ):
        self.raw_env = raw_env
        self.global_view = global_view
        self.observation_builder = observation_builder or ObservationBuilder()
        self.action_gateway = action_gateway or ActionGateway()
        self.scorer = scorer or EliminationScorer(max_steps=max_steps, max_sim_time=max_sim_time)
        self.event_cursor = EventCursor()
        self.episode_id = None
        self.step_index = 0
        self.sim_data = None
        self.extra_info = {}
        self.controlled = {"blue": [], "red": []}
        self.observations = {}

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.episode_id = str(uuid.uuid4())
        self.step_index = 0
        self.event_cursor.reset()
        self.sim_data, self.extra_info = self.raw_env.reset()
        self.controlled = {
            side: self._alive_fighters(self.sim_data.get(side, {}))
            for side in ("blue", "red")
        }
        self.observations = self._build_observations(events=[])
        return EpisodeStartV1(episode_id=self.episode_id, observations=self.observations)

    def observation_for(self, side, controlled_platform_ids):
        return self.observation_builder.build(
            self.sim_data,
            side=side,
            controlled_platform_ids=controlled_platform_ids,
            episode_id=self.episode_id,
            step_index=self.step_index,
            mission=self._mission(),
            global_view=self.global_view,
        )

    def step(self, actions_by_side):
        gateway_results = {}
        command_dict = {"blue": [], "red": []}
        for side in ("blue", "red"):
            batch = actions_by_side.get(side, ActionBatchV1())
            if not isinstance(batch, ActionBatchV1):
                batch = ActionBatchV1.model_validate(batch)
            gateway_result = self.action_gateway.validate_and_translate(side, batch, self.observations[side])
            gateway_results[side] = gateway_result
            command_dict[side].extend(gateway_result.commands)
        return self.step_validated(command_dict, gateway_results)

    def step_validated(self, command_dict, gateway_results):
        self.sim_data, self.extra_info = self.raw_env.step(command_dict)
        self.step_index += 1
        self._apply_execution_results(gateway_results)
        events = self.event_cursor.parse_new(self._get_raw_events())
        self.observations = self._build_observations(events)
        outcome = self.scorer.evaluate(self.sim_data, self.step_index)
        return StepResultV1(
            episode_id=self.episode_id,
            step_index=self.step_index,
            sim_time=float(self.sim_data.get("time", 0.0)),
            observations=self.observations,
            rewards=outcome.rewards,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            winner=outcome.winner,
            reason=outcome.reason,
            events=events,
            action_reports={side: result.reports for side, result in gateway_results.items()},
            diagnostics={"mission": self._mission()},
        )

    def _build_observations(self, events):
        return {
            side: self.observation_builder.build(
                self.sim_data,
                side=side,
                controlled_platform_ids=self.controlled[side],
                episode_id=self.episode_id,
                step_index=self.step_index,
                events=events,
                mission=self._mission(),
                global_view=self.global_view,
            )
            for side in ("blue", "red")
        }

    def _get_raw_events(self):
        get_events = getattr(self.raw_env, "get_events", None)
        if get_events is None:
            return {}
        try:
            return get_events()
        except (ValueError, KeyError, RuntimeError):
            return {}

    def _apply_execution_results(self, gateway_results):
        raw_results = self.extra_info.get("command_results", {})
        for side, gateway_result in gateway_results.items():
            accepted_reports = [report for report in gateway_result.reports if report.status == "accepted"]
            for report, response in zip(accepted_reports, raw_results.get(side, [])):
                code = response.get("code") if isinstance(response, dict) else getattr(response, "code", 0)
                if code not in (None, 0):
                    report.status = "execution_error"
                    report.reason_code = "afsim_rejected"
                    report.message = str(
                        response.get("message", "")
                        if isinstance(response, dict)
                        else getattr(response, "message", "")
                    )

    def _mission(self):
        return self.extra_info.get("mission_config") or {
            "mission_type": self.extra_info.get("mission_type")
        }

    @staticmethod
    def _alive_fighters(side_data):
        return sorted(
            platform_id
            for platform_id, platform in side_data.items()
            if isinstance(platform, dict) and platform.get("entity_type") == "fighter"
        )
