from dataclasses import dataclass
import time

from .action_gateway import GatewayResult
from .models import ActionReportV1, EpisodeContextV1
from .replay import ReplayWriter, file_sha256
from .scoring import EliminationScorer
from .worker import AgentExecutionError, AgentWorker, AgentWorkerError


@dataclass
class AgentController:
    side: str
    controlled_platform_ids: list
    handle: object
    context: EpisodeContextV1
    restart_count: int = 0
    startup_error: str = None


class CompetitionRunner:
    def __init__(
        self,
        env,
        blue_agent,
        red_agent,
        worker_factory=AgentWorker,
        output_dir="runs",
        max_steps=None,
        step_delay_s=0.0,
        max_consecutive_violations=3,
        max_total_violations=10,
        time_scale=None,
        should_stop=None,
        on_episode_start=None,
        on_step=None,
        on_episode_complete=None,
    ):
        self.env = env
        self.loaded_agents = {"blue": blue_agent, "red": red_agent}
        self.worker_factory = worker_factory
        self.output_dir = output_dir
        self.max_steps = max_steps or getattr(env.scorer, "max_steps", 2000)
        self.step_delay_s = step_delay_s
        self.max_consecutive_violations = max_consecutive_violations
        self.max_total_violations = max_total_violations
        self.time_scale = time_scale
        self.should_stop = should_stop
        self.on_episode_start = on_episode_start
        self.on_step = on_step
        self.on_episode_complete = on_episode_complete

    def run_episode(self, seed=None):
        episode_start = self.env.reset(seed=seed)
        replay = ReplayWriter(self.output_dir, episode_start.episode_id)
        violations = {
            "blue": {"consecutive": 0, "total": 0},
            "red": {"consecutive": 0, "total": 0},
        }
        replay.write(
            "episode_start",
            {
                "episode": episode_start.model_dump(mode="json"),
                "seed": seed,
                "agent_hashes": {
                    side: loaded.source_hash for side, loaded in self.loaded_agents.items()
                },
                "scenario_hash": self._scenario_hash(),
            },
        )
        self._notify(self.on_episode_start, episode_start)

        initial_outcome = self.env.scorer.evaluate(self.env.sim_data, 0)
        if initial_outcome.terminated or initial_outcome.truncated:
            summary = self._make_summary(
                episode_start.episode_id,
                seed,
                0,
                initial_outcome.winner,
                initial_outcome.reason,
                violations,
                replay,
            )
            replay.write("summary", summary)
            replay.write_summary(summary)
            self._notify(self.on_episode_complete, summary)
            return summary

        controllers = self._create_controllers(episode_start, seed)

        final_winner = None
        final_reason = None
        executed_steps = 0
        stopped_by_user = False
        try:
            for _ in range(self.max_steps):
                step_started_at = time.monotonic()
                if self._stop_requested():
                    stopped_by_user = True
                    final_winner = "draw"
                    final_reason = "stopped_by_user"
                    break
                command_dict = {"blue": [], "red": []}
                combined = {
                    "blue": GatewayResult(commands=[], reports=[]),
                    "red": GatewayResult(commands=[], reports=[]),
                }
                side_failed = {"blue": False, "red": False}
                submitted_batches = {"blue": [], "red": []}
                for controller in controllers:
                    observation = self.env.observation_for(
                        controller.side,
                        controller.controlled_platform_ids,
                    )
                    try:
                        if controller.startup_error is not None:
                            raise AgentExecutionError(controller.startup_error)
                        batch = controller.handle.act(observation)
                        submitted_batches[controller.side].append(
                            {
                                "controlled_platform_ids": controller.controlled_platform_ids,
                                "batch": batch.model_dump(mode="json"),
                            }
                        )
                        result = self.env.action_gateway.validate_and_translate(
                            controller.side,
                            batch,
                            observation,
                        )
                        combined[controller.side].commands.extend(result.commands)
                        combined[controller.side].reports.extend(result.reports)
                        side_failed[controller.side] |= result.has_violation
                    except (AgentWorkerError, ValueError, TypeError) as error:
                        side_failed[controller.side] = True
                        combined[controller.side].reports.append(
                            self._agent_failure_report(controller, error)
                        )
                        self._restart_once(controller)

                forfeiting_side = self._update_violations(violations, side_failed)
                if forfeiting_side is not None:
                    outcome = EliminationScorer.forfeit(
                        forfeiting_side,
                        reason="agent_violation_limit",
                    )
                    final_winner = outcome.winner
                    final_reason = outcome.reason
                    replay.write(
                        "forfeit",
                        {
                            "side": forfeiting_side,
                            "violations": violations,
                            "reports": {
                                side: [report.model_dump(mode="json") for report in result.reports]
                                for side, result in combined.items()
                            },
                        },
                    )
                    break

                for side in ("blue", "red"):
                    command_dict[side].extend(combined[side].commands)
                try:
                    step_result = self.env.step_validated(command_dict, combined)
                except Exception:
                    if not self._stop_requested():
                        raise
                    stopped_by_user = True
                    final_winner = "draw"
                    final_reason = "stopped_by_user"
                    break
                executed_steps += 1
                replay.write(
                    "step",
                    {
                        "submitted_actions": submitted_batches,
                        "commands": command_dict,
                        "result": step_result.model_dump(mode="json"),
                    },
                )
                self._notify(self.on_step, step_result)
                if step_result.terminated or step_result.truncated:
                    final_winner = step_result.winner
                    final_reason = step_result.reason
                    break
                if self._wait_for_next_step(step_started_at):
                    stopped_by_user = True
                    final_winner = "draw"
                    final_reason = "stopped_by_user"
                    break
        finally:
            for controller in controllers:
                controller.handle.close()

        if stopped_by_user:
            replay.write(
                "stopped",
                {"step_index": executed_steps, "reason": final_reason},
            )

        summary = self._make_summary(
            episode_start.episode_id,
            seed,
            executed_steps,
            final_winner,
            final_reason,
            violations,
            replay,
        )
        replay.write("summary", summary)
        replay.write_summary(summary)
        self._notify(self.on_episode_complete, summary)
        return summary

    @staticmethod
    def _notify(callback, payload):
        if callback is not None:
            callback(payload)

    def _stop_requested(self):
        return self.should_stop is not None and self.should_stop()

    def _wait_for_next_step(self, step_started_at):
        if self.time_scale is None:
            pause_s = self.step_delay_s
        else:
            pause_s = max(0.0, (1.0 / self.time_scale) - (time.monotonic() - step_started_at))
        deadline = time.monotonic() + pause_s
        while True:
            if self._stop_requested():
                return True
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return False
            time.sleep(min(remaining_s, 0.05))

    def _make_summary(
        self,
        episode_id,
        seed,
        executed_steps,
        winner,
        reason,
        violations,
        replay,
    ):
        return {
            "api_version": "1.0",
            "episode_id": episode_id,
            "seed": seed,
            "executed_steps": executed_steps,
            "winner": winner,
            "reason": reason,
            "violations": violations,
            "agent_hashes": {
                side: loaded.source_hash for side, loaded in self.loaded_agents.items()
            },
            "scenario_hash": self._scenario_hash(),
            "time_scale": self.time_scale,
            "replay_path": str(replay.replay_path),
        }

    def _create_controllers(self, episode_start, seed):
        controllers = []
        for side in ("blue", "red"):
            loaded = self.loaded_agents[side]
            platform_ids = self.env.controlled[side]
            scopes = [platform_ids] if loaded.manifest.topology == "team" else [[item] for item in platform_ids]
            for controlled_ids in scopes:
                handle = self.worker_factory(loaded)
                context = EpisodeContextV1(
                    episode_id=episode_start.episode_id,
                    side=side,
                    controlled_platform_ids=controlled_ids,
                    seed=seed,
                    mission=episode_start.observations[side].mission,
                )
                startup_error = None
                restart_count = 0
                try:
                    handle.start()
                    handle.reset(context)
                except AgentWorkerError as error:
                    startup_error = str(error)
                    if hasattr(handle, "restart"):
                        try:
                            handle.restart()
                            handle.reset(context)
                            startup_error = None
                            restart_count = 1
                        except AgentWorkerError as restart_error:
                            startup_error = str(restart_error)
                            restart_count = 1
                controllers.append(
                    AgentController(
                        side=side,
                        controlled_platform_ids=controlled_ids,
                        handle=handle,
                        context=context,
                        restart_count=restart_count,
                        startup_error=startup_error,
                    )
                )
        return controllers

    def _update_violations(self, violations, side_failed):
        for side in ("blue", "red"):
            if side_failed[side]:
                violations[side]["consecutive"] += 1
                violations[side]["total"] += 1
            else:
                violations[side]["consecutive"] = 0
            if (
                violations[side]["consecutive"] >= self.max_consecutive_violations
                or violations[side]["total"] >= self.max_total_violations
            ):
                return side
        return None

    @staticmethod
    def _restart_once(controller):
        if controller.restart_count >= 1 or not hasattr(controller.handle, "restart"):
            return
        try:
            controller.handle.restart()
            controller.handle.reset(controller.context)
            controller.restart_count += 1
            controller.startup_error = None
        except AgentWorkerError:
            controller.restart_count += 1

    @staticmethod
    def _agent_failure_report(controller, error):
        return ActionReportV1(
            action_index=0,
            action_type="agent_response",
            platform_id=",".join(controller.controlled_platform_ids) or controller.side,
            status="rejected",
            reason_code="agent_failure",
            message=str(error),
        )

    def _scenario_hash(self):
        scenario_path = getattr(self.env.raw_env, "scenario_path", None)
        return file_sha256(scenario_path) if scenario_path else None
