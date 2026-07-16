import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.diagnostics import build_decision_trace
from examples.rules.llm_commander import LLMCommander, LLMStatus
from examples.rules.strategy import PlanSource, Role, StrategyMode, Tactic, TeamPlan
from examples.rules.strategy_manager import StrategyManager
from examples.rules.strategy_scorer import ScoredPlan, ScoringResult


def main():
    test_delayed_ready_consumed_once_and_second_submit()
    test_stale_and_failed_results_do_not_enter_or_consume()
    test_decision_trace_contains_candidate_audit_and_counts()
    test_network_failure_backoff_limits_requests_and_counts_events_once()
    test_success_resets_failure_backoff()
    test_no_valid_candidate_ready_is_discarded()
    test_all_deduped_ready_is_discarded()
    test_rule_replan_continues_during_llm_backoff()


def test_delayed_ready_consumed_once_and_second_submit():
    transport = _DelayedTransport(_valid_response(), delay_s=0.15)
    commander = LLMCommander(transport=transport)
    commander.reset()
    observation, memory_snapshot, situation = _world(step=10)
    current_plan = _current_plan(observation)
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    request_id = commander.submit(_summary(current_plan), observation.step_index)
    assert request_id is not None
    assert commander.submit(_summary(current_plan), observation.step_index) is None
    assert commander.stats()["request_count"] == 1
    assert commander.poll(observation.step_index).status == LLMStatus.PENDING
    ready = _poll_until(commander, observation.step_index + 1, LLMStatus.READY)
    assert len(ready.candidates) == 2

    manager = StrategyManager()
    manager.current_plan = current_plan
    manager._mark_plan_start(observation, _belief())
    scorer = _FakeScorer()
    current_score = _score(current_plan, 0.1)
    rule_plan = _rule_plan(observation)
    decision = manager.decide(
        observation,
        memory_snapshot,
        situation,
        _belief(),
        [rule_plan],
        _summary(current_plan),
        ready,
        current_score,
        scorer,
        commander,
    )
    assert scorer.calls == 1
    assert any(plan.source == PlanSource.LLM for plan in scorer.last_candidates)
    assert commander.stats()["consumed_count"] == 1
    assert commander.poll(observation.step_index + 2).status == LLMStatus.IDLE
    assert decision.metadata["llm_consumed"] is True

    commander.prepare_parse_context(observation, memory_snapshot, situation)
    second = commander.submit(_summary(manager.current_plan), observation.step_index + 3)
    assert second is not None
    assert commander.stats()["request_count"] == 2
    commander.close()


def test_stale_and_failed_results_do_not_enter_or_consume():
    stale_commander = LLMCommander(transport=_DelayedTransport(_valid_response(), delay_s=0.01))
    stale_commander.reset()
    observation, memory_snapshot, situation = _world(step=1)
    stale_commander.prepare_parse_context(observation, memory_snapshot, situation)
    stale_commander.submit(_summary(_current_plan(observation)), observation.step_index)
    stale = _poll_until(stale_commander, observation.step_index + 200, LLMStatus.STALE)
    assert stale.status == LLMStatus.STALE
    assert stale_commander.stats()["stale_count"] == 1
    stale_commander.close()

    failed_commander = LLMCommander(transport=_DelayedTransport(_invalid_response(), delay_s=0.01))
    failed_commander.reset()
    failed_commander.prepare_parse_context(observation, memory_snapshot, situation)
    failed_commander.submit(_summary(_current_plan(observation)), observation.step_index)
    failed = _poll_until(failed_commander, observation.step_index + 1, LLMStatus.READY)
    assert failed.status == LLMStatus.READY
    assert len(failed.candidates) == 0
    assert failed_commander.stats()["consumed_count"] == 0
    failed_commander.close()


def test_decision_trace_contains_candidate_audit_and_counts():
    transport = _DelayedTransport(_valid_response(), delay_s=0.01)
    commander = LLMCommander(transport=transport)
    commander.reset()
    observation, memory_snapshot, situation = _world(step=20)
    current_plan = _current_plan(observation)
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    commander.submit(_summary(current_plan), observation.step_index)
    ready = _poll_until(commander, observation.step_index + 1, LLMStatus.READY)

    manager = StrategyManager()
    manager.current_plan = current_plan
    manager._mark_plan_start(observation, _belief())
    scorer = _FakeScorer()
    current_score = _score(current_plan, 0.1)
    rule_plan = _rule_plan(observation)
    decision = manager.decide(
        observation,
        memory_snapshot,
        situation,
        _belief(),
        [rule_plan],
        _summary(current_plan),
        ready,
        current_score,
        scorer,
        commander,
    )
    trace = build_decision_trace(
        observation,
        memory_snapshot,
        _belief(),
        current_plan,
        decision,
        decision.selected_plan,
        current_score,
        manager.current_plan_score,
        [rule_plan],
        ready,
        actions=SimpleNamespace(model_dump=lambda: {"actions": []}),
        timings={"strategy_manager": 1.0, "total": 2.0},
        total_act_ms=2.0,
        llm_stats=commander.stats(),
    )
    assert trace.generated_rule_count == 1
    assert trace.generated_llm_count == 2
    assert trace.dedupe_before_count == 2
    assert trace.dedupe_after_count == 2
    assert trace.scored_count == 2
    assert trace.candidate_policy == "LLM_ONLY"
    assert trace.llm_primary_active is True
    assert trace.rule_candidates_used_as_fallback is False
    assert trace.llm_request_count == 1
    assert trace.llm_response_count == 1
    assert trace.llm_consumed_count == 1
    assert trace.llm_status_step_counts["READY"] == 1
    assert trace.candidate_audit
    assert any(record.get("source") == "LLM" for record in trace.candidate_audit)
    assert any(record.get("reject_reason") in {None, "not selected"} for record in trace.candidate_audit)
    commander.close()


def test_network_failure_backoff_limits_requests_and_counts_events_once():
    params_path = _params_file(retry_backoff_initial_steps=4, retry_backoff_multiplier=2.0, retry_backoff_max_steps=10)
    commander = LLMCommander(params_path=params_path, transport=_FailingTransport(RuntimeError("network down")))
    commander.reset()
    current_plan = _current_plan(_world(step=0)[0])
    for step in range(20):
        commander.submit(_summary(current_plan), step)
        commander.poll(step)
        time.sleep(0.01)
        commander.poll(step)
    stats = commander.stats()
    assert stats["request_count"] <= 4
    assert stats["response_count"] == stats["failure_event_count"]
    failed_before = stats["failure_event_count"]
    for step in range(20, 24):
        result = commander.poll(step)
        assert result.status == LLMStatus.FAILED
    stats = commander.stats()
    assert stats["failure_event_count"] == failed_before
    assert stats["status_step_counts"]["FAILED"] > stats["failure_event_count"]
    commander.close()


def test_success_resets_failure_backoff():
    params_path = _params_file(retry_backoff_initial_steps=4, retry_backoff_multiplier=2.0, retry_backoff_max_steps=10)
    transport = _SequenceTransport([RuntimeError("network down"), _valid_response()])
    commander = LLMCommander(params_path=params_path, transport=transport)
    commander.reset()
    observation, memory_snapshot, situation = _world(step=0)
    current_plan = _current_plan(observation)
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    commander.submit(_summary(current_plan), 0)
    _poll_until(commander, 0, LLMStatus.FAILED)
    assert commander.stats()["consecutive_failures"] == 1
    assert commander.submit(_summary(current_plan), 1) is None
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    assert commander.submit(_summary(current_plan), 4) is not None
    _poll_until(commander, 5, LLMStatus.READY)
    stats = commander.stats()
    assert stats["consecutive_failures"] == 0
    assert stats["next_retry_step"] == 0
    commander.close()


def test_no_valid_candidate_ready_is_discarded():
    commander = LLMCommander(transport=_DelayedTransport(_invalid_response(), delay_s=0.01))
    commander.reset()
    observation, memory_snapshot, situation = _world(step=30)
    current_plan = _current_plan(observation)
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    commander.submit(_summary(current_plan), observation.step_index)
    ready = _poll_until(commander, observation.step_index + 1, LLMStatus.READY)
    assert not ready.candidates
    manager = StrategyManager()
    manager.current_plan = current_plan
    manager._mark_plan_start(observation, _belief())
    scorer = _FakeScorer()
    decision = manager.decide(
        observation,
        memory_snapshot,
        situation,
        _belief(),
        [_rule_plan(observation)],
        _summary(current_plan),
        ready,
        _score(current_plan, 0.1),
        scorer,
        commander,
    )
    assert decision.metadata["llm_discard_reason"] == "no_valid_candidate"
    assert commander.stats()["discarded_count"] == 1
    assert commander.stats()["discard_reasons"]["no_valid_candidate"] == 1
    assert commander.poll(observation.step_index + 2).status == LLMStatus.IDLE
    commander.close()


def test_all_deduped_ready_is_discarded():
    commander = LLMCommander(transport=_DelayedTransport(_duplicate_rule_response(), delay_s=0.01))
    commander.reset()
    observation, memory_snapshot, situation = _world(step=40)
    current_plan = _current_plan(observation)
    commander.prepare_parse_context(observation, memory_snapshot, situation)
    commander.submit(_summary(current_plan), observation.step_index)
    ready = _poll_until(commander, observation.step_index + 1, LLMStatus.READY)
    manager = StrategyManager()
    manager.current_plan = current_plan
    manager._mark_plan_start(observation, _belief())
    scorer = _FakeScorer()
    decision = manager.decide(
        observation,
        memory_snapshot,
        situation,
        _belief(),
        [_rule_plan(observation)],
        _summary(current_plan),
        ready,
        _score(current_plan, 0.1),
        scorer,
        commander,
    )
    assert decision.metadata["llm_unique_count"] == 1
    assert decision.metadata["dedupe_before_count"] == 2
    assert decision.metadata["dedupe_after_count"] == 1
    assert scorer.calls == 1
    assert len(scorer.last_candidates) == 1
    assert all(plan.source == PlanSource.LLM for plan in scorer.last_candidates)
    assert commander.stats()["consumed_count"] == 1
    commander.close()


def test_rule_replan_continues_during_llm_backoff():
    params_path = _params_file(retry_backoff_initial_steps=10, retry_backoff_multiplier=2.0, retry_backoff_max_steps=20)
    commander = LLMCommander(params_path=params_path, transport=_FailingTransport(RuntimeError("network down")))
    commander.reset()
    observation, memory_snapshot, situation = _world(step=50)
    current_plan = _current_plan(observation)
    commander.submit(_summary(current_plan), observation.step_index)
    failed = _poll_until(commander, observation.step_index, LLMStatus.FAILED)
    manager = StrategyManager()
    manager.current_plan = current_plan
    manager._mark_plan_start(observation, _belief())
    scorer = _FakeScorer()
    rule_plan = _rule_plan(observation)
    decision = manager.decide(
        observation,
        memory_snapshot,
        situation,
        _belief(),
        [rule_plan],
        _summary(current_plan),
        failed,
        _score(current_plan, 0.1),
        scorer,
        commander,
    )
    assert scorer.calls == 0
    assert not scorer.last_candidates
    assert decision.selected_plan == current_plan
    assert decision.metadata["candidate_policy"] == "LLM_ONLY"
    assert decision.metadata["scored_count"] == 0
    assert commander.stats()["request_count"] == 1
    commander.close()


class _DelayedTransport:
    def __init__(self, response, delay_s):
        self.response = response
        self.delay_s = delay_s
        self.calls = 0

    def __call__(self, payload, params):
        self.calls += 1
        time.sleep(self.delay_s)
        return self.response


class _FailingTransport:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self, payload, params):
        self.calls += 1
        raise self.error


class _SequenceTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, payload, params):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _FakeScorer:
    def __init__(self):
        self.calls = 0
        self.last_candidates = []

    def score_candidates(self, observation, memory_snapshot, situation, belief, candidates, current_plan):
        self.calls += 1
        self.last_candidates = list(candidates)
        scored = []
        for index, plan in enumerate(candidates):
            value = 0.2 + index * 0.1
            if plan.source == PlanSource.LLM:
                value += 1.0
            scored.append(_score(plan, value))
        ranked = tuple(sorted(scored, key=lambda item: (-item.final_score, item.plan.plan_id)))
        return ScoringResult(tuple(scored), ranked, ranked[0] if ranked else None, 1.0)


def _poll_until(commander, step, status):
    deadline = time.time() + 3.0
    last = None
    while time.time() < deadline:
        last = commander.poll(step)
        if last.status == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"expected {status}, got {last}")


def _world(step):
    observation = SimpleNamespace(
        step_index=step,
        sim_time=float(step),
        side="blue",
        controlled_platform_ids=("blue_1", "blue_2"),
        own_units=(
            _unit("blue_1"),
            _unit("blue_2"),
        ),
        tracks=(),
    )
    track = SimpleNamespace(status="OBSERVED")
    memory_snapshot = SimpleNamespace(
        tracks={"red_1": track, "red_2": track},
        visible_target_ids=frozenset({"red_1", "red_2"}),
    )
    situation = SimpleNamespace(tracks=())
    return observation, memory_snapshot, situation


def _unit(platform_id):
    return SimpleNamespace(
        platform_id=platform_id,
        weapons=(SimpleNamespace(name="aam_medium", count=2, enabled=True),),
    )


def _current_plan(observation):
    return TeamPlan(
        plan_id="current_plan",
        created_step=observation.step_index,
        created_sim_time=observation.sim_time,
        mode=StrategyMode.PEER,
        tactic=Tactic.DISENGAGE,
        roles={platform_id: Role.DEFENDER for platform_id in observation.controlled_platform_ids},
        target_assignments={platform_id: None for platform_id in observation.controlled_platform_ids},
        primary_target=None,
        valid_for_steps=2,
        source=PlanSource.BASELINE,
        rationale=["test"],
        metadata={},
    )


def _rule_plan(observation):
    return TeamPlan(
        plan_id="rule_focus",
        created_step=observation.step_index,
        created_sim_time=observation.sim_time,
        mode=StrategyMode.PEER,
        tactic=Tactic.FOCUS_FIRE,
        roles={platform_id: Role.PRESSER for platform_id in observation.controlled_platform_ids},
        target_assignments={platform_id: "red_1" for platform_id in observation.controlled_platform_ids},
        primary_target="red_1",
        valid_for_steps=3,
        source=PlanSource.RULE,
        rationale=["test rule"],
        metadata={},
    )


def _score(plan, value):
    return ScoredPlan(
        plan=plan,
        valid=True,
        invalid_reasons=(),
        final_score=value,
        expected_utility=value,
        worst_case_utility=value - 0.1,
        switch_cost=0.0,
        hypothesis_scores=(),
        summary="test",
    )


def _belief():
    return SimpleNamespace(posterior={"FOCUS_BLUE_1": 1.0})


def _summary(current_plan):
    return {
        "current_plan": {
            "plan_id": current_plan.plan_id,
        }
    }


def _valid_response():
    candidates = {
        "candidates": [
            _candidate("FOCUS_FIRE", "red_1"),
            _candidate("SEPARATE_ATTACK", None, {"blue_1": "red_1", "blue_2": "red_2"}),
        ]
    }
    return json.dumps({"choices": [{"message": {"content": json.dumps(candidates)}}]})


def _invalid_response():
    candidates = {
        "candidates": [
            _candidate("FOCUS_FIRE", "missing_target"),
            _candidate("SEPARATE_ATTACK", None, {"blue_1": "red_1", "bad_blue": "red_2"}),
        ]
    }
    return json.dumps({"choices": [{"message": {"content": json.dumps(candidates)}}]})


def _duplicate_rule_response():
    candidate = {
        "mode": "PEER",
        "tactic": "DISENGAGE",
        "primary_target": None,
        "roles": {"blue_1": "DEFENDER", "blue_2": "DEFENDER"},
        "target_assignments": {"blue_1": None, "blue_2": None},
        "valid_for_steps": 3,
        "rationale": ["test"],
    }
    candidates = {
        "candidates": [
            candidate,
            candidate,
        ]
    }
    return json.dumps({"choices": [{"message": {"content": json.dumps(candidates)}}]})


def _params_file(**overrides):
    source = Path("examples/rules/llm_params.json")
    params = json.loads(source.read_text(encoding="utf-8"))
    params.update(overrides)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    with handle:
        json.dump(params, handle)
    return handle.name


def _candidate(tactic, primary_target, assignments=None):
    if assignments is None:
        assignments = {"blue_1": primary_target, "blue_2": primary_target}
    return {
        "mode": "PEER",
        "tactic": tactic,
        "primary_target": primary_target,
        "roles": {"blue_1": "PRESSER", "blue_2": "PRESSER"},
        "target_assignments": assignments,
        "valid_for_steps": 3,
        "rationale": ["test"],
    }


if __name__ == "__main__":
    main()
