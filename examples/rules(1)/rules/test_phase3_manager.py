import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.llm_commander import LLMGenerationResult, LLMStatus
from examples.rules.strategy import PlanSource, Role, StrategyMode, Tactic, TeamPlan
from examples.rules.strategy_manager import ManagerAction, StrategyManager
from examples.rules.strategy_scorer import ScoredPlan, ScoringResult
from examples.rules.team_memory import OBSERVED


def main():
    test_absolute_advantage_passes_after_required_reviews()
    test_relative_advantage_passes()
    test_advantage_too_small_rejects()
    test_leader_streak_and_candidate_change_reset()
    test_semantic_key_survives_plan_id_changes()
    test_minimum_hold_rejects()
    test_invalid_plan_forces_replan()
    test_new_llm_ready_triggers_one_review()
    test_belief_label_stability()
    test_strong_event_lowers_threshold()
    test_normal_steps_do_not_switch_every_step()


def test_absolute_advantage_passes_after_required_reviews():
    manager = _manager(leader_required_reviews=2, switch_absolute_advantage=0.03, switch_relative_advantage=10.0)
    obs = _observation(10)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    current = _score(manager.current_plan, 0.20)
    candidate = _score(_plan("candidate_a", Tactic.FOCUS_FIRE, target="red_1"), 0.24)
    first, first_detail = manager._switch_allowed_with_reason(obs, candidate, current, ["review interval"])
    second, second_detail = manager._switch_allowed_with_reason(_observation(11), candidate, current, ["review interval"])
    assert not first
    assert "leader streak below required reviews" in first_detail["reject_reasons"]
    assert second
    assert second_detail["absolute_delta"] >= 0.03


def test_relative_advantage_passes():
    manager = _manager(leader_required_reviews=1, switch_absolute_advantage=1.0, switch_relative_advantage=0.20, score_delta_epsilon=0.05)
    obs = _observation(10)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    allowed, detail = manager._switch_allowed_with_reason(
        obs,
        _score(_plan("candidate_b", Tactic.FOCUS_FIRE, target="red_1"), 0.07),
        _score(manager.current_plan, 0.05),
        ["review interval"],
    )
    assert allowed
    assert detail["relative_delta"] >= 0.20


def test_advantage_too_small_rejects():
    manager = _manager(leader_required_reviews=1, switch_absolute_advantage=0.05, switch_relative_advantage=0.5)
    obs = _observation(10)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    allowed, detail = manager._switch_allowed_with_reason(
        obs,
        _score(_plan("candidate_c", Tactic.FOCUS_FIRE, target="red_1"), 0.11),
        _score(manager.current_plan, 0.10),
        ["review interval"],
    )
    assert not allowed
    assert "score advantage below threshold" in detail["reject_reasons"]


def test_leader_streak_and_candidate_change_reset():
    manager = _manager(leader_required_reviews=2, switch_absolute_advantage=0.01, switch_relative_advantage=10.0)
    obs = _observation(10)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    current = _score(manager.current_plan, 0.10)
    a = _score(_plan("candidate_a", Tactic.FOCUS_FIRE, target="red_1"), 0.12)
    b = _score(_plan("candidate_b", Tactic.SEPARATE_ATTACK, target="red_2"), 0.13)
    manager._switch_allowed_with_reason(obs, a, current, ["review interval"])
    manager._switch_allowed_with_reason(_observation(11), b, current, ["review interval"])
    assert manager.leader_streak == 1


def test_semantic_key_survives_plan_id_changes():
    manager = _manager(leader_required_reviews=2, switch_absolute_advantage=0.01, switch_relative_advantage=10.0)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    current = _score(manager.current_plan, 0.10)
    a1 = _score(_plan("candidate_a_1", Tactic.FOCUS_FIRE, target="red_1"), 0.12)
    a2 = _score(_plan("candidate_a_2", Tactic.FOCUS_FIRE, target="red_1"), 0.12)
    first, _ = manager._switch_allowed_with_reason(_observation(10), a1, current, ["review interval"])
    second, _ = manager._switch_allowed_with_reason(_observation(11), a2, current, ["review interval"])
    assert not first
    assert second


def test_minimum_hold_rejects():
    manager = _manager(leader_required_reviews=1, switch_absolute_advantage=0.01, switch_relative_advantage=0.01)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 20
    allowed, detail = manager._switch_allowed_with_reason(
        _observation(10),
        _score(_plan("candidate", Tactic.FOCUS_FIRE, target="red_1"), 1.0),
        _score(manager.current_plan, 0.0),
        ["review interval"],
    )
    assert not allowed
    assert "minimum hold active" in detail["reject_reasons"]


def test_invalid_plan_forces_replan():
    manager = _manager(leader_required_reviews=5)
    obs = _observation(10)
    memory = _memory()
    situation = _situation()
    manager.current_plan = _plan("bad_current", Tactic.FOCUS_FIRE, target="missing")
    manager._mark_plan_start(obs, _belief("UNKNOWN"))
    candidate = _plan("rule_candidate", Tactic.FOCUS_FIRE, target="red_1")
    scorer = _FakeScorer([_score(candidate, 0.0)])
    decision = manager.decide(
        obs,
        memory,
        situation,
        _belief("UNKNOWN"),
        [candidate],
        {},
        LLMGenerationResult(None, LLMStatus.IDLE, None),
        _score(manager.current_plan, 0.0),
        scorer,
        _NoLLM(),
    )
    assert decision.action == ManagerAction.FULL_REPLAN
    assert decision.switch_allowed
    assert decision.selected_plan.plan_id == "rule_candidate"


def test_new_llm_ready_triggers_one_review():
    manager = _manager()
    obs = _observation(10)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager._mark_plan_start(obs, _belief("UNKNOWN"))
    manager.last_known_enemy_ids = {"red_1"}
    manager.last_active_own_ids = {"blue_1", "blue_2"}
    manager.last_replan_step = 0
    manager.plan_valid_until = 20
    candidate = _plan("rule_candidate", Tactic.FOCUS_FIRE, target="red_1")
    scorer = _FakeScorer([_score(candidate, 0.0)])
    decision = manager.decide(
        obs,
        _memory(),
        _situation(),
        _belief("UNKNOWN"),
        [candidate],
        {},
        LLMGenerationResult("r1", LLMStatus.READY, 9, candidates=()),
        _score(manager.current_plan, 0.0),
        scorer,
        _NoLLM(),
    )
    assert "new llm ready" in decision.metadata["triggers"]
    next_decision = manager.decide(
        _observation(11),
        _memory(),
        _situation(),
        _belief("UNKNOWN"),
        [candidate],
        {},
        LLMGenerationResult(None, LLMStatus.IDLE, None),
        _score(manager.current_plan, 0.0),
        scorer,
        _NoLLM(),
    )
    assert "new llm ready" not in next_decision.metadata.get("triggers", [])


def test_belief_label_stability():
    manager = _manager(belief_label_stable_steps=3)
    obs = _observation(0)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager._mark_plan_start(obs, _belief("UNKNOWN"))
    triggers_1 = manager._triggers(_observation(1), _memory(), _situation(), _belief("FOCUS_BLUE_1"), _score(manager.current_plan, 0.0))
    triggers_2 = manager._triggers(_observation(2), _memory(), _situation(), _belief("UNKNOWN"), _score(manager.current_plan, 0.0))
    triggers_3 = manager._triggers(_observation(3), _memory(), _situation(), _belief("FOCUS_BLUE_2"), _score(manager.current_plan, 0.0))
    triggers_4 = manager._triggers(_observation(4), _memory(), _situation(), _belief("FOCUS_BLUE_2"), _score(manager.current_plan, 0.0))
    triggers_5 = manager._triggers(_observation(5), _memory(), _situation(), _belief("FOCUS_BLUE_2"), _score(manager.current_plan, 0.0))
    triggers_6 = manager._triggers(_observation(6), _memory(), _situation(), _belief("FOCUS_BLUE_2"), _score(manager.current_plan, 0.0))
    assert "stable label shift" not in triggers_1 + triggers_2 + triggers_3 + triggers_4
    assert "stable label shift" in triggers_5
    assert "stable label shift" not in triggers_6


def test_strong_event_lowers_threshold():
    manager = _manager(
        leader_required_reviews=3,
        strong_event_required_reviews=1,
        switch_absolute_advantage=0.10,
        switch_relative_advantage=10.0,
        strong_event_threshold_multiplier=0.5,
    )
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    allowed, detail = manager._switch_allowed_with_reason(
        _observation(10),
        _score(_plan("candidate", Tactic.FOCUS_FIRE, target="red_1"), 0.06),
        _score(manager.current_plan, 0.0),
        ["stable label shift"],
    )
    assert allowed
    assert detail["required_streak"] == 1
    assert detail["threshold_multiplier"] == 0.5


def test_normal_steps_do_not_switch_every_step():
    manager = _manager()
    obs = _observation(0)
    manager.current_plan = _plan("current", Tactic.DISENGAGE)
    manager._mark_plan_start(obs, _belief("UNKNOWN"))
    manager.last_known_enemy_ids = {"red_1"}
    manager.last_active_own_ids = {"blue_1", "blue_2"}
    manager.last_replan_step = 0
    manager.plan_valid_until = 20
    decision_1 = manager.decide(
        _observation(1),
        _memory(),
        SimpleNamespace(tracks=()),
        _belief("UNKNOWN"),
        [_plan("rule_candidate", Tactic.FOCUS_FIRE, target="red_1")],
        {},
        LLMGenerationResult(None, LLMStatus.IDLE, None),
        _score(manager.current_plan, 0.0),
        _FakeScorer([]),
        _NoLLM(),
    )
    decision_2 = manager.decide(
        _observation(2),
        _memory(),
        SimpleNamespace(tracks=()),
        _belief("UNKNOWN"),
        [_plan("rule_candidate", Tactic.FOCUS_FIRE, target="red_1")],
        {},
        LLMGenerationResult(None, LLMStatus.IDLE, None),
        _score(manager.current_plan, 0.0),
        _FakeScorer([]),
        _NoLLM(),
    )
    assert decision_1.action == ManagerAction.CONTINUE
    assert decision_2.action == ManagerAction.CONTINUE


class _FakeScorer:
    def __init__(self, scored):
        self.scored = tuple(scored)
        self.calls = 0

    def score_candidates(self, observation, memory_snapshot, situation, belief, candidates, current_plan):
        self.calls += 1
        scores = []
        by_id = {score.plan.plan_id: score for score in self.scored}
        for candidate in candidates:
            scores.append(by_id.get(candidate.plan_id, _score(candidate, 0.0)))
        ranked = tuple(sorted(scores, key=lambda score: (-score.final_score, score.plan.plan_id)))
        return ScoringResult(tuple(scores), ranked, ranked[0] if ranked else None, 0.0)

    def score_current_plan(self, observation, memory_snapshot, situation, belief, current_plan):
        return _score(current_plan, -1.0)


class _NoLLM:
    def has_inflight_request(self):
        return False

    def submit(self, summary, request_step):
        return None

    def discard_ready_result(self, *args, **kwargs):
        return None

    def consume_ready_result(self):
        return None


def _manager(**overrides):
    params = json.loads(Path("examples/rules/strategy_manager_params.json").read_text(encoding="utf-8"))
    params["candidate_policy"] = "RULE_PRIMARY"
    params.update(overrides)
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    with handle:
        json.dump(params, handle)
    return StrategyManager(params_path=handle.name)


def _plan(plan_id, tactic, target=None):
    roles = {"blue_1": Role.PRESSER, "blue_2": Role.SUPPORTER}
    if tactic == Tactic.DISENGAGE:
        roles = {"blue_1": Role.DEFENDER, "blue_2": Role.DEFENDER}
    targets = {"blue_1": target, "blue_2": target}
    return TeamPlan(
        plan_id=plan_id,
        created_step=0,
        created_sim_time=0.0,
        mode=StrategyMode.PEER,
        tactic=tactic,
        roles=roles,
        target_assignments=targets,
        primary_target=target,
        valid_for_steps=5,
        source=PlanSource.RULE,
        rationale=[],
        metadata={},
    )


def _score(plan, value):
    return ScoredPlan(
        plan=plan,
        valid=True,
        invalid_reasons=(),
        final_score=value,
        expected_utility=value,
        worst_case_utility=value - 0.01,
        switch_cost=0.0,
        hypothesis_scores=(),
        summary="test",
    )


def _observation(step):
    return SimpleNamespace(
        step_index=step,
        sim_time=float(step),
        side="blue",
        controlled_platform_ids=("blue_1", "blue_2"),
        own_units=(
            SimpleNamespace(platform_id="blue_1", weapons=(SimpleNamespace(name="aam_medium", count=2),)),
            SimpleNamespace(platform_id="blue_2", weapons=(SimpleNamespace(name="aam_medium", count=2),)),
        ),
    )


def _memory(lost=False):
    status = "LOST" if lost else OBSERVED
    return SimpleNamespace(
        tracks={"red_1": SimpleNamespace(status=status)},
        visible_target_ids=frozenset({"red_1"} if not lost else ()),
    )


def _situation():
    pair = SimpleNamespace(distance_3d_m=50000.0, closing_speed_mps=120.0, alignment=0.6)
    return SimpleNamespace(
        tracks=(
            SimpleNamespace(ownship_id="blue_1", target_id="red_1", is_observed=True, pair=pair),
            SimpleNamespace(ownship_id="blue_2", target_id="red_2", is_observed=True, pair=pair),
        )
    )


def _belief(label):
    return SimpleNamespace(report_label=label, posterior={"FOCUS_BLUE_1": 1.0})


if __name__ == "__main__":
    main()
