import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.strategy import PlanSource, Role, StrategyMode, Tactic, TeamPlan
from examples.rules.strategy_manager import StrategyManager
from examples.rules.strategy_scorer import (
    HypothesisRollout,
    PredictedAircraftState,
    RolloutFrame,
    StrategyScorer,
    UtilityBreakdown,
    HypothesisScore,
    ScoredPlan,
    _expected_losses_from_engagement,
    _prelaunch_probability,
)
from examples.rules.team_memory import OBSERVED


def main():
    test_focus_gets_prelaunch_loss_under_independent_dual_threat()
    test_mutual_and_separate_reduce_second_enemy_prelaunch_risk()
    test_far_second_enemy_keeps_focus_reasonable()
    test_prelaunch_probability_increases_as_time_to_fire_nears()
    test_manager_risk_strong_event_fires_once_and_cools_down()


def test_focus_gets_prelaunch_loss_under_independent_dual_threat():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    score = scorer.score_candidates(obs, memory, situation, belief, [focus], focus).scored_plans[0]
    diag = score.hypothesis_scores[0].diagnostics
    assert diag["enemy_threat_chain_count"] >= 2
    assert diag["prelaunch_own_expected_losses"] > 0.20
    assert score.hypothesis_scores[0].breakdown.terminal_survival < 0.50


def test_mutual_and_separate_reduce_second_enemy_prelaunch_risk():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    separate = _plan("separate", Tactic.SEPARATE_ATTACK, {"blue_1": "red_1", "blue_2": "red_2"})
    mutual = _plan(
        "mutual",
        Tactic.MUTUAL_SUPPORT,
        {"blue_1": "red_1", "blue_2": "red_2"},
        {"blue_1": Role.PRESSER, "blue_2": Role.SUPPORTER},
    )
    result = scorer.score_candidates(obs, memory, situation, belief, [focus, separate, mutual], focus)
    diagnostics = {
        score.plan.plan_id: score.hypothesis_scores[0].diagnostics
        for score in result.scored_plans
    }
    assert diagnostics["separate"]["prelaunch_own_expected_losses"] < diagnostics["focus"]["prelaunch_own_expected_losses"]
    assert diagnostics["mutual"]["prelaunch_own_expected_losses"] < diagnostics["focus"]["prelaunch_own_expected_losses"]


def test_far_second_enemy_keeps_focus_reasonable():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world(red2_far=True)
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    score = scorer.score_candidates(obs, memory, situation, belief, [focus], focus).scored_plans[0]
    diag = score.hypothesis_scores[0].diagnostics
    assert diag["prelaunch_own_expected_losses"] < 0.35
    assert score.hypothesis_scores[0].breakdown.unpressed_enemy_risk < 0.45


def test_prelaunch_probability_increases_as_time_to_fire_nears():
    distant = _prelaunch_probability(0.8, 20.0, 12.0, 24.0)
    near = _prelaunch_probability(0.8, 2.0, 2.0, 24.0)
    assert near > distant
    assert 0.0 <= distant <= 1.0
    assert 0.0 <= near <= 1.0


def test_manager_risk_strong_event_fires_once_and_cools_down():
    manager = StrategyManager()
    manager.current_plan = _plan("current", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    manager.plan_start_step = 0
    manager.minimum_hold_until = 0
    high = _score_with_breakdown(manager.current_plan, enemy_fire=0.60, unpressed=0.30)
    first = manager._risk_replan_triggers(10, high)
    second = manager._risk_replan_triggers(11, high)
    assert "enemy fire window high" in first
    assert "unpressed enemy high risk" in first
    assert "enemy fire window high" not in second
    assert "unpressed enemy high risk" not in second
    low = _score_with_breakdown(manager.current_plan, enemy_fire=0.10, unpressed=0.05)
    assert manager._risk_replan_triggers(12, low) == []
    blocked_by_cooldown = manager._risk_replan_triggers(20, high)
    assert blocked_by_cooldown == []
    after_cooldown = manager._risk_replan_triggers(31, high)
    assert "enemy fire window high" in after_cooldown


def _world(red2_far=False):
    red2_lat = 2.0 if red2_far else 0.24
    obs = SimpleNamespace(
        step_index=1,
        sim_time=1.0,
        side="blue",
        controlled_platform_ids=("blue_1", "blue_2"),
        own_units=(
            _unit("blue_1", _pos(0.0, -0.05, 8000), 0),
            _unit("blue_2", _pos(0.0, 0.05, 8000), 0),
        ),
    )
    memory = SimpleNamespace(
        tracks={
            "red_1": _track("red_1", _pos(0.25, -0.04, 8200), 180),
            "red_2": _track("red_2", _pos(red2_lat, 0.04, 8200), 180),
        },
        visible_target_ids=frozenset({"red_1", "red_2"}),
    )
    situation = SimpleNamespace(tracks=(), enemy_centroid=None)
    belief = SimpleNamespace(posterior={"SPLIT_ATTACK": 1.0})
    return obs, memory, situation, belief


def _plan(plan_id, tactic, assignments=None, roles=None):
    assignments = assignments or {"blue_1": "red_1", "blue_2": "red_1"}
    roles = roles or {"blue_1": Role.PRESSER, "blue_2": Role.PRESSER}
    primary = next((target for target in assignments.values() if target is not None), None)
    return TeamPlan(
        plan_id=plan_id,
        created_step=1,
        created_sim_time=1.0,
        mode=StrategyMode.PEER,
        tactic=tactic,
        roles=roles,
        target_assignments=dict(assignments),
        primary_target=primary if tactic != Tactic.SEPARATE_ATTACK else None,
        valid_for_steps=3,
        source=PlanSource.RULE,
        rationale=[],
        metadata={"bracket_sides": {"blue_1": -1, "blue_2": 1}} if tactic == Tactic.BRACKET else {},
    )


def _score_with_breakdown(plan, enemy_fire, unpressed):
    breakdown = UtilityBreakdown(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        unpressed,
        enemy_fire,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    hypothesis = HypothesisScore(
        "SPLIT_ATTACK",
        1.0,
        0.0,
        breakdown,
        (),
        {"prelaunch_own_expected_losses": 0.8, "enemy_threat_chain_count": 2},
    )
    return ScoredPlan(plan, True, (), 0.0, 0.0, 0.0, 0.0, (hypothesis,), "test")


def _unit(platform_id, position, heading):
    return SimpleNamespace(
        platform_id=platform_id,
        position=position,
        velocity=SimpleNamespace(north_mps=280.0, east_mps=0.0, up_mps=0.0),
        attitude=SimpleNamespace(heading_deg=heading),
        weapons=(SimpleNamespace(name="aam_medium", count=2, enabled=True),),
    )


def _track(target_id, position, heading):
    return SimpleNamespace(
        target_id=target_id,
        target_side="red",
        model="fighter",
        position=position,
        velocity=SimpleNamespace(north_mps=-280.0, east_mps=0.0, up_mps=0.0),
        attitude=SimpleNamespace(heading_deg=heading),
        detected_by=("blue_1", "blue_2"),
        status=OBSERVED,
    )


def _pos(latitude, longitude, altitude_m):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


if __name__ == "__main__":
    main()
