import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.strategy import PlanSource, Role, StrategyMode, Tactic, TeamPlan
from examples.rules.strategy_scorer import (
    HypothesisRollout,
    PendingShot,
    PredictedAircraftState,
    RolloutFrame,
    StrategyScorer,
    _expected_losses_from_pending,
    _fire_window_estimate,
)
from examples.rules.team_memory import OBSERVED


def main():
    test_unpressed_high_threat_penalizes_focus_fire()
    test_far_second_enemy_allows_focus_fire()
    test_separate_attack_beats_focus_under_dual_threat()
    test_pending_shot_survives_shooter_loss_for_exchange()
    test_duplicate_attack_waste_reduces_margin()
    test_terminal_ordering()
    test_fire_window_direction_matches_executor_conditions()
    test_fixed_tactic_score_comparison()


def test_unpressed_high_threat_penalizes_focus_fire():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    separate = _plan("separate", Tactic.SEPARATE_ATTACK, {"blue_1": "red_1", "blue_2": "red_2"})
    focus_score = scorer.score_candidates(obs, memory, situation, belief, [focus], focus).scored_plans[0]
    separate_score = scorer.score_candidates(obs, memory, situation, belief, [separate], focus).scored_plans[0]
    focus_unpressed = focus_score.hypothesis_scores[0].breakdown.unpressed_enemy_risk
    separate_unpressed = separate_score.hypothesis_scores[0].breakdown.unpressed_enemy_risk
    assert focus_unpressed > separate_unpressed + 0.15


def test_far_second_enemy_allows_focus_fire():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world(red2_far=True)
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    score = scorer.score_candidates(obs, memory, situation, belief, [focus], focus).scored_plans[0]
    assert score.hypothesis_scores[0].breakdown.unpressed_enemy_risk < 0.45


def test_separate_attack_beats_focus_under_dual_threat():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    separate = _plan("separate", Tactic.SEPARATE_ATTACK, {"blue_1": "red_1", "blue_2": "red_2"})
    result = scorer.score_candidates(obs, memory, situation, belief, [focus, separate], focus)
    scores = {score.plan.plan_id: score.final_score for score in result.scored_plans}
    assert scores["separate"] > scores["focus"]


def test_pending_shot_survives_shooter_loss_for_exchange():
    frame = RolloutFrame(
        10.0,
        own_states=(),
        enemy_states=(PredictedAircraftState("red_1", 0, 0, 8000, 250, 0, True, True, None),),
    )
    rollout = HypothesisRollout(
        "FOCUS_BLUE_1",
        (RolloutFrame(0.0, (PredictedAircraftState("blue_1", 0, 0, 8000, 250, 0, True, True, None),), frame.enemy_states), frame),
        True,
        (),
        (PendingShot("blue_1", "red_1", "own", 0.0, 1.0, 0.8, True),),
    )
    own_loss, enemy_loss = _expected_losses_from_pending(rollout, StrategyScorer()._params)
    assert own_loss == 0.0
    assert enemy_loss > 0.7


def test_duplicate_attack_waste_reduces_margin():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    separate = _plan("separate", Tactic.SEPARATE_ATTACK, {"blue_1": "red_1", "blue_2": "red_2"})
    result = scorer.score_candidates(obs, memory, situation, belief, [focus, separate], focus)
    breakdowns = {score.plan.plan_id: score.hypothesis_scores[0].breakdown for score in result.scored_plans}
    assert breakdowns["focus"].duplicate_attack_waste > breakdowns["separate"].duplicate_attack_waste


def test_terminal_ordering():
    params = StrategyScorer()._params
    def rollout(own_prob, enemy_prob):
        return HypothesisRollout(
            "FOCUS_BLUE_1",
            (
                RolloutFrame(0.0, _states("blue", 2), _states("red", 2)),
                RolloutFrame(10.0, _states("blue", 2), _states("red", 2)),
            ),
            True,
            (),
            (
                PendingShot("blue_1", "red_1", "own", 0.0, 1.0, enemy_prob, True),
                PendingShot("red_1", "blue_1", "enemy", 0.0, 1.0, own_prob, True),
            ),
        )
    from examples.rules.strategy_scorer import _terminal_survival
    two_one = _terminal_survival(rollout(0.0, 0.9), params)
    one_one = _terminal_survival(rollout(0.9, 0.9), params)
    one_two = _terminal_survival(rollout(0.9, 0.0), params)
    assert two_one > one_one > one_two


def test_fire_window_direction_matches_executor_conditions():
    params = StrategyScorer()._params
    attacker = PredictedAircraftState("blue_1", 0, 0, 8000, 300, 0, True, True, None)
    close_target = PredictedAircraftState("red_1", 25000, 0, 8000, 250, 180, True, True, None)
    far_target = PredictedAircraftState("red_2", 90000, 0, 8000, 250, 180, True, True, None)
    close_p, _ = _fire_window_estimate(attacker, close_target, True, True, False, params)
    far_p, _ = _fire_window_estimate(attacker, far_target, True, True, False, params)
    no_ammo_p, _ = _fire_window_estimate(attacker, close_target, False, True, False, params)
    coasting_p, _ = _fire_window_estimate(attacker, close_target, True, False, True, params)
    assert close_p > far_p
    assert no_ammo_p == 0.0
    assert 0.0 < coasting_p < close_p


def test_fixed_tactic_score_comparison():
    scorer = StrategyScorer()
    obs, memory, situation, belief = _world()
    focus = _plan("focus", Tactic.FOCUS_FIRE, {"blue_1": "red_1", "blue_2": "red_1"})
    separate = _plan("separate", Tactic.SEPARATE_ATTACK, {"blue_1": "red_1", "blue_2": "red_2"})
    mutual = _plan("mutual", Tactic.MUTUAL_SUPPORT, {"blue_1": "red_1", "blue_2": "red_2"}, {"blue_1": Role.PRESSER, "blue_2": Role.SUPPORTER})
    result = scorer.score_candidates(obs, memory, situation, belief, [focus, separate, mutual], focus)
    scores = {score.plan.plan_id: score.final_score for score in result.scored_plans}
    assert len(set(round(value, 4) for value in scores.values())) >= 2


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
    belief = SimpleNamespace(posterior={"FOCUS_BLUE_1": 1.0})
    return obs, memory, situation, belief


def _plan(plan_id, tactic, assignments, roles=None):
    roles = roles or {"blue_1": Role.PRESSER, "blue_2": Role.PRESSER}
    targets = dict(assignments)
    primary = next((target for target in targets.values() if target is not None), None)
    return TeamPlan(
        plan_id=plan_id,
        created_step=1,
        created_sim_time=1.0,
        mode=StrategyMode.PEER,
        tactic=tactic,
        roles=roles,
        target_assignments=targets,
        primary_target=primary if tactic != Tactic.SEPARATE_ATTACK else None,
        valid_for_steps=3,
        source=PlanSource.RULE,
        rationale=[],
        metadata={"bracket_sides": {"blue_1": -1, "blue_2": 1}} if tactic == Tactic.BRACKET else {},
    )


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


def _states(prefix, count):
    return tuple(PredictedAircraftState(f"{prefix}_{i}", i * 1000, 0, 8000, 250, 0, True, True, None) for i in range(count))


def _pos(latitude, longitude, altitude_m):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


if __name__ == "__main__":
    main()
