import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.strategy_scorer import (
    PendingShot,
    PredictedAircraftState,
    RolloutFrame,
    StrategyScorer,
    _expected_losses_from_engagement,
    _fire_window_detail,
    _fire_window_estimate,
)


def main():
    test_hot_flank_cold_use_dynamic_ranges()
    test_large_heading_error_lowers_window()
    test_executor_eligible_geometry_scores_high()
    test_future_closing_allows_medium_prediction()
    test_outside_horizon_window_stays_low()
    test_pending_and_terminal_use_dynamic_window()


def test_hot_flank_cold_use_dynamic_ranges():
    params = StrategyScorer()._params
    attacker = _state("blue", 0, 0, 0, 300)
    hot = _state("red_hot", 140000, 0, 180, 250)
    flank = _state("red_flank", 120000, 0, 90, 250)
    cold = _state("red_cold", 90000, 0, 0, 250)
    hot_detail = _fire_window_detail(attacker, hot, True, True, False, params)
    flank_detail = _fire_window_detail(attacker, flank, True, True, False, params)
    cold_detail = _fire_window_detail(attacker, cold, True, True, False, params)
    assert hot_detail["aspect_class"] == "HOT"
    assert flank_detail["aspect_class"] == "FLANK"
    assert cold_detail["aspect_class"] == "COLD"
    assert hot_detail["dynamic_launch_range_m"] == 150000.0
    assert flank_detail["dynamic_launch_range_m"] == 115000.0
    assert cold_detail["dynamic_launch_range_m"] == 85000.0
    assert hot_detail["probability"] > flank_detail["probability"] >= cold_detail["probability"]


def test_large_heading_error_lowers_window():
    params = StrategyScorer()._params
    aligned = _state("blue", 0, 0, 0, 300)
    away = _state("blue", 0, 0, 180, 300)
    target = _state("red", 100000, 0, 180, 250)
    aligned_p, _ = _fire_window_estimate(aligned, target, True, True, False, params)
    away_p, _ = _fire_window_estimate(away, target, True, True, False, params)
    assert aligned_p > 0.70
    assert away_p < aligned_p * 0.55


def test_executor_eligible_geometry_scores_high():
    params = StrategyScorer()._params
    attacker = _state("blue", 0, 0, 0, 300)
    target = _state("red", 100000, 0, 180, 250)
    detail = _fire_window_detail(attacker, target, True, True, False, params)
    assert detail["within_launch_envelope"]
    assert detail["probability"] > 0.70


def test_future_closing_allows_medium_prediction():
    params = StrategyScorer()._params
    attacker = _state("blue", 0, 0, 0, 330)
    target = _state("red", 170000, 0, 180, 300)
    detail = _fire_window_detail(attacker, target, True, True, False, params)
    assert not detail["within_launch_envelope"]
    assert 0.20 <= detail["probability"] <= 0.70


def test_outside_horizon_window_stays_low():
    params = StrategyScorer()._params
    attacker = _state("blue", 0, 0, 0, 280)
    target = _state("red", 260000, 0, 180, 260)
    probability, time_to_fire = _fire_window_estimate(attacker, target, True, True, False, params)
    assert probability == 0.0
    assert time_to_fire == float("inf")


def test_pending_and_terminal_use_dynamic_window():
    params = StrategyScorer()._params
    attacker = _state("blue", 0, 0, 0, 300)
    target = _state("red", 100000, 0, 180, 250)
    probability, _ = _fire_window_estimate(attacker, target, True, True, False, params)
    rollout = type(
        "Rollout",
        (),
        {
            "frames": (RolloutFrame(0.0, (attacker,), (target,)), RolloutFrame(20.0, (attacker,), (target,))),
            "pending_shots": (PendingShot("blue", "red", "own", 0.0, 1.0, probability, True),),
            "threat_chains": (),
        },
    )()
    _, enemy_loss = _expected_losses_from_engagement(rollout, params)
    assert enemy_loss > 0.45


def _state(aircraft_id, north_m, east_m, heading_deg, speed_mps):
    return PredictedAircraftState(
        aircraft_id=aircraft_id,
        north_m=float(north_m),
        east_m=float(east_m),
        altitude_m=8000.0,
        speed_mps=float(speed_mps),
        heading_deg=float(heading_deg),
        alive=True,
        observed=True,
        target_id=None,
    )


if __name__ == "__main__":
    main()
