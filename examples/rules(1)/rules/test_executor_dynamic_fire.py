import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.config import ENGAGEMENT_PENDING_SHOT_TIMEOUT_S
from examples.rules.executor import Executor
from examples.rules.strategy import PlanSource, Role, StrategyMode, Tactic, TeamPlan
from examples.rules.team_memory import COASTING, OBSERVED


def main():
    test_hot_aligned_can_fire()
    test_flank_outside_dynamic_range()
    test_cold_outside_dynamic_range()
    test_heading_error_blocks_fire()
    test_non_observed_blocks_fire()
    test_direct_fire_requires_self_detection()
    test_weapon_and_cooldown_blocks_fire()
    test_pending_limits_block_and_clear()
    test_one_weapon_action_per_platform()


def test_hot_aligned_can_fire():
    executor = Executor()
    observation, memory, ownship, target = _case(distance_lon_deg=1.0, target_heading=270.0, own_heading=90.0)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert eligibility.eligible, eligibility.ineligible_reasons
    assert executor.build_weapon_action(observation, memory, ownship, target, _plan(target.target_id))["type"] == "fire"


def test_flank_outside_dynamic_range():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=1.2, target_heading=0.0, own_heading=90.0)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert not eligibility.eligible
    assert "outside_dynamic_range" in eligibility.ineligible_reasons


def test_cold_outside_dynamic_range():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=1.0, target_heading=90.0, own_heading=90.0)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert not eligibility.eligible
    assert "outside_dynamic_range" in eligibility.ineligible_reasons


def test_heading_error_blocks_fire():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=1.0, target_heading=270.0, own_heading=270.0)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert not eligibility.eligible
    assert "heading_error" in eligibility.ineligible_reasons


def test_non_observed_blocks_fire():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=0.5, target_heading=270.0, own_heading=90.0, status=COASTING)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert not eligibility.eligible
    assert "target_not_observed" in eligibility.ineligible_reasons


def test_direct_fire_requires_self_detection():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=0.5, target_heading=270.0, own_heading=90.0, detected_by=("blue_2",))
    direct = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0, require_direct_detection=True)
    co_fire_geometry = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0, require_direct_detection=False)
    assert not direct.eligible
    assert "not_detected_by_self" in direct.ineligible_reasons
    assert co_fire_geometry.eligible


def test_weapon_and_cooldown_blocks_fire():
    executor = Executor()
    observation, _, ownship, target = _case(distance_lon_deg=0.5, target_heading=270.0, own_heading=90.0, weapon_count=0)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert "no_aam_medium" in eligibility.ineligible_reasons

    observation, _, ownship, target = _case(distance_lon_deg=0.5, target_heading=270.0, own_heading=90.0)
    executor._last_launch_time[ownship.platform_id] = observation.sim_time - 10.0
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert "cooldown" in eligibility.ineligible_reasons
    assert eligibility.cooldown_remaining_s > 0.0


def test_pending_limits_block_and_clear():
    executor = Executor()
    observation, memory, ownship, target = _case(distance_lon_deg=0.5, target_heading=270.0, own_heading=90.0)
    executor._record_pending_shot(observation, ownship.platform_id, target.target_id)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert "pending_shooter_limit" in eligibility.ineligible_reasons
    assert "pending_target_limit" in eligibility.ineligible_reasons

    observation.sim_time += ENGAGEMENT_PENDING_SHOT_TIMEOUT_S + 1.0
    executor._refresh_pending_shots(observation, memory)
    eligibility = executor.evaluate_fire_eligibility(observation, ownship, target, 28.0)
    assert eligibility.eligible, eligibility.ineligible_reasons


def test_one_weapon_action_per_platform():
    executor = Executor()
    observation, memory, _, _ = _two_ship_case()
    actions = executor.build_actions(observation, memory, SimpleNamespace(tracks=()), _plan("red_1", two_ship=True))
    dumped = actions.model_dump()
    weapon_counts = {}
    for action in dumped["actions"]:
        if action["type"] in {"fire", "co_fire"}:
            weapon_counts[action["platform_id"]] = weapon_counts.get(action["platform_id"], 0) + 1
    assert weapon_counts
    assert all(count == 1 for count in weapon_counts.values())


def _case(
    distance_lon_deg,
    target_heading,
    own_heading,
    status=OBSERVED,
    detected_by=("blue_1",),
    weapon_count=2,
):
    ownship = _ownship("blue_1", _pos(0.0, 0.0), own_heading, weapon_count)
    target = _target("red_1", _pos(0.0, distance_lon_deg), target_heading, status, detected_by)
    observation = SimpleNamespace(
        step_index=1,
        sim_time=100.0,
        controlled_platform_ids=("blue_1",),
        own_units=(ownship,),
    )
    memory = SimpleNamespace(tracks={target.target_id: target}, events_history=())
    return observation, memory, ownship, target


def _two_ship_case():
    blue_1 = _ownship("blue_1", _pos(0.0, 0.0), 90.0, 2)
    blue_2 = _ownship("blue_2", _pos(0.02, 0.0), 90.0, 2)
    red_1 = _target("red_1", _pos(0.0, 0.5), 270.0, OBSERVED, ("blue_1", "blue_2"))
    observation = SimpleNamespace(
        step_index=1,
        sim_time=100.0,
        controlled_platform_ids=("blue_1", "blue_2"),
        own_units=(blue_1, blue_2),
    )
    memory = SimpleNamespace(tracks={red_1.target_id: red_1}, events_history=())
    return observation, memory, blue_1, red_1


def _plan(target_id, two_ship=False):
    ids = ("blue_1", "blue_2") if two_ship else ("blue_1",)
    return TeamPlan(
        plan_id="test_plan",
        created_step=1,
        created_sim_time=100.0,
        mode=StrategyMode.PEER,
        tactic=Tactic.FOCUS_FIRE,
        roles={platform_id: Role.PRESSER for platform_id in ids},
        target_assignments={platform_id: target_id for platform_id in ids},
        primary_target=target_id,
        valid_for_steps=5,
        source=PlanSource.RULE,
        rationale=[],
        metadata={},
    )


def _ownship(platform_id, position, heading, weapon_count):
    return SimpleNamespace(
        platform_id=platform_id,
        position=position,
        attitude=SimpleNamespace(heading_deg=heading),
        weapons=(SimpleNamespace(name="aam_medium", count=weapon_count, enabled=weapon_count > 0),),
    )


def _target(target_id, position, heading, status, detected_by):
    return SimpleNamespace(
        target_id=target_id,
        target_side="red",
        model="fighter",
        position=position,
        velocity=SimpleNamespace(north_mps=0.0, east_mps=0.0, up_mps=0.0),
        attitude=SimpleNamespace(heading_deg=heading),
        detected_by=tuple(detected_by),
        status=status,
    )


def _pos(latitude, longitude, altitude_m=8000.0):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


if __name__ == "__main__":
    main()
