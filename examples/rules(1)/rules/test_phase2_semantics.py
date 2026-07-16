import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.executor import Executor
from examples.rules.strategy import Role, RuleCandidateGenerator, Tactic, validate_plan
from examples.rules.team_memory import LOST, OBSERVED


def main():
    test_mutual_support_roles_and_support_target()
    test_mutual_support_roles_can_swap()
    test_separate_attack_assigns_different_targets()
    test_bracket_uses_opposite_lateral_offsets()
    test_defend_counter_is_one_defender_one_presser()
    test_tactics_produce_distinct_guidance()
    test_degraded_cases_are_safe()


def test_mutual_support_roles_and_support_target():
    observation, memory, situation = _world()
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    plan = _plan(plans, Tactic.MUTUAL_SUPPORT)
    roles = set(plan.roles.values())
    assert roles == {Role.PRESSER, Role.SUPPORTER}
    supporter = _platform_with_role(plan, Role.SUPPORTER)
    assert plan.target_assignments[supporter] != plan.primary_target


def test_mutual_support_roles_can_swap():
    observation_a, memory_a, situation_a = _world()
    plan_a = _plan(RuleCandidateGenerator().generate(observation_a, memory_a, situation_a, None), Tactic.MUTUAL_SUPPORT)
    observation, memory, situation = _world(blue_2_better=True)
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    plan = _plan(plans, Tactic.MUTUAL_SUPPORT)
    assert _platform_with_role(plan, Role.PRESSER) != _platform_with_role(plan_a, Role.PRESSER)
    assert _platform_with_role(plan, Role.SUPPORTER) != _platform_with_role(plan_a, Role.SUPPORTER)


def test_separate_attack_assigns_different_targets():
    observation, memory, situation = _world()
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    plan = _plan(plans, Tactic.SEPARATE_ATTACK)
    targets = [target for target in plan.target_assignments.values() if target is not None]
    assert len(targets) == 2
    assert len(set(targets)) == 2
    assert validate_plan(plan, observation, memory, situation).valid


def test_bracket_uses_opposite_lateral_offsets():
    observation, memory, situation = _world()
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    plan = _plan(plans, Tactic.BRACKET)
    guidance = Executor().preview_flight_guidance(observation, memory, situation, plan)
    sides = plan.metadata["bracket_sides"]
    assert set(sides.values()) == {-1, 1}
    target = memory.tracks[plan.primary_target]
    signed_offsets = []
    for unit in observation.own_units:
        bearing = _bearing_deg(unit.position, target.position)
        signed_offsets.append(_signed_angle(guidance[unit.platform_id].heading_deg - bearing))
    assert signed_offsets[0] * signed_offsets[1] < 0.0


def test_defend_counter_is_one_defender_one_presser():
    observation, memory, situation = _world()
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    plan = _plan(plans, Tactic.DEFEND_COUNTER)
    assert set(plan.roles.values()) == {Role.DEFENDER, Role.PRESSER}
    defender = _platform_with_role(plan, Role.DEFENDER)
    presser = _platform_with_role(plan, Role.PRESSER)
    assert plan.target_assignments[defender] is not None
    assert plan.target_assignments[presser] == plan.target_assignments[defender]
    guidance = Executor().preview_flight_guidance(observation, memory, situation, plan)
    assert guidance[defender].mach != guidance[presser].mach


def test_tactics_produce_distinct_guidance():
    observation, memory, situation = _world()
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    executor = Executor()
    signatures = {}
    for tactic in (Tactic.FOCUS_FIRE, Tactic.SEPARATE_ATTACK, Tactic.BRACKET, Tactic.MUTUAL_SUPPORT):
        plan = _plan(plans, tactic)
        guidance = executor.preview_flight_guidance(observation, memory, situation, plan)
        signatures[tactic] = tuple(
            sorted(
                (
                    platform_id,
                    plan.roles[platform_id].value,
                    guidance[platform_id].target_id,
                    round(guidance[platform_id].heading_deg, 1),
                    round(guidance[platform_id].mach, 2),
                )
                for platform_id in guidance
            )
        )
    assert len(set(signatures.values())) == len(signatures)


def test_degraded_cases_are_safe():
    observation, memory, situation = _world(target_ids=("red_1",))
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    assert _plan(plans, Tactic.MUTUAL_SUPPORT).target_assignments
    assert not [plan for plan in plans if plan.tactic == Tactic.SEPARATE_ATTACK]

    observation, memory, situation = _world(target_ids=("red_1",), lost_targets={"red_1"})
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    assert len(plans) == 1
    assert plans[0].tactic == Tactic.DISENGAGE

    observation, memory, situation = _world(own_ids=("blue_1",))
    plans = RuleCandidateGenerator().generate(observation, memory, situation, None)
    assert plans
    assert all(validate_plan(plan, observation, memory, situation).valid for plan in plans)


def _world(own_ids=("blue_1", "blue_2"), target_ids=("red_1", "red_2"), lost_targets=frozenset(), blue_2_better=False):
    own_positions = {
        "blue_1": _pos(0.0, 0.0, 8000.0),
        "blue_2": _pos(0.0, 0.18 if not blue_2_better else 0.02, 9000.0),
    }
    if blue_2_better:
        own_positions["blue_1"] = _pos(0.0, -0.22, 8500.0)
    target_positions = {
        "red_1": _pos(0.45, 0.02, 9000.0),
        "red_2": _pos(0.30, 0.35, 8500.0),
    }
    own_units = tuple(_unit(platform_id, own_positions[platform_id]) for platform_id in own_ids)
    tracks = {
        target_id: _track(target_id, target_positions[target_id], LOST if target_id in lost_targets else OBSERVED)
        for target_id in target_ids
    }
    memory = SimpleNamespace(tracks=tracks, visible_target_ids=frozenset(tid for tid, track in tracks.items() if track.status == OBSERVED))
    situation_tracks = []
    for own_id in own_ids:
        for target_id in target_ids:
            if target_id in lost_targets:
                continue
            situation_tracks.append(_track_situation(own_id, own_positions[own_id], target_id, target_positions[target_id], blue_2_better))
    situation = SimpleNamespace(tracks=tuple(situation_tracks), enemy_centroid=_centroid([track.position for track in tracks.values() if track.status == OBSERVED]))
    observation = SimpleNamespace(
        step_index=1,
        sim_time=1.0,
        side="blue",
        controlled_platform_ids=tuple(own_ids),
        own_units=own_units,
        tracks=(),
    )
    return observation, memory, situation


def _track_situation(own_id, own_pos, target_id, target_pos, blue_2_better):
    distance = _distance_m(own_pos, target_pos)
    closing = 130.0 if (own_id, target_id) in {("blue_1", "red_2"), ("blue_2", "red_1")} else 60.0
    alignment = 0.72 if own_id == "blue_1" and target_id == "red_2" and not blue_2_better else 0.25
    own_alignment = 0.80 if (own_id == "blue_1" and not blue_2_better) or (own_id == "blue_2" and blue_2_better) else 0.35
    if own_id == "blue_2" and target_id == "red_1" and blue_2_better:
        distance *= 0.55
        closing = 150.0
        own_alignment = 0.90
    return SimpleNamespace(
        ownship_id=own_id,
        target_id=target_id,
        is_observed=True,
        pair=SimpleNamespace(
            distance_3d_m=distance,
            horizontal_distance_m=distance,
            altitude_delta_m=target_pos.altitude_m - own_pos.altitude_m,
            closing_speed_mps=closing,
            bearing_deg=_bearing_deg(own_pos, target_pos),
            alignment=alignment,
            own_alignment=own_alignment,
        ),
    )


def _plan(plans, tactic):
    for plan in plans:
        if plan.tactic == tactic:
            return plan
    raise AssertionError(f"missing plan {tactic}")


def _platform_with_role(plan, role):
    for platform_id, value in plan.roles.items():
        if value == role:
            return platform_id
    raise AssertionError(f"missing role {role}")


def _unit(platform_id, position):
    return SimpleNamespace(
        platform_id=platform_id,
        position=position,
        attitude=SimpleNamespace(heading_deg=0.0),
        weapons=(SimpleNamespace(name="aam_medium", count=2, enabled=True),),
    )


def _track(target_id, position, status):
    return SimpleNamespace(
        target_id=target_id,
        target_side="red",
        model="fighter",
        position=position,
        velocity=SimpleNamespace(north_mps=0.0, east_mps=0.0, up_mps=0.0),
        attitude=SimpleNamespace(heading_deg=180.0),
        detected_by=("blue_1",),
        status=status,
    )


def _pos(latitude, longitude, altitude_m):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


def _centroid(positions):
    positions = list(positions)
    if not positions:
        return None
    return _pos(
        sum(position.latitude for position in positions) / len(positions),
        sum(position.longitude for position in positions) / len(positions),
        sum(position.altitude_m for position in positions) / len(positions),
    )


def _distance_m(a, b):
    horizontal = math.hypot((b.latitude - a.latitude) * 111000.0, (b.longitude - a.longitude) * 111000.0)
    return math.hypot(horizontal, b.altitude_m - a.altitude_m)


def _bearing_deg(a, b):
    north = b.latitude - a.latitude
    east = b.longitude - a.longitude
    return (math.degrees(math.atan2(east, north)) + 360.0) % 360.0


def _signed_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


if __name__ == "__main__":
    main()
