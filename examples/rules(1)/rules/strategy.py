from dataclasses import dataclass, field
from enum import Enum

from .team_memory import COASTING, OBSERVED


class StrategyMode(Enum):
    PEER = "PEER"
    DYNAMIC_LEAD_SUPPORT = "DYNAMIC_LEAD_SUPPORT"


class Role(Enum):
    SHOOTER = "SHOOTER"
    SUPPORTER = "SUPPORTER"
    DEFENDER = "DEFENDER"
    PRESSER = "PRESSER"
    TRACK_HOLDER = "TRACK_HOLDER"


class Tactic(Enum):
    FOCUS_FIRE = "FOCUS_FIRE"
    SEPARATE_ATTACK = "SEPARATE_ATTACK"
    BRACKET = "BRACKET"
    HIGH_LOW = "HIGH_LOW"
    MUTUAL_SUPPORT = "MUTUAL_SUPPORT"
    DEFEND_COUNTER = "DEFEND_COUNTER"
    DISENGAGE = "DISENGAGE"


class PlanSource(Enum):
    BASELINE = "BASELINE"
    RULE = "RULE"
    LLM = "LLM"
    FALLBACK = "FALLBACK"


@dataclass(frozen=True)
class TeamPlan:
    plan_id: str
    created_step: int
    created_sim_time: float
    mode: StrategyMode
    tactic: Tactic
    roles: dict
    target_assignments: dict
    primary_target: object = None
    valid_for_steps: int = 1
    source: PlanSource = PlanSource.RULE
    rationale: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    errors: list
    warnings: list


class RuleCandidateGenerator:
    def generate(self, observation, memory_snapshot, situation, belief):
        del belief
        controlled_ids = list(observation.controlled_platform_ids)
        live_ids = [
            unit.platform_id
            for unit in observation.own_units
            if unit.platform_id in controlled_ids
        ]
        known_targets = _known_targets(memory_snapshot)
        if not known_targets:
            return self._validated_unique(
                [
                    self._make_plan(
                        observation,
                        "rule_000_disengage",
                        StrategyMode.PEER,
                        Tactic.DISENGAGE,
                        {platform_id: Role.DEFENDER for platform_id in live_ids},
                        {platform_id: None for platform_id in live_ids},
                        None,
                        ["no observed or coasting enemy tracks are available"],
                        {"candidate_kind": "safe_disengage"},
                    )
                ],
                observation,
                memory_snapshot,
                situation,
            )

        ranked_targets = _rank_targets_for_focus(live_ids, known_targets, situation)
        primary_target = ranked_targets[0] if ranked_targets else known_targets[0]
        shooter_order = _rank_shooters(live_ids, primary_target, situation, observation)
        pressure_order = _rank_defenders(live_ids, situation)
        shooter = shooter_order[0] if shooter_order else (live_ids[0] if live_ids else None)
        defender = pressure_order[0] if pressure_order else (live_ids[0] if live_ids else None)
        supporter = _first_other(live_ids, shooter)
        presser = _first_other(live_ids, defender)
        mutual_roles, mutual_assignments, mutual_metadata = _mutual_support_plan_parts(
            live_ids,
            known_targets,
            primary_target,
            situation,
            observation,
        )
        separate_assignments = _separate_assignments(live_ids, known_targets, situation)
        bracket_metadata = _bracket_metadata(live_ids, situation)
        defend_threat = _highest_threat_target(defender, situation) if defender is not None else primary_target
        defend_target = defend_threat or primary_target
        plans = []

        plans.append(
            self._make_plan(
                observation,
                "rule_000_focus_fire_peer",
                StrategyMode.PEER,
                Tactic.FOCUS_FIRE,
                {platform_id: Role.PRESSER for platform_id in live_ids},
                {platform_id: primary_target for platform_id in live_ids},
                primary_target,
                ["peer focus fire candidate uses the best shared target geometry"],
                {"primary_target_basis": "distance_alignment_weapon"},
            )
        )
        if len(known_targets) >= 2:
            plans.append(
                self._make_plan(
                    observation,
                    "rule_010_separate_attack_peer",
                    StrategyMode.PEER,
                    Tactic.SEPARATE_ATTACK,
                    {platform_id: Role.PRESSER for platform_id in live_ids},
                    separate_assignments,
                    None,
                    ["peer separate attack candidate assigns different known targets"],
                    {"target_count": len(known_targets)},
                )
            )
        plans.extend(
            [
                self._make_plan(
                    observation,
                    "rule_020_bracket_peer",
                    StrategyMode.PEER,
                    Tactic.BRACKET,
                    {platform_id: Role.PRESSER for platform_id in live_ids},
                    {platform_id: primary_target for platform_id in live_ids},
                    primary_target,
                    ["bracket assigns opposite lateral entries around the main target"],
                    bracket_metadata,
                ),
                self._make_plan(
                    observation,
                    "rule_030_high_low_peer",
                    StrategyMode.PEER,
                    Tactic.HIGH_LOW,
                    {platform_id: Role.PRESSER for platform_id in live_ids},
                    {platform_id: primary_target for platform_id in live_ids},
                    primary_target,
                    ["high-low is represented as a high-level formation tactic only"],
                    {},
                ),
                self._make_plan(
                    observation,
                    "rule_040_mutual_support_peer",
                    StrategyMode.PEER,
                    Tactic.MUTUAL_SUPPORT,
                    mutual_roles,
                    mutual_assignments,
                    primary_target,
                    ["mutual support pairs a presser with a covering supporter"],
                    mutual_metadata,
                ),
            ]
        )
        if shooter is not None and supporter is not None:
            plans.append(
                self._make_plan(
                    observation,
                    "rule_100_focus_fire_lead_support",
                    StrategyMode.DYNAMIC_LEAD_SUPPORT,
                    Tactic.FOCUS_FIRE,
                    {shooter: Role.SHOOTER, supporter: Role.SUPPORTER},
                    {platform_id: primary_target for platform_id in live_ids},
                    primary_target,
                    ["best shooter is paired with a supporter on the shared target"],
                    {"shooter": shooter, "supporter": supporter},
                )
            )
        if defender is not None and presser is not None:
            plans.append(
                self._make_plan(
                    observation,
                    "rule_110_defend_counter_lead_support",
                    StrategyMode.DYNAMIC_LEAD_SUPPORT,
                    Tactic.DEFEND_COUNTER,
                    {defender: Role.DEFENDER, presser: Role.PRESSER},
                    _defend_counter_assignments(live_ids, defender, presser, defend_target),
                    defend_target,
                    ["highest pressure ownship defends while the paired aircraft presses"],
                    {"defender": defender, "presser": presser, "threat_target": defend_target},
                )
            )
        if shooter is not None and supporter is not None:
            plans.append(
                self._make_plan(
                    observation,
                    "rule_120_attack_track_hold_lead_support",
                    StrategyMode.DYNAMIC_LEAD_SUPPORT,
                    Tactic.FOCUS_FIRE,
                    {shooter: Role.SHOOTER, supporter: Role.TRACK_HOLDER},
                    {platform_id: primary_target for platform_id in live_ids},
                    primary_target,
                    ["shooter attacks while the paired aircraft preserves track continuity"],
                    {"shooter": shooter, "track_holder": supporter},
                )
            )
        plans.append(
            self._make_plan(
                observation,
                "rule_900_disengage",
                StrategyMode.PEER,
                Tactic.DISENGAGE,
                {platform_id: Role.DEFENDER for platform_id in live_ids},
                {platform_id: None for platform_id in live_ids},
                None,
                ["safe disengage fallback candidate is always available"],
                {"candidate_kind": "safe_disengage"},
            )
        )
        return self._validated_unique(plans, observation, memory_snapshot, situation)

    def _make_plan(
        self,
        observation,
        plan_id,
        mode,
        tactic,
        roles,
        target_assignments,
        primary_target,
        rationale,
        metadata,
    ):
        live_ids = [
            unit.platform_id
            for unit in observation.own_units
            if unit.platform_id in observation.controlled_platform_ids
        ]
        complete_roles = {platform_id: roles.get(platform_id, Role.PRESSER) for platform_id in live_ids}
        complete_targets = {
            platform_id: target_assignments.get(platform_id)
            for platform_id in live_ids
        }
        return TeamPlan(
            plan_id=plan_id,
            created_step=observation.step_index,
            created_sim_time=observation.sim_time,
            mode=mode,
            tactic=tactic,
            roles=complete_roles,
            target_assignments=complete_targets,
            primary_target=primary_target,
            valid_for_steps=3,
            source=PlanSource.RULE,
            rationale=list(rationale),
            metadata=dict(metadata),
        )

    def _validated_unique(self, plans, observation, memory_snapshot, situation):
        unique = []
        seen = set()
        for plan in plans:
            signature = (
                plan.mode.value,
                plan.tactic.value,
                tuple((key, plan.roles[key].value) for key in sorted(plan.roles)),
                tuple(
                    (key, plan.target_assignments[key])
                    for key in sorted(plan.target_assignments)
                ),
                plan.primary_target,
            )
            if signature in seen:
                continue
            seen.add(signature)
            result = validate_plan(plan, observation, memory_snapshot, situation)
            if result.valid:
                unique.append(plan)
        return unique


def validate_plan(plan, observation, memory_snapshot, situation):
    errors = []
    warnings = []
    controlled_ids = list(observation.controlled_platform_ids)
    controlled_set = set(controlled_ids)
    live_ids = [
        unit.platform_id
        for unit in observation.own_units
        if unit.platform_id in controlled_set
    ]
    known_targets = _known_targets(memory_snapshot)
    known_target_set = set(known_targets)

    role_ids = set(plan.roles)
    assignment_ids = set(plan.target_assignments)
    invalid_role_ids = sorted(role_ids - controlled_set)
    invalid_assignment_ids = sorted(assignment_ids - controlled_set)
    if invalid_role_ids:
        errors.append(f"roles reference uncontrolled platform ids: {invalid_role_ids}")
    if invalid_assignment_ids:
        errors.append(
            f"target_assignments reference uncontrolled platform ids: {invalid_assignment_ids}"
        )

    for platform_id in live_ids:
        if platform_id not in plan.roles:
            errors.append(f"missing role for live platform {platform_id}")
        if platform_id not in plan.target_assignments:
            errors.append(f"missing target assignment for live platform {platform_id}")

    for platform_id, target_id in sorted(plan.target_assignments.items()):
        if target_id is None:
            continue
        track = memory_snapshot.tracks.get(target_id)
        if target_id not in known_target_set or track is None:
            errors.append(f"{platform_id} target {target_id} is not observed or coasting")
            continue
        if not _is_aircraft_track(track):
            errors.append(f"{platform_id} target {target_id} is not an aircraft track")
            continue
        if track.status == COASTING:
            warnings.append(
                f"{platform_id} target {target_id} is COASTING and must not be used for fire"
            )
        if getattr(track, "status", None) not in {OBSERVED, COASTING}:
            errors.append(f"{platform_id} target {target_id} is not live")

    if plan.valid_for_steps <= 0:
        errors.append("valid_for_steps must be greater than 0")

    if plan.tactic == Tactic.FOCUS_FIRE and plan.primary_target is None:
        errors.append("FOCUS_FIRE requires primary_target")

    if plan.tactic == Tactic.FOCUS_FIRE and plan.primary_target is not None:
        for platform_id, target_id in sorted(plan.target_assignments.items()):
            if target_id is not None and target_id != plan.primary_target:
                errors.append(
                    f"FOCUS_FIRE assignment for {platform_id} does not match primary_target"
                )

    if plan.tactic == Tactic.SEPARATE_ATTACK and len(known_targets) >= 2:
        non_empty_targets = [
            plan.target_assignments.get(platform_id)
            for platform_id in live_ids
            if plan.target_assignments.get(platform_id) is not None
        ]
        if len(non_empty_targets) >= 2 and len(set(non_empty_targets)) < len(non_empty_targets):
            errors.append("SEPARATE_ATTACK must assign different targets when two are available")

    if plan.tactic == Tactic.MUTUAL_SUPPORT and len(live_ids) >= 2:
        role_values = set(plan.roles.get(platform_id) for platform_id in live_ids)
        if Role.PRESSER not in role_values or Role.SUPPORTER not in role_values:
            errors.append("MUTUAL_SUPPORT requires PRESSER and SUPPORTER roles when two ownships are live")

    if plan.tactic == Tactic.BRACKET and len(live_ids) >= 2:
        sides = {}
        if isinstance(plan.metadata, dict):
            sides = plan.metadata.get("bracket_sides", {}) or {}
        live_sides = [sides.get(platform_id) for platform_id in live_ids]
        if len(live_sides) >= 2 and (live_sides[0] is None or live_sides[1] is None or live_sides[0] == live_sides[1]):
            errors.append("BRACKET requires different lateral side assignments for two ownships")

    if plan.tactic == Tactic.DEFEND_COUNTER:
        role_values = set(plan.roles.values())
        if Role.DEFENDER not in role_values or Role.PRESSER not in role_values:
            errors.append("DEFEND_COUNTER requires both DEFENDER and PRESSER roles")

    if plan.mode == StrategyMode.DYNAMIC_LEAD_SUPPORT:
        role_values = set(plan.roles.values())
        if len(role_values) < 2:
            errors.append("DYNAMIC_LEAD_SUPPORT requires functionally distinct roles")
        allowed_pairs = [
            {Role.SHOOTER, Role.SUPPORTER},
            {Role.DEFENDER, Role.PRESSER},
            {Role.SHOOTER, Role.TRACK_HOLDER},
        ]
        if not any(pair.issubset(role_values) for pair in allowed_pairs):
            errors.append("DYNAMIC_LEAD_SUPPORT role combination is not explicit enough")

    if not known_targets and plan.tactic not in {Tactic.DISENGAGE, Tactic.MUTUAL_SUPPORT}:
        errors.append("plans without known enemy tracks must be DISENGAGE or safe keep")
    if not known_targets and any(plan.target_assignments.values()):
        errors.append("plans without known enemy tracks must not assign targets")

    return PlanValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _known_targets(memory_snapshot):
    return [
        target_id
        for target_id, track in sorted(memory_snapshot.tracks.items())
        if track.status in {OBSERVED, COASTING} and _is_aircraft_track(track)
    ]


def _is_aircraft_track(track):
    model = str(getattr(track, "model", "") or "").lower()
    target_id = str(getattr(track, "target_id", "") or "").lower()
    if "aam" in model or "missile" in model or "weapon" in model:
        return False
    if "_aam_" in target_id or "missile" in target_id:
        return False
    return True


def _rank_targets_for_focus(live_ids, known_targets, situation):
    scores = []
    for target_id in known_targets:
        distances = []
        closing = []
        alignments = []
        for track in situation.tracks:
            if track.target_id != target_id or track.ownship_id not in live_ids:
                continue
            distances.append(track.pair.distance_3d_m)
            closing.append(track.pair.closing_speed_mps)
            alignments.append(track.pair.alignment)
        if not distances:
            scores.append((float("inf"), target_id))
            continue
        score = (
            min(distances)
            - 100.0 * max(closing)
            - 5000.0 * max(alignments)
        )
        scores.append((score, target_id))
    return [target_id for _, target_id in sorted(scores)]


def _rank_shooters(live_ids, target_id, situation, observation):
    weapon_counts = {
        unit.platform_id: _weapon_remaining(unit, "aam_medium")
        for unit in observation.own_units
    }
    scores = []
    for platform_id in live_ids:
        pair = _pair_for(situation, platform_id, target_id)
        if pair is None:
            scores.append((float("inf"), platform_id))
            continue
        weapon_bonus = 100000.0 if weapon_counts.get(platform_id, 0) > 0 else 0.0
        score = (
            pair.pair.distance_3d_m
            - 100.0 * pair.pair.closing_speed_mps
            - 5000.0 * pair.pair.own_alignment
            - weapon_bonus
        )
        scores.append((score, platform_id))
    return [platform_id for _, platform_id in sorted(scores)]


def _rank_defenders(live_ids, situation):
    scores = []
    for platform_id in live_ids:
        pressure = 0.0
        found = False
        for track in situation.tracks:
            if track.ownship_id != platform_id:
                continue
            found = True
            pressure = max(
                pressure,
                max(0.0, track.pair.closing_speed_mps)
                + 200.0 * max(0.0, track.pair.alignment)
                + max(0.0, 80000.0 - track.pair.distance_3d_m) / 1000.0,
            )
        scores.append((-pressure if found else 0.0, platform_id))
    return [platform_id for _, platform_id in sorted(scores)]


def _pair_for(situation, ownship_id, target_id):
    for track in situation.tracks:
        if track.ownship_id == ownship_id and track.target_id == target_id:
            return track
    return None


def _first_other(items, selected):
    for item in items:
        if item != selected:
            return item
    return None


def _separate_assignments(live_ids, known_targets, situation):
    if not live_ids:
        return {}
    if len(known_targets) < 2:
        return {platform_id: known_targets[0] if known_targets else None for platform_id in live_ids}
    if len(live_ids) < 2:
        return {live_ids[0]: _best_target_for_platform(live_ids[0], known_targets, situation)}
    first, second = live_ids[:2]
    best = None
    for target_a in known_targets:
        for target_b in known_targets:
            if target_a == target_b:
                continue
            score = _assignment_cost(first, target_a, situation) + _assignment_cost(second, target_b, situation)
            candidate = (score, target_a, target_b)
            if best is None or candidate < best:
                best = candidate
    assignments = {first: best[1], second: best[2]} if best else {first: known_targets[0], second: known_targets[1]}
    for index, platform_id in enumerate(live_ids[2:], start=2):
        assignments[platform_id] = known_targets[index % len(known_targets)]
    return assignments


def _defend_counter_assignments(live_ids, defender, presser, primary_target):
    return {
        platform_id: (primary_target if platform_id in {defender, presser} else None)
        for platform_id in live_ids
    }


def _mutual_support_plan_parts(live_ids, known_targets, primary_target, situation, observation):
    if not live_ids:
        return {}, {}, {}
    if len(live_ids) == 1:
        platform_id = live_ids[0]
        return (
            {platform_id: Role.PRESSER},
            {platform_id: primary_target},
            {"presser": platform_id, "supporter": None, "support_target": None},
        )
    presser_order = _rank_mutual_pressers(live_ids, primary_target, situation, observation)
    presser = presser_order[0]
    supporter = _first_other(live_ids, presser)
    support_target = _support_target(supporter, primary_target, known_targets, situation)
    roles = {platform_id: Role.PRESSER for platform_id in live_ids}
    roles[supporter] = Role.SUPPORTER
    assignments = {platform_id: primary_target for platform_id in live_ids}
    assignments[supporter] = support_target or primary_target
    return (
        roles,
        assignments,
        {"presser": presser, "supporter": supporter, "support_target": assignments[supporter]},
    )


def _rank_mutual_pressers(live_ids, target_id, situation, observation):
    shooter_order = _rank_shooters(live_ids, target_id, situation, observation)
    defender_order = _rank_defenders(live_ids, situation)
    pressure_rank = {platform_id: index for index, platform_id in enumerate(defender_order)}
    scores = []
    for index, platform_id in enumerate(shooter_order):
        threat_penalty = (len(live_ids) - pressure_rank.get(platform_id, len(live_ids))) * 1000.0
        scores.append((index * 100.0 + threat_penalty, platform_id))
    return [platform_id for _, platform_id in sorted(scores)]


def _support_target(supporter, primary_target, known_targets, situation):
    alternatives = [target_id for target_id in known_targets if target_id != primary_target]
    if alternatives:
        return _best_target_for_platform(supporter, alternatives, situation)
    return primary_target


def _best_target_for_platform(platform_id, known_targets, situation):
    return min(known_targets, key=lambda target_id: _assignment_cost(platform_id, target_id, situation))


def _assignment_cost(platform_id, target_id, situation):
    pair = _pair_for(situation, platform_id, target_id)
    if pair is None:
        return 1e12
    return (
        pair.pair.distance_3d_m
        - 120.0 * pair.pair.closing_speed_mps
        - 7000.0 * pair.pair.own_alignment
        - 3000.0 * pair.pair.alignment
    )


def _highest_threat_target(platform_id, situation):
    if platform_id is None:
        return None
    best = None
    for track in situation.tracks:
        if track.ownship_id != platform_id:
            continue
        pressure = (
            max(0.0, track.pair.closing_speed_mps)
            + 250.0 * max(0.0, track.pair.alignment)
            + max(0.0, 90000.0 - track.pair.distance_3d_m) / 800.0
        )
        candidate = (-pressure, track.target_id)
        if best is None or candidate < best:
            best = candidate
    return best[1] if best else None


def _bracket_metadata(live_ids, situation):
    if len(live_ids) < 2:
        return {"bracket_sides": {platform_id: -1 for platform_id in live_ids}}
    ordered = sorted(live_ids, key=lambda platform_id: _average_bearing(platform_id, situation))
    return {
        "bracket_sides": {
            ordered[0]: -1,
            ordered[1]: 1,
            **{platform_id: (-1 if index % 2 == 0 else 1) for index, platform_id in enumerate(ordered[2:], start=2)},
        }
    }


def _average_bearing(platform_id, situation):
    bearings = [track.pair.bearing_deg for track in situation.tracks if track.ownship_id == platform_id]
    if not bearings:
        return 0.0
    return sum(bearings) / len(bearings)


def _weapon_remaining(unit, weapon_name):
    for weapon in unit.weapons:
        if weapon.name == weapon_name:
            return int(weapon.count)
    return 0
