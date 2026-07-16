from dataclasses import dataclass, field
from itertools import permutations

from air_combat_challenge.competition.models import ActionBatchV1

from .behavior_tree import Action, Condition, Selector, Sequence
from .geometry import bearing, clamp, difference, distance, normalize


CRANK_ANGLE_DEG = 47.0
CRANK_DURATION_S = 58.0
HOT_RANGE_M = 150000.0
FLANK_RANGE_M = 115000.0
COLD_RANGE_M = 85000.0
DRAG_DURATION_S = 72.0
RECORD_DURATION_S = 85.0


@dataclass
class Leg:
    mode: str
    heading: float
    altitude: float
    until: float


@dataclass
class State:
    legs: dict = field(default_factory=dict)
    shots: list = field(default_factory=list)
    threats: list = field(default_factory=list)
    assignments: dict = field(default_factory=dict)
    initialized: set = field(default_factory=set)


@dataclass
class Context:
    observation: object
    ownship: object
    target: object
    slot: int
    actions: list


class CrankCounterTree:
    def __init__(self):
        self.state = State()
        self.tree = Selector(
            Sequence(Condition(self._unsafe), Action(self._recover)),
            Sequence(Condition(self._threatened), Action(self._counter_or_drag)),
            Sequence(Condition(self._dragging), Action(self._fly_leg)),
            Sequence(Condition(self._cranking), Action(self._fly_leg)),
            Sequence(Condition(self._empty), Action(self._disengage)),
            Sequence(Condition(self._has_target), Action(self._recommit)),
            Action(self._search),
        )

    def reset(self):
        self.state = State()

    def act(self, observation):
        ownships = sorted(observation.own_units, key=lambda unit: unit.platform_id)
        enemies = sorted(
            (
                track
                for track in observation.tracks
                if track.target_side != observation.side
                and "MISSILE" not in track.model.upper()
                and "_AAM_" not in track.target_id.upper()
            ),
            key=lambda track: track.target_id,
        )
        own_ids = {unit.platform_id for unit in ownships}
        self._events(observation, own_ids)
        self._prune(observation.sim_time, own_ids)
        targets = self._assign(ownships, enemies)
        actions = []
        for slot, ownship in enumerate(ownships):
            target = targets.get(ownship.platform_id)
            if target is not None and ownship.platform_id not in self.state.initialized:
                target_bearing = bearing(ownship.position, target.position)
                self.state.legs[ownship.platform_id] = Leg(
                    "crank",
                    normalize(target_bearing + (CRANK_ANGLE_DEG if slot == 0 else -CRANK_ANGLE_DEG)),
                    9800.0 if slot == 0 else 6800.0,
                    observation.sim_time + CRANK_DURATION_S,
                )
                self.state.initialized.add(ownship.platform_id)
            self.tree.tick(Context(observation, ownship, target, slot, actions))
        return ActionBatchV1.model_validate({"actions": actions})

    def _events(self, observation, own_ids):
        for event in observation.events:
            if event.event_type == "WeaponFired":
                if event.shooter in own_ids and event.target:
                    self._add_unique(self.state.shots, event.shooter, event.target, observation.sim_time)
                elif event.target in own_ids and event.shooter:
                    self._add_unique(self.state.threats, event.shooter, event.target, observation.sim_time)
            elif event.event_type in {"WeaponHit", "WeaponMissed"}:
                pair = (event.shooter, event.target)
                self.state.shots = [item for item in self.state.shots if item[:2] != pair]
                self.state.threats = [item for item in self.state.threats if item[:2] != pair]

    @staticmethod
    def _add_unique(records, shooter, target, now):
        if not any(item[:2] == (shooter, target) for item in records):
            records.append((shooter, target, now + RECORD_DURATION_S))

    def _prune(self, now, own_ids):
        self.state.shots = [item for item in self.state.shots if now <= item[2]]
        self.state.threats = [item for item in self.state.threats if item[1] in own_ids and now <= item[2]]
        self.state.legs = {
            platform_id: leg
            for platform_id, leg in self.state.legs.items()
            if platform_id in own_ids and now <= leg.until
        }

    def _assign(self, ownships, enemies):
        if not ownships or not enemies:
            self.state.assignments = {}
            return {}
        old = dict(self.state.assignments)

        def cost(unit, target):
            value = distance(unit.position, target.position) / 1000.0
            if old.get(unit.platform_id) == target.target_id:
                value -= 12.0
            return value

        if len(enemies) >= len(ownships):
            selected = min(
                permutations(enemies, len(ownships)),
                key=lambda choice: sum(cost(unit, target) for unit, target in zip(ownships, choice)),
            )
            ids = {unit.platform_id: target.target_id for unit, target in zip(ownships, selected)}
        else:
            ids = {unit.platform_id: min(enemies, key=lambda target: cost(unit, target)).target_id for unit in ownships}
        self.state.assignments = ids
        lookup = {target.target_id: target for target in enemies}
        return {platform_id: lookup[target_id] for platform_id, target_id in ids.items()}

    @staticmethod
    def _unsafe(context):
        return context.ownship.position.altitude_m < 2100.0

    def _threatened(self, context):
        return any(item[1] == context.ownship.platform_id for item in self.state.threats)

    def _dragging(self, context):
        leg = self.state.legs.get(context.ownship.platform_id)
        return leg is not None and leg.mode == "drag"

    def _cranking(self, context):
        leg = self.state.legs.get(context.ownship.platform_id)
        return leg is not None and leg.mode == "crank"

    @staticmethod
    def _empty(context):
        return not any(item.enabled and item.count > 0 and item.name.startswith("aam_") for item in context.ownship.weapons)

    @staticmethod
    def _has_target(context):
        return context.target is not None

    def _recover(self, context):
        context.actions.append(self._flight(context, context.ownship.attitude.heading_deg, 8200.0, 0.82))

    def _counter_or_drag(self, context):
        fired = self._try_fire(context, 36.0)
        leg = self.state.legs.get(context.ownship.platform_id) if fired else None
        if leg is None or leg.mode != "drag":
            threat = next(item for item in self.state.threats if item[1] == context.ownship.platform_id)
            shooter = next((track for track in context.observation.tracks if track.target_id == threat[0]), None)
            heading = (
                bearing(context.ownship.position, shooter.position) + 180.0
                if shooter is not None
                else context.ownship.attitude.heading_deg + 180.0
            )
            leg = self._start_drag(context, heading)
        context.actions.append(self._flight(context, leg.heading, leg.altitude, 1.4))

    def _fly_leg(self, context):
        leg = self.state.legs[context.ownship.platform_id]
        context.actions.append(self._flight(context, leg.heading, leg.altitude, 1.18 if leg.mode == "crank" else 1.4))

    def _recommit(self, context):
        if self._try_fire(context, 28.0):
            self._fly_leg(context)
            return
        target_bearing = bearing(context.ownship.position, context.target.position)
        altitude = clamp(context.target.position.altitude_m + (900.0 if context.slot == 0 else -900.0), 5200.0, 10800.0)
        context.actions.append(self._flight(context, target_bearing, altitude, 1.2))

    def _try_fire(self, context, max_error):
        if context.target is None:
            return False
        platform_id = context.ownship.platform_id
        target_id = context.target.target_id
        if any(item[0] == platform_id or item[1] == target_id for item in self.state.shots):
            return False
        target_bearing = bearing(context.ownship.position, context.target.position)
        if abs(difference(target_bearing, context.ownship.attitude.heading_deg)) > max_error:
            return False
        target_to_shooter = bearing(context.target.position, context.ownship.position)
        aspect = abs(difference(target_to_shooter, context.target.attitude.heading_deg))
        launch_range = (
            HOT_RANGE_M
            if aspect <= 60.0
            else FLANK_RANGE_M
            if aspect <= 120.0
            else COLD_RANGE_M
        )
        if distance(context.ownship.position, context.target.position) > launch_range:
            return False
        weapon = next((item for item in context.ownship.weapons if item.name == "aam_medium" and item.enabled and item.count > 0), None)
        if weapon is None or context.ownship.platform_id not in context.target.detected_by:
            return False
        context.actions.append({"type": "fire", "platform_id": platform_id, "weapon_name": "aam_medium", "target_id": target_id})
        self.state.shots.append((platform_id, target_id, context.observation.sim_time + RECORD_DURATION_S))
        self._start_drag(context, target_bearing + 180.0)
        return True

    def _start_drag(self, context, heading):
        leg = Leg(
            "drag",
            normalize(heading),
            clamp(context.ownship.position.altitude_m + (700.0 if context.slot == 0 else -700.0), 5000.0, 10500.0),
            context.observation.sim_time + DRAG_DURATION_S,
        )
        self.state.legs[context.ownship.platform_id] = leg
        return leg

    def _disengage(self, context):
        heading = bearing(context.ownship.position, context.target.position) + 180.0 if context.target else context.ownship.attitude.heading_deg
        context.actions.append(self._flight(context, heading, 8000.0, 1.3))

    def _search(self, context):
        heading = normalize(context.ownship.attitude.heading_deg + (15.0 if context.slot == 0 else -15.0))
        context.actions.append(self._flight(context, heading, 9000.0 if context.slot == 0 else 7000.0, 0.95))

    @staticmethod
    def _flight(context, heading, altitude, mach):
        return {"type": "set_flight", "platform_id": context.ownship.platform_id, "heading_deg": normalize(heading), "altitude_m": altitude, "mach": mach}
