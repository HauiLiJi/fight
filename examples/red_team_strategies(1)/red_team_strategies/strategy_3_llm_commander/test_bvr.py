from types import SimpleNamespace
import unittest

from .bvr import (
    BvrController,
    fighter_tracks,
    in_launch_envelope,
    is_missile_track,
    launch_range_m,
)
from .enemy_contact import EnemyContactTracker
from .missile_threat import MissileThreatTracker


def _point(latitude, longitude, altitude_m=8000.0):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


def _weapon(name="aam_medium", count=6):
    return SimpleNamespace(name=name, enabled=True, count=count)


def _unit(platform_id="red_fighter_01", heading_deg=90.0):
    return SimpleNamespace(
        platform_id=platform_id,
        position=_point(0.0, 0.0),
        attitude=SimpleNamespace(heading_deg=heading_deg),
        weapons=[_weapon()],
    )


def _track(target_id="blue_fighter_01", model="F22", longitude=1.3, heading_deg=270.0):
    return SimpleNamespace(
        target_id=target_id,
        target_side="blue",
        model=model,
        position=_point(0.0, longitude),
        attitude=SimpleNamespace(heading_deg=heading_deg),
        detected_by=["red_fighter_01", "red_fighter_02"],
        velocity=SimpleNamespace(north_mps=0.0, east_mps=0.0, up_mps=0.0),
    )


def _observation(sim_time, units, tracks, events=()):
    return SimpleNamespace(
        sim_time=sim_time,
        side="red",
        own_units=list(units),
        tracks=list(tracks),
        events=list(events),
    )


def _event(event_id, event_type, sim_time, shooter=None, target=None):
    return SimpleNamespace(
        event_id=event_id,
        event_type=event_type,
        sim_time=sim_time,
        shooter=shooter,
        target=target,
        weapon="AA_MISSILE_MEDIUM",
        platform=None,
    )


class BvrTests(unittest.TestCase):
    def test_missile_tracks_are_not_fighter_contacts(self):
        fighter = _track()
        missile = _track("blue_fighter_01_aam_medium_1", "AA_MISSILE_MEDIUM_ENTITY")
        observation = _observation(0.0, [_unit()], [fighter, missile])

        self.assertFalse(is_missile_track(fighter))
        self.assertTrue(is_missile_track(missile))
        self.assertEqual([track.target_id for track in fighter_tracks(observation)], [fighter.target_id])

        contacts = EnemyContactTracker().update(_observation(0.0, [_unit()], [missile]))
        self.assertEqual(contacts["known_alive_ids"], [])
        self.assertEqual(contacts["lost_contact_ids"], [])

    def test_head_on_bvr_launch_enters_fixed_crank(self):
        unit = _unit()
        target = _track()
        observation = _observation(0.0, [unit], [target])
        controller = BvrController()

        self.assertTrue(in_launch_envelope(unit, target))
        actions = controller.apply(
            observation,
            [
                {"type": "fire", "platform_id": unit.platform_id, "weapon_name": "aam_medium", "target_id": target.target_id},
                {"type": "set_flight", "platform_id": unit.platform_id, "heading_deg": 90.0, "altitude_m": 8000.0, "mach": 0.9},
            ],
            {},
            {},
        )

        self.assertEqual([action["type"] for action in actions], ["fire", "set_flight"])
        flight = actions[1]
        self.assertEqual(flight["mach"], 1.18)
        self.assertAlmostEqual(flight["heading_deg"], 43.0, delta=0.1)
        self.assertEqual(controller.metadata()["bvr_mode"][unit.platform_id], "bvr_crank")

    def test_flank_and_cold_aspects_use_shorter_launch_ranges(self):
        unit = _unit()
        flank = _track(longitude=1.0, heading_deg=0.0)
        cold = _track(longitude=1.0, heading_deg=90.0)
        distant_flank = _track(longitude=1.2, heading_deg=0.0)

        self.assertEqual(launch_range_m(unit, flank), 115000.0)
        self.assertEqual(launch_range_m(unit, cold), 85000.0)
        self.assertTrue(in_launch_envelope(unit, flank))
        self.assertFalse(in_launch_envelope(unit, cold))
        self.assertFalse(in_launch_envelope(unit, distant_flank))

    def test_weapon_fired_event_triggers_counterfire_and_defense_once(self):
        unit = _unit()
        shooter = _track()
        event = _event(7, "WeaponFired", 10.0, shooter.target_id, unit.platform_id)
        observation = _observation(10.0, [unit], [shooter], [event])
        tracker = MissileThreatTracker()
        threats = tracker.update(observation)
        controller = BvrController()

        actions = controller.apply(
            observation,
            [{"type": "set_flight", "platform_id": unit.platform_id, "heading_deg": 90.0, "altitude_m": 8000.0, "mach": 0.9}],
            {},
            threats,
        )
        self.assertEqual([action["type"] for action in actions], ["fire", "set_flight"])
        self.assertEqual(actions[0]["target_id"], shooter.target_id)
        self.assertEqual(actions[1]["mach"], 1.4)
        self.assertEqual(controller.metadata()["bvr_mode"][unit.platform_id], "missile_defense")

        repeated = tracker.update(observation)
        self.assertEqual(len(repeated["active_threats"]), 1)
        cleared = tracker.update(
            _observation(11.0, [unit], [shooter], [event, _event(8, "WeaponMissed", 11.0, shooter.target_id, unit.platform_id)])
        )
        self.assertEqual(cleared["active_threats"], [])


if __name__ == "__main__":
    unittest.main()
