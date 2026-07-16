from types import SimpleNamespace
import unittest

from .geometry import bearing_deg, destination_point, distance_m, formation_slots
from .tactics import _defensive_regroup, _regroup_mach


def _point(latitude, longitude, altitude_m=8000.0):
    return SimpleNamespace(
        latitude=latitude,
        longitude=longitude,
        altitude_m=altitude_m,
    )


def _unit(platform_id, latitude, longitude, altitude_m=8000.0, heading_deg=90.0):
    return SimpleNamespace(
        platform_id=platform_id,
        position=_point(latitude, longitude, altitude_m),
        attitude=SimpleNamespace(heading_deg=heading_deg),
    )


def _track(latitude, longitude):
    return SimpleNamespace(position=_point(latitude, longitude))


class DefensiveRegroupTests(unittest.TestCase):
    def test_slots_have_fixed_target_separation(self):
        units = [_unit("red_fighter_01", 0.0, 0.0), _unit("red_fighter_02", 0.0, 0.6)]
        slots = formation_slots(units, 270.0, 12000.0, 4000.0, 8500.0)
        self.assertAlmostEqual(distance_m(slots[0], slots[1]), 12000.0, delta=5.0)

    def test_wide_formation_flies_toward_slots_and_away_from_enemy(self):
        units = [_unit("red_fighter_01", -0.3, 0.0), _unit("red_fighter_02", 0.3, 0.0)]
        enemy = _track(0.0, 0.2)
        actions = _defensive_regroup(units, [enemy])
        self.assertEqual(len(actions), 2)
        self.assertEqual({action["mach"] for action in actions}, {0.96})
        safe_heading = bearing_deg(enemy.position, _point(0.0, 0.0))
        for action in actions:
            self.assertLess(
                abs(((action["heading_deg"] - safe_heading + 180.0) % 360.0) - 180.0),
                90.0,
            )

    def test_speed_bands_expand_hold_and_recover_formation(self):
        for separation_m, expected_mach in (
            (8000.0, 0.82),
            (12000.0, 0.90),
            (30000.0, 0.96),
        ):
            self.assertEqual(_regroup_mach(separation_m, 4000.0), expected_mach)

    def test_wide_formation_converges_to_target_band(self):
        units = [_unit("red_fighter_01", -0.3, 0.0), _unit("red_fighter_02", 0.3, 0.0)]
        enemy = _track(0.0, 0.2)
        distances = []
        for _ in range(140):
            distances.append(distance_m(units[0].position, units[1].position))
            for unit, action in zip(units, _defensive_regroup(units, [enemy])):
                unit.position = destination_point(
                    unit.position,
                    action["heading_deg"],
                    action["mach"] * 300.0,
                    altitude_m=action["altitude_m"],
                )
        self.assertLess(distances[-1], distances[0])
        self.assertTrue(
            all(later <= earlier + 1.0 for earlier, later in zip(distances, distances[1:]))
        )
        self.assertGreaterEqual(distances[-1], 10000.0)
        self.assertLessEqual(distances[-1], 14000.0)

    def test_single_survivor_keeps_valid_retreat_action(self):
        unit = _unit("red_fighter_01", 0.0, 0.0)
        actions = _defensive_regroup([unit], [_track(0.0, 1.0)])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["platform_id"], unit.platform_id)
        self.assertGreaterEqual(actions[0]["heading_deg"], 0.0)
        self.assertLess(actions[0]["heading_deg"], 360.0)


if __name__ == "__main__":
    unittest.main()
