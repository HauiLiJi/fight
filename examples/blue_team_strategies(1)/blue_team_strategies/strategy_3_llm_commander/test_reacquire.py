from types import SimpleNamespace
import unittest

from .enemy_contact import EnemyContactTracker
from .geometry import bearing_deg, distance_m
from .tactics import translate_tactic_to_actions


def _point(latitude, longitude, altitude_m=8000.0):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


def _unit(platform_id, latitude, longitude, heading_deg=90.0):
    return SimpleNamespace(
        platform_id=platform_id,
        position=_point(latitude, longitude),
        attitude=SimpleNamespace(heading_deg=heading_deg),
    )


def _track(target_id, latitude, longitude, north_mps=0.0, east_mps=0.0):
    return SimpleNamespace(
        target_id=target_id,
        target_side="blue",
        position=_point(latitude, longitude),
        velocity=SimpleNamespace(
            north_mps=north_mps,
            east_mps=east_mps,
            up_mps=0.0,
        ),
        detected_by=["red_fighter_01", "red_fighter_02"],
    )


def _observation(sim_time, tracks=(), events=()):
    return SimpleNamespace(
        sim_time=sim_time,
        side="red",
        own_units=[
            _unit("red_fighter_01", -0.05, 0.0),
            _unit("red_fighter_02", 0.05, 0.0),
        ],
        tracks=list(tracks),
        events=list(events),
    )


def _tactic(name):
    return {
        "tactic": name,
        "primary_target": "blue_fighter_01",
        "secondary_target": "",
    }


def _angle_error(actual, expected):
    return abs(((actual - expected + 180.0) % 360.0) - 180.0)


class ReacquireTests(unittest.TestCase):
    def test_contact_prediction_is_capped_at_120_seconds(self):
        tracker = EnemyContactTracker()
        tracker.update(_observation(0.0, [_track("blue_fighter_01", 0.0, 1.0, north_mps=100.0)]))

        state = tracker.update(_observation(200.0))
        contact = state["lost_contacts"][0]
        initial = _point(0.0, 1.0)
        predicted_distance = distance_m(initial, contact["predicted_position"])

        self.assertTrue(contact["prediction_capped"])
        self.assertAlmostEqual(predicted_distance, 12000.0, delta=50.0)

    def test_destroyed_contact_is_not_reacquired(self):
        tracker = EnemyContactTracker()
        tracker.update(_observation(0.0, [_track("blue_fighter_01", 0.0, 1.0)]))
        event = SimpleNamespace(
            event_type="PlatformBroken",
            platform="blue_fighter_01",
            target=None,
        )
        state = tracker.update(_observation(1.0, events=[event]))

        self.assertEqual(state["lost_contact_ids"], [])
        self.assertEqual(state["destroyed_ids"], ["blue_fighter_01"])

    def test_lost_contact_generates_forward_search_actions_without_fire(self):
        tracker = EnemyContactTracker()
        tracker.update(_observation(0.0, [_track("blue_fighter_01", 0.0, 1.0)]))
        observation = _observation(30.0)
        contact_state = tracker.update(observation)

        batch = translate_tactic_to_actions(
            observation,
            _tactic("focus_fire"),
            {},
            contact_state,
        )
        headings = [action.heading_deg for action in batch.actions]

        self.assertEqual(len(batch.actions), 2)
        self.assertTrue(all(action.type == "set_flight" for action in batch.actions))
        self.assertTrue(all(_angle_error(heading, 90.0) < 90.0 for heading in headings))

    def test_defensive_regroup_approaches_far_contact_and_retreats_from_close_contact(self):
        far_observation = _observation(0.0, [_track("blue_fighter_01", 0.0, 1.0)])
        far_batch = translate_tactic_to_actions(
            far_observation,
            _tactic("defensive_regroup"),
            {},
        )
        self.assertTrue(
            all(_angle_error(action.heading_deg, 90.0) < 90.0 for action in far_batch.actions)
        )

        close_observation = _observation(0.0, [_track("blue_fighter_01", 0.0, 0.2)])
        close_batch = translate_tactic_to_actions(
            close_observation,
            _tactic("defensive_regroup"),
            {},
        )
        self.assertTrue(
            all(_angle_error(action.heading_deg, 270.0) < 90.0 for action in close_batch.actions)
        )

    def test_no_contact_history_still_outputs_search_formation(self):
        observation = _observation(30.0)
        batch = translate_tactic_to_actions(observation, _tactic("split_targets"), {}, {})

        self.assertEqual(len(batch.actions), 2)
        self.assertTrue(all(action.type == "set_flight" for action in batch.actions))


if __name__ == "__main__":
    unittest.main()
