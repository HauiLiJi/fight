import math
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.rules.engagement_envelope import (
    AspectClass,
    angle_diff_deg,
    classify_target_aspect,
    compute_dynamic_launch_range_m,
    compute_shooter_heading_error_deg,
    compute_target_aspect_deg,
)


def main():
    test_target_aspect_classes()
    test_shooter_heading_error()
    test_angle_diff_wraparound()
    test_dynamic_ranges()
    test_boundary_classification()


def test_target_aspect_classes():
    shooter = _pos(0.0, 0.0)
    target = _pos(0.0, 1.0)
    assert _close(compute_target_aspect_deg(shooter, target, 270.0), 0.0)
    assert classify_target_aspect(compute_target_aspect_deg(shooter, target, 270.0)) == AspectClass.HOT
    assert _close(compute_target_aspect_deg(shooter, target, 0.0), 90.0)
    assert classify_target_aspect(compute_target_aspect_deg(shooter, target, 0.0)) == AspectClass.FLANK
    assert _close(compute_target_aspect_deg(shooter, target, 90.0), 180.0)
    assert classify_target_aspect(compute_target_aspect_deg(shooter, target, 90.0)) == AspectClass.COLD


def test_shooter_heading_error():
    shooter = _pos(0.0, 0.0)
    target = _pos(0.0, 1.0)
    assert _close(compute_shooter_heading_error_deg(shooter, target, 90.0), 0.0)
    assert _close(compute_shooter_heading_error_deg(shooter, target, 270.0), 180.0)


def test_angle_diff_wraparound():
    assert _close(abs(angle_diff_deg(1.0, 359.0)), 2.0)
    assert _close(abs(angle_diff_deg(359.0, 1.0)), 2.0)


def test_dynamic_ranges():
    assert compute_dynamic_launch_range_m(AspectClass.HOT) == 150000.0
    assert compute_dynamic_launch_range_m(AspectClass.FLANK) == 115000.0
    assert compute_dynamic_launch_range_m(AspectClass.COLD) == 85000.0


def test_boundary_classification():
    assert classify_target_aspect(60.0) == AspectClass.HOT
    assert classify_target_aspect(120.0) == AspectClass.FLANK
    assert classify_target_aspect(120.001) == AspectClass.COLD


def _pos(latitude, longitude, altitude_m=0.0):
    return SimpleNamespace(latitude=latitude, longitude=longitude, altitude_m=altitude_m)


def _close(a, b, tolerance=1e-3):
    return math.isclose(float(a), float(b), abs_tol=tolerance)


if __name__ == "__main__":
    main()
