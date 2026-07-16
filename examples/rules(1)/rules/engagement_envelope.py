import math
from dataclasses import dataclass
from enum import Enum

from .config import (
    EARTH_RADIUS_M,
    ENGAGEMENT_COLD_RANGE_M,
    ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG,
    ENGAGEMENT_FLANK_ASPECT_MAX_DEG,
    ENGAGEMENT_FLANK_RANGE_M,
    ENGAGEMENT_HOT_ASPECT_MAX_DEG,
    ENGAGEMENT_HOT_RANGE_M,
    ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG,
)


class AspectClass(Enum):
    HOT = "HOT"
    FLANK = "FLANK"
    COLD = "COLD"


@dataclass(frozen=True)
class EngagementGeometry:
    distance_m: float
    target_aspect_deg: float
    aspect_class: AspectClass
    shooter_heading_error_deg: float
    dynamic_launch_range_m: float
    within_dynamic_range: bool
    heading_aligned_normal: bool
    heading_aligned_counter: bool


@dataclass(frozen=True)
class FireEligibility:
    eligible: bool
    platform_id: str
    target_id: object
    distance_m: object
    dynamic_launch_range_m: object
    aspect_deg: object
    aspect_class: object
    heading_error_deg: object
    heading_error_limit_deg: float
    target_observed: bool
    detected_by_self: bool
    has_weapon: bool
    cooldown_ready: bool
    cooldown_remaining_s: float
    pending_shot_clear: bool
    ineligible_reasons: tuple
    detected_by: tuple = ()
    pending_shot_ids: tuple = ()


def compute_target_aspect_deg(shooter_position, target_position, target_heading_deg):
    """Angle between target heading and bearing from target toward shooter, in degrees."""
    target_to_shooter = bearing_deg(target_position, shooter_position)
    return abs(angle_diff_deg(target_to_shooter, target_heading_deg))


def compute_shooter_heading_error_deg(shooter_position, target_position, shooter_heading_deg):
    """Angle between shooter heading and bearing from shooter toward target, in degrees."""
    shooter_to_target = bearing_deg(shooter_position, target_position)
    return abs(angle_diff_deg(shooter_to_target, shooter_heading_deg))


def classify_target_aspect(target_aspect_deg):
    if float(target_aspect_deg) <= ENGAGEMENT_HOT_ASPECT_MAX_DEG:
        return AspectClass.HOT
    if float(target_aspect_deg) <= ENGAGEMENT_FLANK_ASPECT_MAX_DEG:
        return AspectClass.FLANK
    return AspectClass.COLD


def compute_dynamic_launch_range_m(aspect_class):
    if aspect_class == AspectClass.HOT:
        return ENGAGEMENT_HOT_RANGE_M
    if aspect_class == AspectClass.FLANK:
        return ENGAGEMENT_FLANK_RANGE_M
    return ENGAGEMENT_COLD_RANGE_M


def evaluate_engagement_geometry(shooter_position, shooter_heading_deg, target_position, target_heading_deg):
    distance = distance_m(shooter_position, target_position)
    aspect = compute_target_aspect_deg(shooter_position, target_position, target_heading_deg)
    aspect_class = classify_target_aspect(aspect)
    heading_error = compute_shooter_heading_error_deg(shooter_position, target_position, shooter_heading_deg)
    launch_range = compute_dynamic_launch_range_m(aspect_class)
    return EngagementGeometry(
        distance_m=distance,
        target_aspect_deg=aspect,
        aspect_class=aspect_class,
        shooter_heading_error_deg=heading_error,
        dynamic_launch_range_m=launch_range,
        within_dynamic_range=distance <= launch_range,
        heading_aligned_normal=heading_error <= ENGAGEMENT_NORMAL_HEADING_ERROR_MAX_DEG,
        heading_aligned_counter=heading_error <= ENGAGEMENT_COUNTER_HEADING_ERROR_MAX_DEG,
    )


def bearing_deg(a, b):
    """Bearing from position a to position b, clockwise from north, normalized to [0, 360)."""
    lat1 = math.radians(a.latitude)
    lat2 = math.radians(b.latitude)
    dlon = math.radians(b.longitude - a.longitude)
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return normalize_deg(math.degrees(math.atan2(y, x)))


def angle_diff_deg(a, b):
    """Smallest signed difference a-b in degrees, in [-180, 180)."""
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def normalize_deg(value):
    return float(value) % 360.0


def distance_m(a, b):
    lat1 = math.radians(a.latitude)
    lon1 = math.radians(a.longitude)
    lat2 = math.radians(b.latitude)
    lon2 = math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    horizontal = 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))
    vertical = float(b.altitude_m) - float(a.altitude_m)
    return math.sqrt(horizontal * horizontal + vertical * vertical)
