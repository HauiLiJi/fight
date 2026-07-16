import math
from dataclasses import dataclass


EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float
    altitude_m: float


def clamp(value, low, high):
    return max(low, min(high, value))


def wrap_heading_deg(value):
    return value % 360.0


def distance_m(a, b):
    lat1 = math.radians(a.latitude)
    lon1 = math.radians(a.longitude)
    lat2 = math.radians(b.latitude)
    lon2 = math.radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    horizontal = EARTH_RADIUS_M * 2.0 * math.asin(math.sqrt(hav))
    vertical = a.altitude_m - b.altitude_m
    return math.hypot(horizontal, vertical)


def bearing_deg(a, b):
    lat1 = math.radians(a.latitude)
    lon1 = math.radians(a.longitude)
    lat2 = math.radians(b.latitude)
    lon2 = math.radians(b.longitude)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )
    return wrap_heading_deg(math.degrees(math.atan2(x, y)))


def heading_offset_deg(base_heading, offset_deg):
    return wrap_heading_deg(base_heading + offset_deg)


def midpoint_position(a, b):
    """Return a local formation centre that works for the scenario map scale."""
    return GeoPoint(
        latitude=(a.latitude + b.latitude) / 2.0,
        longitude=(a.longitude + b.longitude) / 2.0,
        altitude_m=(a.altitude_m + b.altitude_m) / 2.0,
    )


def destination_point(origin, heading_deg, distance_meters, altitude_m=None):
    """Project a point along a great-circle bearing without depending on simulator APIs."""
    angular_distance = distance_meters / EARTH_RADIUS_M
    bearing = math.radians(heading_deg)
    latitude = math.radians(origin.latitude)
    longitude = math.radians(origin.longitude)

    target_latitude = math.asin(
        math.sin(latitude) * math.cos(angular_distance)
        + math.cos(latitude) * math.sin(angular_distance) * math.cos(bearing)
    )
    target_longitude = longitude + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(latitude),
        math.cos(angular_distance) - math.sin(latitude) * math.sin(target_latitude),
    )
    return GeoPoint(
        latitude=math.degrees(target_latitude),
        longitude=math.degrees(target_longitude),
        altitude_m=origin.altitude_m if altitude_m is None else altitude_m,
    )


def predict_position(position, north_mps, east_mps, up_mps, elapsed_s):
    """Project a tracked platform with its last reported N/E/U velocity."""
    horizontal_speed = math.hypot(north_mps, east_mps)
    if horizontal_speed <= 1.0e-6:
        return GeoPoint(
            latitude=position.latitude,
            longitude=position.longitude,
            altitude_m=position.altitude_m + up_mps * elapsed_s,
        )
    heading = math.degrees(math.atan2(east_mps, north_mps))
    return destination_point(
        position,
        heading,
        horizontal_speed * elapsed_s,
        altitude_m=position.altitude_m + up_mps * elapsed_s,
    )


def formation_slots(units, safe_heading_deg, separation_m, anchor_lead_m, altitude_m):
    """Create deterministic left/right formation slots around a safe moving anchor."""
    centre = midpoint_position(units[0].position, units[1].position)
    anchor = destination_point(
        centre,
        safe_heading_deg,
        anchor_lead_m,
        altitude_m=altitude_m,
    )
    half_separation = separation_m / 2.0
    return (
        destination_point(
            anchor,
            heading_offset_deg(safe_heading_deg, -90.0),
            half_separation,
            altitude_m=altitude_m,
        ),
        destination_point(
            anchor,
            heading_offset_deg(safe_heading_deg, 90.0),
            half_separation,
            altitude_m=altitude_m,
        ),
    )


def altitude_delta_m(unit, track):
    return track.position.altitude_m - unit.position.altitude_m


def pair_distance_m(units):
    if len(units) < 2:
        return 0.0
    return distance_m(units[0].position, units[1].position)


def select_nearest_track(unit, tracks):
    if not tracks:
        return None, float("inf")
    return min(
        ((track, distance_m(unit.position, track.position)) for track in tracks),
        key=lambda item: item[1],
    )


def count_nearby_tracks(unit, tracks, radius_m):
    return sum(
        1 for track in tracks if distance_m(unit.position, track.position) <= radius_m
    )


def support_available(units, index, max_support_distance_m=25000.0):
    if len(units) < 2:
        return False
    other_index = 1 - index
    return distance_m(
        units[index].position,
        units[other_index].position,
    ) <= max_support_distance_m


def unit_risk_score(unit, tracks, ally_units, unit_index, missile_threat_state=None):
    nearest_track, nearest_distance = select_nearest_track(unit, tracks)
    enemy_count_nearby = count_nearby_tracks(unit, tracks, 35000.0)
    isolated = not support_available(ally_units, unit_index)
    score = 0.0
    if nearest_track is not None:
        if nearest_distance < 18000.0:
            score += 3.0
        elif nearest_distance < 30000.0:
            score += 2.0
        elif nearest_distance < 50000.0:
            score += 1.0
    score += max(0, enemy_count_nearby - 1) * 1.5
    if isolated:
        score += 2.0
    active_threats = (missile_threat_state or {}).get("active_threats", [])
    incoming_count = sum(
        1 for threat in active_threats if threat.get("target_id") == unit.platform_id
    )
    if incoming_count:
        score += 4.0 + max(0, incoming_count - 1) * 1.5
    score += max(0.0, (6000.0 - unit.position.altitude_m) / 2500.0)
    return {
        "score": round(score, 2),
        "nearest_enemy_distance": None if nearest_track is None else round(nearest_distance, 1),
        "enemy_count_nearby": enemy_count_nearby,
        "is_isolated": isolated,
        "mutual_support_available": not isolated,
        "incoming_missile_count": incoming_count,
    }
