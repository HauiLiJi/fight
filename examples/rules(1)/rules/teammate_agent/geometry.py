import math


EARTH_RADIUS_M = 6371000.0


def clamp(value, low, high):
    return max(low, min(high, value))


def distance_m(a, b):
    lat1 = math.radians(a.latitude)
    lon1 = math.radians(a.longitude)
    lat2 = math.radians(b.latitude)
    lon2 = math.radians(b.longitude)
    hav = (
        math.sin((lat2 - lat1) / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2.0) ** 2
    )
    horizontal = EARTH_RADIUS_M * 2.0 * math.asin(math.sqrt(hav))
    return math.hypot(horizontal, a.altitude_m - b.altitude_m)


def bearing_deg(a, b):
    lat1 = math.radians(a.latitude)
    lon1 = math.radians(a.longitude)
    lat2 = math.radians(b.latitude)
    lon2 = math.radians(b.longitude)
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return math.degrees(math.atan2(x, y)) % 360.0


def heading_offset_deg(base_heading, offset_deg):
    return (base_heading + offset_deg) % 360.0


def select_nearest_track(unit, tracks):
    if not tracks:
        return None, float("inf")
    return min(
        ((track, distance_m(unit.position, track.position)) for track in tracks),
        key=lambda item: item[1],
    )


def support_available(units, index, max_support_distance_m=25000.0):
    if len(units) < 2:
        return False
    return distance_m(units[index].position, units[1 - index].position) <= max_support_distance_m


def unit_risk_score(unit, tracks, ally_units, unit_index, missile_threat_state=None):
    nearest_track, nearest_distance = select_nearest_track(unit, tracks)
    enemy_count = sum(
        distance_m(unit.position, track.position) <= 35000.0 for track in tracks
    )
    isolated = not support_available(ally_units, unit_index)
    score = 0.0
    if nearest_track is not None:
        if nearest_distance < 18000.0:
            score += 3.0
        elif nearest_distance < 30000.0:
            score += 2.0
        elif nearest_distance < 50000.0:
            score += 1.0
    score += max(0, enemy_count - 1) * 1.5
    if isolated:
        score += 2.0
    active = (missile_threat_state or {}).get("active_threats", [])
    incoming = sum(item.get("target_id") == unit.platform_id for item in active)
    if incoming:
        score += 4.0 + max(0, incoming - 1) * 1.5
    score += max(0.0, (6000.0 - unit.position.altitude_m) / 2500.0)
    return {"score": round(score, 2)}
