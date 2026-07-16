import math


EARTH_RADIUS_M = 6371000.0


def normalize(value):
    return value % 360.0


def difference(first, second):
    return (first - second + 180.0) % 360.0 - 180.0


def clamp(value, low, high):
    return max(low, min(high, value))


def distance(first, second):
    latitude = math.radians((first.latitude + second.latitude) * 0.5)
    north = math.radians(second.latitude - first.latitude) * EARTH_RADIUS_M
    east = math.radians(second.longitude - first.longitude) * EARTH_RADIUS_M * math.cos(latitude)
    vertical = second.altitude_m - first.altitude_m
    return math.sqrt(north * north + east * east + vertical * vertical)


def bearing(first, second):
    latitude = math.radians((first.latitude + second.latitude) * 0.5)
    north = math.radians(second.latitude - first.latitude) * EARTH_RADIUS_M
    east = math.radians(second.longitude - first.longitude) * EARTH_RADIUS_M * math.cos(latitude)
    return normalize(math.degrees(math.atan2(east, north)))
