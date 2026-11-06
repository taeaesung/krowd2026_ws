import math

from mola_slam_mapping.gnss_imu_pose_node import geodetic_to_utm
from mola_slam_mapping.route_initial_pose_node import nearest_route_seed


def test_nearest_route_seed_returns_map_pose_and_route_yaw():
    latitude = 37.2990583
    longitude = 127.0370520
    easting, northing, _, _, _ = geodetic_to_utm(latitude, longitude, 115.169)
    yaw_enu_from_map = math.radians(-131.66832598057167)
    route = [
        (0, easting + 1.6563684428, northing + 1.8729604311, 115.169, yaw_enu_from_map),
        (1, easting + 20.0, northing, 115.169, 0.0),
    ]
    seed = nearest_route_seed(
        latitude,
        longitude,
        route,
        latitude,
        longitude,
        115.169,
        1.6563684428,
        1.8729604311,
        -1.0295233097,
        yaw_enu_from_map,
        2.5038,
        0.0,
    )
    assert seed.waypoint_index == 0
    assert seed.distance < 3.0
    assert abs(seed.x) < 0.01
    assert abs(seed.y) < 0.01
    assert abs(seed.yaw) < 1e-9
