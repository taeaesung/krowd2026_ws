import math

import pytest

from types import SimpleNamespace

from mola_slam_mapping.gnss_imu_pose_node import (
    GnssImuPoseNode,
    circular_lerp,
    course_spread,
    enu_heading_from_map_yaw,
    euler_from_quaternion,
    geodetic_to_enu,
    geodetic_to_utm,
    inverse_quaternion,
    multiply_quaternions,
    quaternion_from_euler,
    rotate_vector,
    window_start_index,
    yaw_from_quaternion,
)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def test_geodetic_origin_is_zero_enu():
    enu = geodetic_to_enu(
        37.3001273,
        127.0412465,
        117.312,
        37.3001273,
        127.0412465,
        117.312,
    )
    assert enu == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-9)

def test_geodetic_to_utm_matches_proj_reference():
    easting, northing, height, zone, band = geodetic_to_utm(
        37.3001273,
        127.0412465,
        117.312,
    )
    assert easting == pytest.approx(326396.14, abs=0.01)
    assert northing == pytest.approx(4129965.97, abs=0.01)
    assert (height, zone, band) == (117.312, 52, "S")



def test_geodetic_longitude_increment_is_east():
    latitude = 37.3
    longitude = 127.04
    east_m = 10.0
    delta_longitude = math.degrees(
        east_m / (6378137.0 * math.cos(math.radians(latitude)))
    )
    enu = geodetic_to_enu(
        latitude,
        longitude + delta_longitude,
        100.0,
        latitude,
        longitude,
        100.0,
    )
    assert enu[0] == pytest.approx(east_m, abs=0.03)
    assert enu[1] == pytest.approx(0.0, abs=0.03)


def test_inverse_quaternion_restores_vector():
    quaternion = quaternion_from_euler(
        math.radians(73.0),
        math.radians(4.0),
        math.radians(-2.0),
    )
    vector = (2.0, -1.0, 0.5)
    rotated = rotate_vector(quaternion, vector)
    restored = rotate_vector(inverse_quaternion(quaternion), rotated)
    assert restored == pytest.approx(vector, abs=1.0e-12)


def test_map_orientation_is_relative_to_saved_enu_map():
    q_enu_map = quaternion_from_euler(math.radians(80.0), 0.0, 0.0)
    q_enu_base = quaternion_from_euler(math.radians(125.0), 0.0, 0.0)
    q_map_base = multiply_quaternions(
        inverse_quaternion(q_enu_map),
        q_enu_base,
    )
    yaw, pitch, roll = euler_from_quaternion(q_map_base)
    assert math.degrees(_wrap(yaw)) == pytest.approx(45.0)
    assert pitch == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)


def test_heading_filter_crosses_wrap_without_large_jump():
    previous = math.radians(179.0)
    filtered = circular_lerp(previous, math.radians(-179.0), 0.5)
    assert math.degrees(_wrap(filtered)) == pytest.approx(-180.0)


def test_lidar_map_yaw_converts_to_the_enu_heading_the_pose_is_built_from():
    """gnss_imu_pose_node composes q_map_base = inverse(q_enu_map) * q_enu_base;
    seeding from a LiDAR map yaw has to invert exactly that."""

    for georeference_yaw, heading in ((0.3, 1.0), (-2.0, 2.9), (3.0, -3.0)):
        q_enu_map = quaternion_from_euler(georeference_yaw, 0.0, 0.0)
        q_map_base = multiply_quaternions(
            inverse_quaternion(q_enu_map), quaternion_from_euler(heading, 0.0, 0.0)
        )
        yaw_map = yaw_from_quaternion(*q_map_base)
        recovered = enu_heading_from_map_yaw(yaw_map, q_enu_map)
        assert _wrap(recovered - heading) == pytest.approx(0.0, abs=1.0e-9)


def test_alignment_window_drops_the_oldest_samples_past_its_travel():
    cumulative = [0.0, 1.0, 2.5, 4.0, 7.0, 9.5, 12.0]
    assert window_start_index(cumulative, 100.0) == 0
    assert window_start_index(cumulative, 10.0) == 2  # 12.0 - 2.5 = 9.5
    assert window_start_index(cumulative, 0.0) == len(cumulative) - 1
    assert window_start_index([], 10.0) == 0


def test_course_spread_is_the_largest_deviation_from_the_circular_mean():
    mean, spread = course_spread([0.1, -0.1, 0.05])
    assert mean == pytest.approx(0.0167, abs=1e-3)
    assert spread == pytest.approx(0.1167, abs=1e-3)
    # across the +-pi wrap
    mean, spread = course_spread([math.pi - 0.1, -math.pi + 0.1])
    assert abs(_wrap(mean - math.pi)) < 1e-9
    assert spread == pytest.approx(0.1)
    assert course_spread([]) == (0.0, math.inf)


def _alignment_buffer(points, min_distance=5.0, min_straightness=0.95):
    """A node with only the fields the straight-segment test reads."""

    node = GnssImuPoseNode.__new__(GnssImuPoseNode)
    node._min_distance = min_distance
    node._min_straightness = min_straightness
    node._alignment_samples = [
        SimpleNamespace(antenna_enu=(x, y, 0.0), course=0.0) for x, y in points
    ]
    cumulative, travel = [], 0.0
    for previous, current in zip(points, points[1:]):
        travel += math.hypot(current[0] - previous[0], current[1] - previous[1])
        cumulative.append(travel)
    node._alignment_cumulative = [0.0] + cumulative
    return node


def test_a_straight_driven_after_a_turn_still_commits_the_heading():
    """2026-09-17: 82 m by hand, heading never committed - the whole-buffer
    chord had a manoeuvre in it. The test window must be the recent stretch."""

    radius = 8.0
    arc = [
        (radius * math.sin(a), radius * (1.0 - math.cos(a)))
        for a in [i * 0.125 for i in range(13)]  # 12 m of a left-hand curve
    ]
    end_x, end_y = arc[-1]
    straight = [(end_x, end_y + i * 0.5) for i in range(1, 25)]  # 12 m due north
    node = _alignment_buffer(arc + straight)

    assert node._alignment_chord() is None  # everything since launch: too bent
    start = window_start_index(node._alignment_cumulative, 2.0 * node._min_distance)
    chord, straightness, count, yaw = node._alignment_chord(start)
    assert chord >= 5.0
    assert straightness >= 0.95
    assert count >= 5
    assert yaw == pytest.approx(math.pi / 2.0, abs=math.radians(3.0))


def test_a_short_or_bent_recent_stretch_does_not_commit():
    node = _alignment_buffer([(i * 0.5, 0.0) for i in range(8)])  # 3.5 m only
    assert node._alignment_chord() is None
    radius = 3.0
    circle = [
        (radius * math.sin(a), radius * (1.0 - math.cos(a)))
        for a in [i * 0.2 for i in range(20)]
    ]
    node = _alignment_buffer(circle)
    start = window_start_index(node._alignment_cumulative, 10.0)
    assert node._alignment_chord(start) is None


def test_eskf_heading_status_reads_only_the_eskf_entry():
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

    from mola_slam_mapping.gnss_imu_pose_node import eskf_heading_status

    other = DiagnosticStatus(name="ntrip", values=[KeyValue(key="heading_verified", value="True")])
    eskf = DiagnosticStatus(
        name="gps_imu_eskf",
        values=[
            KeyValue(key="heading_verified", value="True"),
            KeyValue(key="heading_source", value="rtk_course"),
        ],
    )
    assert eskf_heading_status(DiagnosticArray(status=[other]), "gps_imu_eskf") is None
    assert eskf_heading_status(
        DiagnosticArray(status=[other, eskf]), "gps_imu_eskf"
    ) == (True, "rtk_course")
    prior = DiagnosticStatus(
        name="gps_imu_eskf",
        values=[
            KeyValue(key="heading_verified", value="False"),
            KeyValue(key="heading_source", value="saved_prior"),
        ],
    )
    assert eskf_heading_status(DiagnosticArray(status=[prior]), "gps_imu_eskf") == (
        False,
        "saved_prior",
    )


def test_heading_acquisition_goes_eskf_then_lidar_then_motion():
    from mola_slam_mapping.gnss_imu_pose_node import heading_acquisition_state

    def state(**kw):
        values = dict(
            aligned=False, seed_source=None, seeding_enabled=True,
            eskf_pending=False, lidar_pending=False,
        )
        values.update(kw)
        return heading_acquisition_state(**values)

    # No GNSS fix: nothing can seed and no straight segment can be measured,
    # so motion is never requested.
    assert state(gnss_ok=False) == "WAITING_GNSS"
    assert state(gnss_ok=False, eskf_pending=True) == "WAITING_GNSS"
    assert state(eskf_pending=True, lidar_pending=True) == "WAITING_ESKF"
    assert state(lidar_pending=True) == "WAITING_LIDAR"
    assert state() == "NEEDS_MOTION"
    assert state(seeding_enabled=False, eskf_pending=True) == "NEEDS_MOTION"
    assert state(aligned=True, seed_source="ESKF (rtk_course)") == "ACQUIRED ESKF (rtk_course)"
    assert state(aligned=True) == "ACQUIRED rtk_straight"
