import math
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped, TransformStamped

from mola_slam_mapping.waypoint_saver_node import (
    WaypointSaverNode,
    circular_yaw_spread,
    fit_rigid_transform,
    interpolate_position,
    straight_suffix_metrics,
    transform_pose_stamped,
    yaw_from_quaternion,
)


def _pose(x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    pose = PoseStamped()
    pose.header.frame_id = 'odom'
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(0.5 * yaw)
    pose.pose.orientation.w = math.cos(0.5 * yaw)
    return pose


def _transform(
    target: str,
    source: str,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = target
    transform.child_frame_id = source
    transform.transform.translation.x = x
    transform.transform.translation.y = y
    transform.transform.rotation.z = math.sin(0.5 * yaw)
    transform.transform.rotation.w = math.cos(0.5 * yaw)
    return transform


def test_yaw_spread_handles_samples_across_pi_wrap():
    spread = circular_yaw_spread(
        [math.radians(179.0), math.radians(-179.0), math.radians(178.0)]
    )

    assert math.degrees(spread) == pytest.approx(3.0)


def test_straight_suffix_requires_a_distance_long_straight_baseline():
    straight = [_pose(float(x), 0.0) for x in range(11)]
    corner = [_pose(0.0, 0.0), _pose(5.0, 0.0), _pose(5.0, 5.0)]

    travel, ratio = straight_suffix_metrics(straight, 10.0)
    assert travel == pytest.approx(10.0)
    assert ratio == pytest.approx(1.0)

    travel, ratio = straight_suffix_metrics(corner, 10.0)
    assert travel == pytest.approx(10.0)
    assert ratio == pytest.approx(math.sqrt(0.5))


def test_pose_transform_rotates_position_and_heading_together():
    pose = _pose(2.0, 0.0, math.radians(10.0))
    transform = _transform(
        'utm', 'odom', 100.0, 200.0, math.radians(90.0)
    )

    result = transform_pose_stamped(pose, transform)

    assert result.header.frame_id == 'utm'
    assert result.pose.position.x == pytest.approx(100.0)
    assert result.pose.position.y == pytest.approx(202.0)
    yaw = math.atan2(
        2.0
        * (
            result.pose.orientation.w * result.pose.orientation.z
            + result.pose.orientation.x * result.pose.orientation.y
        ),
        1.0
        - 2.0
        * (
            result.pose.orientation.y * result.pose.orientation.y
            + result.pose.orientation.z * result.pose.orientation.z
        ),
    )
    assert math.degrees(yaw) == pytest.approx(100.0)


def test_alignment_gate_waits_for_straight_motion_and_stable_yaw_window():
    node = SimpleNamespace(
        _raw_waypoints=[_pose(float(x), 0.0) for x in range(11)],
        _alignment_min_distance=10.0,
        _alignment_min_straightness=0.95,
        _alignment_yaw_window_ns=2_000_000_000,
        _alignment_min_yaw_samples=5,
        _alignment_max_yaw_spread=math.radians(3.0),
        _alignment_yaw_samples=deque(
            [
                (0, math.radians(20.0)),
                (500_000_000, math.radians(20.5)),
                (1_000_000_000, math.radians(19.5)),
                (1_500_000_000, math.radians(20.2)),
                (2_000_000_000, math.radians(20.0)),
            ]
        ),
    )

    ready, travel, straightness, spread = (
        WaypointSaverNode._alignment_conditions_met(node)
    )
    assert ready
    assert travel == pytest.approx(10.0)
    assert straightness == pytest.approx(1.0)
    assert math.degrees(spread) == pytest.approx(1.0)

    node._alignment_yaw_samples[-1] = (
        2_000_000_000,
        math.radians(30.0),
    )
    assert not WaypointSaverNode._alignment_conditions_met(node)[0]


def _rotated(points, yaw, dx=0.0, dy=0.0, dz=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(c * x - s * y + dx, s * x + c * y + dy, z + dz) for x, y, z in points]


def test_rigid_fit_recovers_the_frame_offset_and_ignores_a_lost_tracking_stretch():
    """testweek_0905_3: 74 m of driving recorded as 2.3 m must not steer the fit."""

    # A 200 m L-shaped route in the fused (map) frame.
    route = [(float(i), 0.0, 0.0) for i in range(100)] + [
        (99.0, float(i), 0.0) for i in range(1, 100)
    ]
    # LiDAR odometry saw it 60 deg rotated and 10 m away...
    lidar = _rotated(route, math.radians(-60.0), 10.0, -5.0, 0.3)
    # ...except for 25 samples where it stood still while the vehicle moved.
    for index in range(20, 45):
        lidar[index] = lidar[19]

    matrix, stats = fit_rigid_transform(lidar, route, outlier_distance_m=3.0)

    assert stats["pairs"] == len(route)
    # The stuck samples are 1..25 m from where they belong; the first three sit
    # inside the 3 m gate and are harmless, the other 22 must be excluded.
    assert stats["pairs"] - stats["inliers"] == 22
    # Those three in-gate samples still bias the solve by a few centimetres -
    # invisible against a placement whose purpose is metre-level truth.
    assert stats["residual_median_m"] < 0.1
    yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    assert yaw == pytest.approx(60.0, abs=0.1)
    recovered = (matrix @ np.array([*lidar[80], 1.0]))[:3]
    assert recovered == pytest.approx(route[80], abs=0.2)


def test_rigid_fit_survives_a_fifth_of_the_pairs_stuck_at_one_point():
    """The live pattern at 1 m route sampling: 74 of ~370 pairs map ONE LiDAR
    position onto 74 m of fused route. A least-squares solve seeded on all
    pairs, then rejecting, kept 11/367 live; the fit must instead explain the
    tracked majority."""

    route = [(float(i), 0.0, 0.0) for i in range(120)] + [
        (119.0, float(i), 0.0) for i in range(1, 120)
    ] + [(119.0 - i, 119.0, 0.0) for i in range(1, 120)]
    lidar = _rotated(route, math.radians(101.0), -30.0, 40.0)
    stuck_from, stuck_to = 5, 5 + 74
    for index in range(stuck_from, stuck_to):
        lidar[index] = lidar[stuck_from - 1]

    matrix, stats = fit_rigid_transform(lidar, route, outlier_distance_m=3.0)

    assert stats["inliers"] >= len(route) - 74
    assert stats["residual_median_m"] < 0.05
    yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    assert yaw == pytest.approx(-101.0, abs=0.05)


def test_rigid_fit_needs_three_pairs():
    assert fit_rigid_transform([(0, 0, 0), (1, 0, 0)], [(0, 0, 0), (1, 0, 0)], 3.0) is None


def test_position_interpolation_brackets_but_refuses_long_gaps():
    stamps = [0, 100, 200, 1_000]
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (10.0, 0.0, 0.0)]

    assert interpolate_position(stamps, positions, 150, 200) == pytest.approx((1.5, 0.0, 0.0))
    assert interpolate_position(stamps, positions, 100, 200) == (1.0, 0.0, 0.0)
    assert interpolate_position(stamps, positions, 500, 200) is None  # 800-tick gap
    assert interpolate_position(stamps, positions, -1, 200) is None
    assert interpolate_position(stamps, positions, 1_001, 200) is None


def _placement_node(buffer, route, lidar, min_pairs=20):
    stamps = [index * 100_000_000 for index in range(len(route))]
    node = SimpleNamespace(
        _tf_buffer=buffer,
        _utm_frame='utm',
        _enu_frame='enu',
        _route_frame='map',
        _pointcloud_map_frame='odom',
        _raw_waypoints=[_pose(x, y) for x, y, _ in route],
        _raw_waypoint_stamps_ns=stamps,
        _pointcloud_pose_stamps_ns=stamps[: len(lidar)],
        _pointcloud_positions=list(lidar),
        _placement_fit_min_pairs=min_pairs,
        _placement_fit_outlier_m=3.0,
        _placement_fit_min_inlier_fraction=0.5,
        _placement_fit_max_gap_ns=500_000_000,
        _placement_fit=None,
        _last_pointcloud_transform=None,
        _last_pointcloud_enu_transform=None,
        get_logger=lambda: SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None
        ),
    )
    node._placement_fit_pairs = lambda: WaypointSaverNode._placement_fit_pairs(node)
    node._fit_pointcloud_placement = (
        lambda: WaypointSaverNode._fit_pointcloud_placement(node)
    )
    return node


def test_pointcloud_placement_is_the_whole_run_fit_composed_with_the_static_georeference():
    """Neither the shutdown instant nor the alignment instant of map->odom."""

    route = [(float(i), 0.0, 0.0) for i in range(60)]
    lidar = _rotated(route, math.radians(-90.0), 5.0, 0.0)

    class Buffer:
        def lookup_transform(self, target, source, *args, **kwargs):
            assert source == 'map', "the fit must compose with the static utm/enu<-map edge"
            return _transform(target, source, 1000.0 if target == 'utm' else 0.0, 0.0, 0.0)

    node = _placement_node(Buffer(), route, lidar)
    WaypointSaverNode._refresh_pointcloud_transform(node)

    assert node._placement_fit["inliers"] == 60
    utm = node._last_pointcloud_transform
    assert utm.header.frame_id == 'utm' and utm.child_frame_id == 'odom'
    assert math.degrees(yaw_from_quaternion(utm.transform.rotation)) == pytest.approx(90.0)
    # odom origin (which is route (−5·cos, ..)): map->odom takes (5,0)->(0,0) after 90 deg
    assert utm.transform.translation.x == pytest.approx(1000.0, abs=1e-6)
    assert utm.transform.translation.y == pytest.approx(-5.0, abs=1e-6)
    enu = node._last_pointcloud_enu_transform
    assert enu.header.frame_id == 'enu'
    assert enu.transform.translation.x == pytest.approx(0.0, abs=1e-6)


def test_pointcloud_placement_falls_back_to_live_tf_until_enough_pairs():
    route = [(float(i), 0.0, 0.0) for i in range(10)]
    lidar = list(route)
    lookups = []

    class Buffer:
        def lookup_transform(self, target, source, *args, **kwargs):
            lookups.append((target, source))
            return _transform(target, source, 7.0, 0.0)

    node = _placement_node(Buffer(), route, lidar, min_pairs=20)
    WaypointSaverNode._refresh_pointcloud_transform(node)

    assert node._placement_fit is None
    assert ('utm', 'odom') in lookups and ('enu', 'odom') in lookups
    assert node._last_pointcloud_transform.transform.translation.x == 7.0


def test_minority_inlier_fit_is_refused_and_an_accepted_fit_is_kept():
    """Seen live: 3/47 pairs 'agreeing' while odometry was lost, yaw swinging
    -108 -> +29 deg between refreshes. Such a solution must never be adopted."""

    route = [(float(i), 0.0, 0.0) for i in range(60)]
    lidar = _rotated(route, math.radians(-90.0), 5.0, 0.0)

    class Buffer:
        def lookup_transform(self, target, source, *args, **kwargs):
            return _transform(target, source, 0.0, 0.0, 0.0)

    node = _placement_node(Buffer(), route, lidar)
    WaypointSaverNode._refresh_pointcloud_transform(node)
    accepted = node._placement_fit
    assert accepted["inliers"] == 60

    # Odometry now stands still for the next 100 samples while the vehicle moves:
    # the pairs are dominated by the lost stretch.
    stuck = [lidar[-1]] * 100
    node._raw_waypoints += [_pose(60.0 + i, 0.0) for i in range(100)]
    node._raw_waypoint_stamps_ns += [
        (60 + i) * 100_000_000 for i in range(100)
    ]
    node._pointcloud_pose_stamps_ns += [(60 + i) * 100_000_000 for i in range(100)]
    node._pointcloud_positions += stuck
    WaypointSaverNode._refresh_pointcloud_transform(node)

    assert node._placement_fit is accepted
    assert math.degrees(
        yaw_from_quaternion(node._last_pointcloud_transform.transform.rotation)
    ) == pytest.approx(90.0)


def test_projection_rebuild_moves_every_buffered_waypoint_with_one_transform():
    transforms = {
        'utm': _transform('utm', 'odom', 100.0, 200.0),
        'map': _transform('map', 'odom'),
    }

    class Buffer:
        def lookup_transform(self, target, source, *args, **kwargs):
            assert source == 'odom'
            return transforms[target]

    node = SimpleNamespace(
        _raw_waypoints=[_pose(0.0, 0.0), _pose(10.0, 0.0)],
        _tf_buffer=Buffer(),
        _utm_frame='utm',
        _visualization_frame='map',
        _route_frame='odom',
        _pointcloud_map_frame='odom',
        _last_route_transform=None,
        _last_pointcloud_transform=None,
        _last_visualization_transform=None,
        _alignment_ready=True,
        _corridor_half_width=1.0,
        _waypoints=[],
        _visualization_route=[],
        _latest_valid_local_pose=_pose(12.0, 0.0),
        _label_center=None,
        _record_alignment_yaw=lambda *args: None,
        _publish_markers=lambda: None,
    )

    assert WaypointSaverNode._refresh_route_projection(
        node, observation_stamp_ns=1
    )
    # The RViz label window centres on the vehicle, in the visualization frame -
    # here 2 m past the newest waypoint, which is where the car actually is.
    assert node._label_center == pytest.approx((12.0, 0.0))
    assert [point.x for point in node._waypoints] == pytest.approx(
        [100.0, 110.0]
    )
    assert [point.y for point in node._waypoints] == pytest.approx(
        [200.0, 200.0]
    )

    transforms['utm'] = _transform(
        'utm', 'odom', 100.0, 200.0, math.pi / 2.0
    )
    assert WaypointSaverNode._refresh_route_projection(
        node, observation_stamp_ns=2
    )
    assert [point.x for point in node._waypoints] == pytest.approx(
        [100.0, 100.0]
    )
    assert [point.y for point in node._waypoints] == pytest.approx(
        [200.0, 210.0]
    )
