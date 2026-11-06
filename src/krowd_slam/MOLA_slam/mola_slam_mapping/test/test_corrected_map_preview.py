import math
from types import SimpleNamespace

import numpy as np
import pytest
from nav_msgs.msg import Odometry

from mola_slam_mapping.corrected_map_preview_node import (
    CorrectedMapPreviewNode,
    cloud_from_points,
    deduplicate_voxels,
    transform_points,
    voxel_keys,
)


def _matrix(x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    yaw = math.radians(yaw_deg)
    return np.array([
        [math.cos(yaw), -math.sin(yaw), 0.0, x],
        [math.sin(yaw), math.cos(yaw), 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def test_points_in_the_same_voxel_share_a_key_and_neighbours_do_not():
    points = np.array([
        [0.01, 0.01, 0.01],
        [0.24, 0.24, 0.24],  # same 0.25 m voxel
        [0.26, 0.01, 0.01],  # next voxel along x
        [-0.01, 0.01, 0.01],  # the voxel below zero, not the same one
    ])

    keys = voxel_keys(points, 0.25)

    assert keys[0] == keys[1]
    assert len({int(keys[0]), int(keys[2]), int(keys[3])}) == 3


def test_voxel_keys_reject_a_point_too_far_to_index():
    with pytest.raises(ValueError, match="too far"):
        voxel_keys(np.array([[1.0e9, 0.0, 0.0]]), 0.25)

    with pytest.raises(ValueError, match="voxel size"):
        voxel_keys(np.array([[0.0, 0.0, 0.0]]), 0.0)


def test_deduplication_keeps_the_first_point_of_each_voxel_in_order():
    points = np.array([
        [0.10, 0.0, 0.0],
        [5.00, 0.0, 0.0],
        [0.20, 0.0, 0.0],  # duplicate of the first voxel
        [5.10, 0.0, 0.0],  # duplicate of the second
        [9.00, 0.0, 0.0],
    ])

    kept = deduplicate_voxels(points, 0.25)

    assert np.allclose(kept, [[0.10, 0.0, 0.0], [5.00, 0.0, 0.0], [9.00, 0.0, 0.0]])
    assert len(deduplicate_voxels(np.zeros((0, 3)), 0.25)) == 0


def test_transform_places_a_scan_by_the_pose():
    # A point 2 m ahead of a vehicle at (10, 5) heading +y is at (10, 7).
    placed = transform_points(np.array([[2.0, 0.0, 0.0]]), _matrix(10.0, 5.0, yaw_deg=90.0))

    assert np.allclose(placed, [[10.0, 7.0, 0.0]], atol=1e-9)
    assert len(transform_points(np.zeros((0, 3)), _matrix())) == 0


def test_the_preview_moves_with_the_pose_it_is_given():
    """The whole point: the same scan lands elsewhere on a corrected pose.

    A pose 2 m higher and rotated 1 deg in yaw is the kind of difference
    between LiDAR odometry and the RTK-corrected fused pose that the preview
    exists to show.
    """

    scan = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0]])

    drifted = transform_points(scan, _matrix(0.0, 0.0, 2.0, yaw_deg=1.0))
    corrected = transform_points(scan, _matrix())

    separation = np.linalg.norm(drifted - corrected, axis=1)
    assert separation.min() > 2.0  # the 2 m of z alone, plus the 20 m lever arm


def _reporting_node(**state):
    logged = {"info": [], "warning": []}
    node = SimpleNamespace(
        _scans_accumulated=0,
        _points=np.zeros((0, 3)),
        _pending=[],
        _scans_dropped=0,
        get_parameter=lambda name: SimpleNamespace(value="/fused_localization/pose"),
        get_logger=lambda: SimpleNamespace(
            info=lambda m: logged["info"].append(m),
            warning=lambda m: logged["warning"].append(m),
        ),
    )
    node.__dict__.update(state)
    return node, logged


def test_a_preview_that_can_place_nothing_says_so():
    """The silent-failure case found by the live smoke test.

    With every scan dropped for want of a bracketing pose, the earlier report
    saw no accumulated scans and no queue and returned without a word, so a
    completely dead preview looked exactly like an idle one.
    """

    node, logged = _reporting_node(_scans_dropped=4)

    CorrectedMapPreviewNode._report(node)

    assert not logged["info"]
    assert len(logged["warning"]) == 1
    assert "4 dropped" in logged["warning"][0]
    assert "max_pose_time_difference_sec" in logged["warning"][0]


def test_a_working_preview_reports_at_info_and_an_idle_one_is_silent():
    working, logged = _reporting_node(
        _scans_accumulated=12, _points=np.zeros((500, 3)), _scans_dropped=1
    )
    CorrectedMapPreviewNode._report(working)
    assert len(logged["info"]) == 1 and not logged["warning"]
    assert "12 scans placed, 500 points" in logged["info"][0]

    idle, idle_log = _reporting_node()
    CorrectedMapPreviewNode._report(idle)
    assert not idle_log["info"] and not idle_log["warning"]


def test_pose_bracket_tolerance_is_wider_than_the_fusion_nodes():
    """/lidar_odometry/pose has a p95 gap of 0.2-0.3 s, so 0.15 s drops runs."""

    from pathlib import Path

    source = Path(
        CorrectedMapPreviewNode.__module__.replace(".", "/") + ".py"
    ).name
    text = (Path(__file__).resolve().parents[1] / "mola_slam_mapping" / source).read_text()
    assert 'declare_parameter("max_pose_time_difference_sec", 0.30)' in text


def test_cloud_message_is_a_valid_xyz_float32_cloud():
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 0.25]])

    message = cloud_from_points(points, "map", Odometry().header.stamp)

    assert message.header.frame_id == "map"
    assert (message.height, message.width) == (1, 2)
    assert [field.name for field in message.fields] == ["x", "y", "z"]
    assert message.point_step == 12 and message.row_step == 24
    assert np.allclose(
        np.frombuffer(message.data, dtype=np.float32).reshape(-1, 3), points
    )
