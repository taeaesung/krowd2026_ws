"""Lane detections placed on the map with the localized pose."""

import math

import numpy as np
import pytest

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from mola_slam_mapping.lane_map_overlay_node import (
    PoseBuffer,
    Pose2D,
    base_to_map,
    lanes_from_markers,
    map_to_base,
    parse_namespace,
)

S = 1_000_000_000


def test_namespace_gives_class_track_and_fallback():
    assert parse_namespace("leftSolid_track_3") == ("leftSolid", 3, False)
    assert parse_namespace("rightDashed_track_12_2d_fallback") == ("rightDashed", 12, True)
    assert parse_namespace("rightSolid") == ("rightSolid", -1, False)


def test_only_lane_line_strips_are_taken():
    delete = Marker(action=Marker.DELETEALL)
    lane = Marker(
        ns="leftSolid_track_1", type=Marker.LINE_STRIP, action=Marker.ADD,
        points=[Point(x=1.0, y=1.5, z=0.8), Point(x=6.0, y=1.6, z=0.8)],
    )
    lane.header.stamp = Time(sec=10, nanosec=500_000_000)
    lane.scale.x = 0.14
    text = Marker(ns="label", type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
    stamp_ns, lanes = lanes_from_markers(MarkerArray(markers=[delete, lane, text]))
    assert stamp_ns == 10 * S + 500_000_000
    assert len(lanes) == 1
    assert lanes[0].class_name == "leftSolid"
    np.testing.assert_allclose(lanes[0].points, [[1.0, 1.5], [6.0, 1.6]])  # z dropped


def test_a_lane_1_5_m_left_lands_1_5_m_left_of_the_vehicle_on_the_map():
    # Vehicle at (100, 50) facing north: its left is -x.
    pose = Pose2D(0, 100.0, 50.0, 2.0, math.pi / 2)
    on_map = base_to_map(np.array([[5.0, 1.5]]), pose)
    np.testing.assert_allclose(on_map, [[98.5, 55.0]])
    np.testing.assert_allclose(map_to_base(on_map, pose), [[5.0, 1.5]])


def test_pose_is_interpolated_to_the_image_time_and_extrapolated_briefly():
    poses = PoseBuffer(max_gap_ns=int(0.5 * S), max_extrapolation_ns=int(0.5 * S))
    poses.add(Pose2D(10 * S, 0.0, 0.0, 0.0, 0.0, speed_mps=4.0))
    poses.add(Pose2D(10 * S + 200_000_000, 0.8, 0.0, 0.0, 0.0, speed_mps=4.0))
    middle = poses.at(10 * S + 100_000_000)
    assert middle.x == pytest.approx(0.4)
    # 150 ms past the last pose: carried forward at 4 m/s.
    ahead = poses.at(10 * S + 350_000_000)
    assert ahead.x == pytest.approx(0.8 + 4.0 * 0.15)
    # Too far past, or before the buffer: no pose rather than a wrong one.
    assert poses.at(11 * S) is None
    assert poses.at(9 * S) is None


def test_a_gap_in_the_poses_is_not_bridged():
    poses = PoseBuffer(max_gap_ns=int(0.5 * S), max_extrapolation_ns=int(0.5 * S))
    poses.add(Pose2D(10 * S, 0.0, 0.0, 0.0, 0.0))
    poses.add(Pose2D(12 * S, 8.0, 0.0, 0.0, 0.0))
    assert poses.at(11 * S) is None
