import math

import pytest
from nav_msgs.msg import Odometry

from mola_slam_mapping.map_odom_tf_node import (
    BASE_LINK_TF_SOURCES,
    AutoBaseLinkTfGate,
    MapOdomTfNode,
    compose_map_to_odom,
    interpolate_pose_by_stamp,
    odometry_to_transform,
    path_jump_limit_m,
    same_frame_pose_distance_xy,
    validate_base_link_tf_source,
    vehicle_marker_from_odometry,
)


def _odometry(parent, child, x, y, yaw, stamp_ns=0):
    message = Odometry()
    message.header.frame_id = parent
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    message.child_frame_id = child
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    message.pose.pose.orientation.z = math.sin(yaw / 2.0)
    message.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return message


def test_base_link_tf_source_is_one_choice_not_two_flags():
    """base_link may have exactly one parent, so the sources are exclusive."""

    assert BASE_LINK_TF_SOURCES == ("none", "lidar", "state", "auto")
    assert validate_base_link_tf_source("State") == "state"
    assert validate_base_link_tf_source("Auto") == "auto"
    assert validate_base_link_tf_source(" lidar ") == "lidar"
    for rejected in ("", "true", "both", "rtk", "gnss"):
        with pytest.raises(ValueError, match="base_link_tf_source"):
            validate_base_link_tf_source(rejected)


def test_the_state_pose_yields_a_map_to_base_link_edge():
    """In localization the RTK pose is already map->base_link.

    That is the edge a base_link-fixed RViz view needs; without it the TF tree
    is two disconnected islands and nothing resolves.
    """

    transform = odometry_to_transform(
        _odometry("map", "base_link", 10.0, -4.0, math.pi / 2.0, stamp_ns=5)
    )
    assert transform.header.frame_id == "map"
    assert transform.child_frame_id == "base_link"
    assert transform.transform.translation.x == pytest.approx(10.0)
    assert transform.header.stamp.nanosec == 5


def test_auto_base_link_tf_waits_out_the_grace_period_before_publishing():
    """A slower owner (the vehicle ESKF) gets time to appear on /tf first."""

    gate = AutoBaseLinkTfGate(grace_ns=2_000_000_000, start_ns=10)
    assert not gate.may_publish("base_link", now_ns=10)
    assert not gate.may_publish("base_link", now_ns=2_000_000_009)
    assert gate.may_publish("base_link", now_ns=2_000_000_010)


def test_auto_base_link_tf_is_not_fooled_by_its_own_echo_on_tf():
    """This node's broadcast comes back on its own /tf subscription."""

    gate = AutoBaseLinkTfGate(grace_ns=0, start_ns=0)
    gate.note_published("map", "base_link", 123)
    assert gate.observe("map", "base_link", 123) is False
    assert gate.may_publish("base_link", now_ns=1)


def test_auto_base_link_tf_yields_to_an_existing_vehicle_eskf_edge():
    """Live vehicle: the ESKF already publishes odom->base_link, so never add a
    second parent - even when the ESKF edge arrives after the grace period."""

    gate = AutoBaseLinkTfGate(grace_ns=0, start_ns=0)
    assert gate.observe("enu", "odom", 5) is False
    assert gate.may_publish("base_link", now_ns=1)  # only base_link matters
    assert gate.observe("odom", "base_link", 7) is False  # never published yet
    assert gate.foreign_parent("base_link") == "odom"
    assert not gate.may_publish("base_link", now_ns=10**12)


def test_auto_base_link_tf_gives_the_edge_up_when_another_owner_appears():
    gate = AutoBaseLinkTfGate(grace_ns=0, start_ns=0)
    gate.note_published("map", "base_link", 100)
    assert gate.observe("map", "base_link", 100) is False
    # Same parent, different stamp: someone else is publishing map->base_link.
    assert gate.observe("map", "base_link", 150) is True
    assert not gate.may_publish("base_link", now_ns=10**12)
    assert gate.observe("map", "base_link", 200) is False  # reported only once


def test_auto_base_link_tf_forgets_old_own_stamps_and_rejects_negative_grace():
    gate = AutoBaseLinkTfGate(grace_ns=0, start_ns=0, own_history=2)
    for stamp in (1, 2, 3):
        gate.note_published("map", "base_link", stamp)
    assert gate.observe("map", "base_link", 3) is False
    with pytest.raises(ValueError, match="grace"):
        AutoBaseLinkTfGate(grace_ns=-1, start_ns=0)


def test_compose_map_to_odom_known_translation_and_rotation():
    lidar = _odometry("odom", "base_link", 2.0, 0.0, 0.0)
    state = _odometry("map", "base_link", 10.0, 2.0, math.pi / 2.0)

    transform = compose_map_to_odom(state, lidar)
    assert transform.header.frame_id == "map"
    assert transform.child_frame_id == "odom"
    assert transform.transform.translation.x == pytest.approx(10.0)
    assert transform.transform.translation.y == pytest.approx(0.0)
    assert transform.transform.rotation.z == pytest.approx(math.sqrt(0.5))
    assert transform.transform.rotation.w == pytest.approx(math.sqrt(0.5))


def test_compose_skips_same_frame_and_rejects_mismatched_vehicle_frames():
    state = _odometry("map", "base_link", 0.0, 0.0, 0.0)
    lidar = _odometry("map", "base_link", 0.0, 0.0, 0.0)
    assert compose_map_to_odom(state, lidar) is None

    lidar.child_frame_id = "other_vehicle"
    with pytest.raises(ValueError, match="different child frames"):
        compose_map_to_odom(state, lidar)


def test_odometry_to_transform_preserves_pose_stamp_and_normalizes_rotation():
    message = _odometry(
        "odom", "base_link", 3.0, -2.0, math.pi / 3.0, 1_234_567_890
    )
    message.pose.pose.position.z = 0.75
    message.pose.pose.orientation.z *= 2.0
    message.pose.pose.orientation.w *= 2.0

    transform = odometry_to_transform(message)
    assert transform.header.frame_id == "odom"
    assert transform.child_frame_id == "base_link"
    assert transform.header.stamp == message.header.stamp
    assert transform.transform.translation.x == pytest.approx(3.0)
    assert transform.transform.translation.y == pytest.approx(-2.0)
    assert transform.transform.translation.z == pytest.approx(0.75)
    rotation = transform.transform.rotation
    assert rotation.z == pytest.approx(math.sin(math.pi / 6.0))
    assert rotation.w == pytest.approx(math.cos(math.pi / 6.0))


def test_odometry_to_transform_rejects_invalid_frames_and_nonfinite_pose():
    message = _odometry("", "base_link", 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        odometry_to_transform(message)

    message.header.frame_id = "odom"
    message.child_frame_id = "odom"
    with pytest.raises(ValueError, match="self transform"):
        odometry_to_transform(message)

    message.child_frame_id = "base_link"
    message.pose.pose.position.z = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        odometry_to_transform(message)


def test_pose_timestamp_interpolation_uses_bracketing_samples():
    early = _odometry("odom", "base_link", 0.0, 0.0, 0.0, 0)
    late = _odometry(
        "odom", "base_link", 10.0, 4.0, math.pi, 2_000_000_000
    )
    matched = interpolate_pose_by_stamp(
        [(2_000_000_000, late), (0, early)],
        1_000_000_000,
        1_000_000_000,
    )
    assert matched is not None
    stamp_ns, message = matched
    assert stamp_ns == 1_000_000_000
    assert message.pose.pose.position.x == pytest.approx(5.0)
    assert message.pose.pose.position.y == pytest.approx(2.0)
    assert message.pose.pose.orientation.z == pytest.approx(math.sqrt(0.5))
    assert message.pose.pose.orientation.w == pytest.approx(math.sqrt(0.5))


def test_pose_timestamp_interpolation_rejects_unbounded_samples():
    early = _odometry("odom", "base_link", 0.0, 0.0, 0.0, 0)
    late = _odometry(
        "odom", "base_link", 10.0, 0.0, 0.0, 2_000_000_000
    )
    samples = [(0, early), (2_000_000_000, late)]
    assert interpolate_pose_by_stamp(samples, 0, 1) == (0, early)
    assert interpolate_pose_by_stamp(samples, -1, 2_000_000_000) is None
    assert interpolate_pose_by_stamp(samples, 1_000_000_000, 999_999_999) is None
    assert interpolate_pose_by_stamp([], 0, 1) is None


def test_path_jump_limit_allows_normal_gap_but_rejects_spike():
    assert path_jump_limit_m(5.0, 20.0, 0.1) == pytest.approx(5.0)
    assert path_jump_limit_m(5.0, 20.0, 0.4) == pytest.approx(8.0)
    assert path_jump_limit_m(5.0, 20.0, -1.0) == pytest.approx(5.0)


def test_lidar_path_proximity_requires_the_same_map_frame():
    state = _odometry("map", "base_link", 10.0, 20.0, 0.0)
    lidar = _odometry("map", "base_link", 13.0, 24.0, 0.0)
    assert same_frame_pose_distance_xy(state, lidar) == pytest.approx(5.0)

    lidar.header.frame_id = "odom"
    assert same_frame_pose_distance_xy(state, lidar) is None


def test_localization_state_remains_usable_before_any_lidar_pose():
    node = MapOdomTfNode.__new__(MapOdomTfNode)
    node._lidar_poses = []
    node._require_lidar_pose_for_map_odom = False

    assert node._broadcast_map_odom(
        _odometry("map", "base_link", 10.0, 20.0, 0.0)
    )


def test_vehicle_marker_is_frame_locked():
    """Otherwise it twitches at the pose rate in a vehicle-frame view."""

    marker = vehicle_marker_from_odometry(
        _odometry("map", "base_link", 1.0, 2.0, 0.0), 2.5, 1.2, 0.8, 0.4
    )
    assert marker.frame_locked


def test_vehicle_marker_is_a_blue_cuboid_at_fused_pose():
    message = _odometry("map", "base_link", 3.0, -2.0, 0.5, 123)
    message.pose.pose.position.z = 0.25

    marker = vehicle_marker_from_odometry(message, 2.5, 1.2, 0.8, 0.4)

    assert marker.header == message.header
    assert marker.ns == "vehicle"
    assert marker.type == marker.CUBE
    assert marker.pose.position.x == pytest.approx(3.0)
    assert marker.pose.position.z == pytest.approx(0.65)
    assert marker.scale.x == pytest.approx(2.5)
    assert marker.scale.y == pytest.approx(1.2)
    assert marker.scale.z == pytest.approx(0.8)
    # Blue, and darker than the Driven Path (0;170;255) it sits on top of.
    assert (marker.color.r, marker.color.g, marker.color.b, marker.color.a) == (
        pytest.approx(0.0),
        pytest.approx(0.24),
        pytest.approx(1.0),
        pytest.approx(0.9),
    )
    assert marker.color.b > marker.color.g > marker.color.r
    assert message.pose.pose.position.z == pytest.approx(0.25)


def test_vehicle_marker_colour_is_overridable_and_validated():
    message = _odometry("map", "base_link", 0.0, 0.0, 0.0, 1)

    marker = vehicle_marker_from_odometry(message, 2.5, 1.2, 0.8, 0.4, (1.0, 0.5, 0.0, 1.0))
    assert (marker.color.r, marker.color.g, marker.color.b) == (
        pytest.approx(1.0),
        pytest.approx(0.5),
        pytest.approx(0.0),
    )

    for bad in ((0.0, 0.0, 1.0), (0.0, 0.0, 1.5, 1.0), (0.0, 0.0, float("nan"), 1.0)):
        with pytest.raises(ValueError, match="colour"):
            vehicle_marker_from_odometry(message, 2.5, 1.2, 0.8, 0.4, bad)
