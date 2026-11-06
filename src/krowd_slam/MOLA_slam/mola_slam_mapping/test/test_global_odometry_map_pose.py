import math

from nav_msgs.msg import Odometry
import numpy as np
import pytest

from mola_slam_mapping.global_odometry_map_pose_node import (
    map_from_ekf_enu_transform,
    quaternion_from_euler,
    transform_global_odometry,
)


def test_global_odometry_is_projected_through_selected_map_georeference():
    origin = (37.239263, 126.7731816, 40.0)
    enu_from_map = quaternion_from_euler(math.radians(90.0), 0.0, 0.0)
    rotation, translation = map_from_ekf_enu_transform(
        origin,
        origin,
        (10.0, 20.0, 0.0),
        enu_from_map,
    )
    message = Odometry()
    message.header.frame_id = "enu"
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = 10.0
    message.pose.pose.position.y = 21.0
    message.pose.pose.orientation.w = 1.0
    message.pose.covariance[0] = 1.0
    message.pose.covariance[7] = 4.0
    message.pose.covariance[14] = 9.0

    output = transform_global_odometry(message, rotation, translation)

    assert output.header.frame_id == "map"
    assert output.child_frame_id == "base_link"
    np.testing.assert_allclose(
        [
            output.pose.pose.position.x,
            output.pose.pose.position.y,
            output.pose.pose.position.z,
        ],
        [1.0, 0.0, 0.0],
        atol=1.0e-9,
    )
    # The map axes are rotated +90 deg in ENU, so ENU heading zero is -90 in map.
    assert output.pose.pose.orientation.z == pytest.approx(
        -math.sqrt(0.5), abs=1.0e-9
    )
    assert output.pose.pose.orientation.w == pytest.approx(
        math.sqrt(0.5), abs=1.0e-9
    )
    assert output.pose.covariance[0] == pytest.approx(4.0)
    assert output.pose.covariance[7] == pytest.approx(1.0)


def test_different_ekf_origin_is_not_treated_as_the_map_origin():
    map_origin = (37.239263, 126.7731816, 40.0)
    ekf_origin = (37.239363, 126.7732816, 40.0)
    rotation, translation = map_from_ekf_enu_transform(
        ekf_origin,
        map_origin,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert np.linalg.norm(translation[:2]) > 10.0
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12)


def test_wrong_global_odometry_frame_is_rejected():
    message = Odometry()
    message.header.frame_id = "odom"
    message.pose.pose.orientation.w = 1.0

    with pytest.raises(ValueError, match="expected odometry frame"):
        transform_global_odometry(message, np.eye(3), np.zeros(3))
