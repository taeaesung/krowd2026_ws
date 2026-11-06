import math

import pytest
from nav_msgs.msg import Odometry

from mola_slam_mapping.rtk_initial_pose_node import (
    icp_search_sigma_m,
    initial_pose_from_odometry,
)


def _rtk_pose(x=12.0, y=-4.0, z=0.3, yaw=math.pi / 2.0, frame="map"):
    odometry = Odometry()
    odometry.header.frame_id = frame
    odometry.header.stamp.sec = 7
    odometry.header.stamp.nanosec = 500
    odometry.child_frame_id = "base_link"
    odometry.pose.pose.position.x = x
    odometry.pose.pose.position.y = y
    odometry.pose.pose.position.z = z
    odometry.pose.pose.orientation.z = math.sin(yaw / 2.0)
    odometry.pose.pose.orientation.w = math.cos(yaw / 2.0)
    return odometry


def test_seed_carries_the_rtk_pose_and_the_priors_real_uncertainty():
    """MOLA sizes its ICP search from the covariance, so it must reflect the
    error of the prior as it reaches MOLA (stale by one RTK period, plus the
    georeference), not RTK's own 1.5 cm."""

    seed = initial_pose_from_odometry(_rtk_pose(), 1.0, 0.5, 3.0, 2.0)

    assert seed.header.frame_id == "map"
    assert (seed.header.stamp.sec, seed.header.stamp.nanosec) == (7, 500)
    assert seed.pose.pose.position.x == 12.0
    assert seed.pose.pose.orientation.z == pytest.approx(math.sin(math.pi / 4.0))

    covariance = seed.pose.covariance
    # ROS order x, y, z, roll, pitch, yaw on the diagonal of a 6x6.
    assert covariance[0] == covariance[7] == 1.0
    assert covariance[14] == 0.25
    assert covariance[21] == covariance[28] == pytest.approx(math.radians(2.0) ** 2)
    assert covariance[35] == pytest.approx(math.radians(3.0) ** 2)
    assert all(v == 0.0 for i, v in enumerate(covariance) if i % 7 != 0)

    # What LidarOdometry::relocalize_near_pose_pdf() turns it into. Must stay
    # under lidar3d-dlio-like.yaml's maximum_sigma (2.0 m) with the defaults.
    assert icp_search_sigma_m(seed) == pytest.approx(math.sqrt(2.0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position_sigma_m": 0.0},
        {"z_sigma_m": -1.0},
        {"yaw_sigma_deg": math.nan},
        {"roll_pitch_sigma_deg": math.inf},
    ],
)
def test_degenerate_sigmas_are_refused(kwargs):
    sigmas = {
        "position_sigma_m": 1.0,
        "z_sigma_m": 1.0,
        "yaw_sigma_deg": 3.0,
        "roll_pitch_sigma_deg": 2.0,
        **kwargs,
    }
    with pytest.raises(ValueError):
        initial_pose_from_odometry(_rtk_pose(), **sigmas)


def test_an_unusable_rtk_pose_is_refused_rather_than_sent():
    with pytest.raises(ValueError, match="frame_id"):
        initial_pose_from_odometry(_rtk_pose(frame=""), 1.0, 1.0, 3.0, 2.0)
    with pytest.raises(ValueError, match="non-finite"):
        initial_pose_from_odometry(_rtk_pose(x=math.nan), 1.0, 1.0, 3.0, 2.0)
