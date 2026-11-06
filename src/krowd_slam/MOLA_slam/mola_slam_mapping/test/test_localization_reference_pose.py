from pathlib import Path

from nav_msgs.msg import Odometry

from mola_slam_mapping.localization_reference_pose_node import VerifiedPoseSelector
from mola_slam_mapping.prepare_lidar_localization_map import exported_layer_path


def _pose(x: float) -> Odometry:
    message = Odometry()
    message.pose.pose.position.x = x
    return message


def test_primary_pose_is_bounded_until_heading_is_verified():
    selector = VerifiedPoseSelector()
    candidate = _pose(1.0)

    assert selector.primary(candidate) is None
    assert selector.verification(False) is None
    assert selector.verification(True) is candidate

    next_candidate = _pose(2.0)
    assert selector.primary(next_candidate) is next_candidate
    assert selector.verification(True) is None


def test_drive_based_pose_is_only_a_pre_verification_fallback():
    selector = VerifiedPoseSelector()
    fallback = _pose(3.0)

    assert selector.fallback(fallback) is fallback
    selector.verification(True)
    assert selector.fallback(fallback) is None


def test_exported_ply_name_matches_mm2ply_layer_naming():
    metric_map = Path("/maps/run/global_map_corrected.mm")
    assert exported_layer_path(metric_map, "localmap") == Path(
        "/maps/run/global_map_corrected_localmap.ply"
    )
