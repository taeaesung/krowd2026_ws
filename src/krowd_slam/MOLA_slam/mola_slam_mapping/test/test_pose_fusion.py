import math
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

from mola_slam_mapping.pose_fusion_node import (
    PoseFusionNode,
    attach_history_pose,
    blend_poses,
    lidar_increment_disagrees,
    pose_distance_m,
    pose_from_matrix,
    propagate_reference,
    reference_is_a_position_spike,
    rotation_difference_deg,
)


def _pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    half = math.radians(yaw_deg) / 2.0
    pose.orientation.z = math.sin(half)
    pose.orientation.w = math.cos(half)
    return pose


def _odometry(pose: Pose, stamp_ns: int = 0) -> Odometry:
    message = Odometry()
    message.header.stamp.sec = stamp_ns // 1_000_000_000
    message.header.stamp.nanosec = stamp_ns % 1_000_000_000
    message.pose.pose = pose
    return message


def _fusion_node(fused: Pose, **overrides):
    """A PoseFusionNode reduced to the state _correct() actually touches."""

    node = SimpleNamespace(
        _fused=fused,
        _latest_lidar=_odometry(_pose()),
        _filter_weight=0.3,
        _max_step_m=5.0,
        _max_step_deg=20.0,
        _max_consecutive_rejections=5,
        _max_reference_speed_mps=25.0,
        _consecutive_rejections=0,
        _corrections_applied=0,
        _corrections_rejected=0,
        _references_discarded=0,
        _reanchors=0,
        _lidar_overrides=0,
        _last_correction_error_m=0.0,
        _last_reference=None,
        _last_matched_lidar=None,
        _increment_mismatch_run=0,
        _mismatch_min_m=0.15,
        _mismatch_ratio=0.5,
        _mismatch_samples=3,
        _stamp_ns=PoseFusionNode._stamp_ns,
        _backfill_history=lambda: None,
        get_logger=lambda: SimpleNamespace(
            warning=lambda *a, **k: None, warn=lambda *a, **k: None
        ),
    )
    node.__dict__.update(overrides)
    return node


def test_pose_from_matrix_round_trips_translation_and_yaw():
    yaw = math.radians(30.0)
    matrix = [
        [math.cos(yaw), -math.sin(yaw), 0.0, 4.0],
        [math.sin(yaw), math.cos(yaw), 0.0, -2.0],
        [0.0, 0.0, 1.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
    pose = pose_from_matrix(matrix)

    assert pose.position.x == pytest.approx(4.0)
    assert pose.position.y == pytest.approx(-2.0)
    assert pose.position.z == pytest.approx(1.5)
    assert rotation_difference_deg(pose, _pose(yaw_deg=30.0)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_blend_moves_the_configured_fraction_towards_the_measurement():
    blended = blend_poses(_pose(0.0, 0.0, yaw_deg=0.0), _pose(10.0, 0.0, yaw_deg=40.0), 0.25)

    assert blended.position.x == pytest.approx(2.5)
    assert rotation_difference_deg(blended, _pose(yaw_deg=10.0)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_blend_of_one_adopts_the_measurement_outright():
    """The first correction has nothing to blend with and must not be damped."""

    measured = _pose(3.0, -4.0, 1.0, yaw_deg=90.0)
    blended = blend_poses(_pose(), measured, 1.0)

    assert pose_distance_m(blended, measured) == pytest.approx(0.0, abs=1e-9)
    assert rotation_difference_deg(blended, measured) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.5])
def test_blend_rejects_weights_outside_the_unit_interval(weight):
    with pytest.raises(ValueError):
        blend_poses(_pose(), _pose(1.0), weight)


def test_repeated_blending_converges_on_a_steady_correction():
    """A constant drift must be tracked, not merely damped towards forever."""

    correction = _pose()
    target = _pose(6.0, 0.0, yaw_deg=20.0)
    for _ in range(200):
        correction = blend_poses(correction, target, 0.05)

    assert pose_distance_m(correction, target) == pytest.approx(0.0, abs=1e-3)


def test_distance_and_rotation_metrics_detect_an_outlier_step():
    small = _pose(0.1, 0.0, yaw_deg=1.0)
    spike = _pose(9.0, 0.0, yaw_deg=45.0)

    assert pose_distance_m(_pose(), small) < 5.0
    assert rotation_difference_deg(_pose(), small) < 20.0
    assert pose_distance_m(_pose(), spike) > 5.0
    assert rotation_difference_deg(_pose(), spike) > 20.0


def test_propagate_carries_the_reference_along_the_lidar_increment():
    """A 5 Hz reference describes a past instant; it must be brought to now."""

    reference = _pose(100.0, 50.0, yaw_deg=0.0)
    matched = _pose(10.0, 0.0, yaw_deg=0.0)
    latest = _pose(12.0, 0.0, yaw_deg=0.0)

    propagated = propagate_reference(reference, matched, latest)

    assert propagated.position.x == pytest.approx(102.0)
    assert propagated.position.y == pytest.approx(50.0)


def test_propagate_applies_the_increment_in_the_reference_frame():
    """The increment is a body motion, so the reference heading rotates it."""

    # Reference is heading +90 deg (along +y); LiDAR moved 2 m along its own +x.
    reference = _pose(0.0, 0.0, yaw_deg=90.0)
    matched = _pose(10.0, 0.0, yaw_deg=0.0)
    latest = _pose(12.0, 0.0, yaw_deg=0.0)

    propagated = propagate_reference(reference, matched, latest)

    assert propagated.position.x == pytest.approx(0.0, abs=1e-9)
    assert propagated.position.y == pytest.approx(2.0)


def test_propagation_is_immune_to_the_odom_origin_distance():
    """The lever-arm defect: an attitude error must not scale with position.

    Two LiDAR samples describing the same 2 m motion produce the same fused
    result whether they sit 10 m or 500 m from the odometry origin. The earlier
    implementation multiplied a filtered rotation by the LiDAR position itself,
    so the far pair leaked metres of error into z.
    """

    reference = _pose(0.0, 0.0, yaw_deg=30.0)
    near = propagate_reference(reference, _pose(10.0, 0.0), _pose(12.0, 0.0))
    far = propagate_reference(reference, _pose(500.0, 0.0), _pose(502.0, 0.0))

    assert pose_distance_m(near, far) == pytest.approx(0.0, abs=1e-9)


def test_history_is_attached_rigidly_to_the_first_correction():
    """Poses from before the first reference land where the first fix says."""

    history = _pose(10.0, 0.0)
    lidar_latest = _pose(12.0, 0.0)
    fused_latest = _pose(102.0, 50.0)

    attached = attach_history_pose(history, lidar_latest, fused_latest)

    # 2 m behind the latest sample in LiDAR terms stays 2 m behind it in fused terms.
    assert attached.position.x == pytest.approx(100.0)
    assert attached.position.y == pytest.approx(50.0)


def test_history_attachment_rotates_with_the_first_fix():
    history = _pose(10.0, 0.0)
    lidar_latest = _pose(12.0, 0.0)
    fused_latest = _pose(102.0, 50.0, yaw_deg=90.0)

    attached = attach_history_pose(history, lidar_latest, fused_latest)

    # "2 m behind along +x" in LiDAR terms is "2 m behind along +y" once the
    # first fix says the vehicle heads +y: (102, 50) - (0, 2).
    assert attached.position.x == pytest.approx(102.0, abs=1e-9)
    assert attached.position.y == pytest.approx(48.0)
    assert rotation_difference_deg(attached, fused_latest) == pytest.approx(0.0, abs=1e-9)


def test_reference_spike_is_judged_on_the_reference_stream_alone():
    """5 m between two 5 Hz samples is 25 m/s: a bad fix, not a fast vehicle."""

    previous = (0, _pose(0.0, 0.0))
    normal = (200_000_000, _pose(1.0, 0.0))
    spike = (200_000_000, _pose(8.0, 0.0))

    assert not reference_is_a_position_spike(previous, normal, 25.0)
    assert reference_is_a_position_spike(previous, spike, 25.0)
    # Nothing to compare against yet, and no time elapsed, are both "not a spike".
    assert not reference_is_a_position_spike(None, spike, 25.0)
    assert not reference_is_a_position_spike(previous, (0, _pose(8.0)), 25.0)


def test_isolated_outlier_is_rejected_without_re_anchoring():
    node = _fusion_node(_pose(0.0, 0.0))

    PoseFusionNode._correct(node, _odometry(_pose(50.0, 0.0), 0), _odometry(_pose()))

    assert node._corrections_rejected == 1
    assert node._reanchors == 0
    assert node._fused.position.x == pytest.approx(0.0)


def test_sustained_disagreement_re_anchors_the_estimate_on_the_reference():
    """The 0902_4 failure: LiDAR jumped 19 m and every correction was rejected.

    A gate with no way back left the estimate on the broken trajectory for the
    rest of the run. After a second of self-consistent references saying
    otherwise, the estimate - not the reference - is what must give way.
    LiDAR keeps covering the same distance as the reference here (it tracks
    again after the jump), so the increment comparison stays quiet and only
    the rejection count can resolve it.
    """

    node = _fusion_node(_pose(0.0, 0.0))
    for index in range(5):
        lidar = _odometry(_pose(0.2 * index, 0.0))
        node._latest_lidar = lidar
        PoseFusionNode._correct(
            node,
            _odometry(_pose(20.0 + 0.2 * index, 0.0), index * 200_000_000),
            lidar,
        )

    assert node._corrections_rejected == 4
    assert node._reanchors == 1
    assert node._lidar_overrides == 0
    assert node._fused.position.x == pytest.approx(20.8)
    # And the estimate is tracking again rather than latched.
    assert node._consecutive_rejections == 0


def test_lidar_increment_disagreement_needs_both_thresholds():
    """Distances covered between two references: LiDAR vs reference."""

    def disagrees(reference_step, lidar_step):
        return lidar_increment_disagrees(
            _pose(0.0), _pose(reference_step), _pose(0.0), _pose(lidar_step), 0.15, 0.5
        )

    assert disagrees(0.8, 0.02)  # LiDAR stopped tracking at speed
    assert disagrees(0.2, 0.0)  # ... or while pulling away from rest
    assert disagrees(0.8, 19.0)  # LiDAR jumped
    assert not disagrees(0.8, 0.75)  # ordinary drift at speed
    assert not disagrees(0.03, 0.0)  # RTK noise at a standstill


def test_stalled_lidar_odometry_makes_the_estimate_follow_the_reference_outright():
    """The testweek_0905_3 failure: LiDAR froze for 74 m and nothing rejected.

    Every correction was small enough to pass the step gate, so the blend
    trailed the vehicle by a steady 1.4-1.9 m and the saved route lagged with
    it. Once the increments have disagreed for three references in a row, the
    reference is adopted outright and the lag is gone.
    """

    node = _fusion_node(_pose(1.0, 0.0))
    lidar = _odometry(_pose())  # frozen
    for index in range(5):
        PoseFusionNode._correct(
            node, _odometry(_pose(1.0 + 0.8 * index, 0.0), index * 200_000_000), lidar
        )

    assert node._lidar_overrides == 2
    assert node._corrections_rejected == 0
    assert node._reanchors == 0
    assert node._fused.position.x == pytest.approx(4.2)


def test_a_single_reference_glitch_does_not_trigger_the_override():
    """One bad sample disagrees twice - out and back - and that is not enough."""

    node = _fusion_node(_pose(0.0, 0.0))
    for index in range(6):
        lidar = _odometry(_pose(0.8 * index, 0.0))
        node._latest_lidar = lidar
        x = 0.8 * index + (0.5 if index == 2 else 0.0)
        PoseFusionNode._correct(node, _odometry(_pose(x, 0.0), index * 200_000_000), lidar)

    assert node._lidar_overrides == 0
    assert node._increment_mismatch_run == 0
    assert node._corrections_applied == 6


def test_a_reference_spike_does_not_count_towards_re_anchoring():
    """Otherwise repeated RTK spikes could drag the estimate onto themselves."""

    node = _fusion_node(_pose(0.0, 0.0))
    lidar = _odometry(_pose())
    PoseFusionNode._correct(node, _odometry(_pose(1.0, 0.0), 0), lidar)
    for index in range(1, 6):
        # 100 m of "motion" per 0.2 s sample: impossible, so never trusted.
        PoseFusionNode._correct(
            node, _odometry(_pose(100.0 * index, 0.0), index * 200_000_000), lidar
        )

    assert node._references_discarded == 5
    assert node._reanchors == 0
    # Only the first, plausible reference moved it: 30% of the way to 1 m.
    assert node._fused.position.x == pytest.approx(0.3)


def test_an_accepted_correction_clears_the_rejection_run():
    node = _fusion_node(_pose(0.0, 0.0))
    lidar = _odometry(_pose())
    PoseFusionNode._correct(node, _odometry(_pose(20.0, 0.0), 0), lidar)
    assert node._consecutive_rejections == 1

    # LiDAR catches up with where the reference always said the vehicle was.
    node._fused = _pose(20.0, 0.0)
    PoseFusionNode._correct(node, _odometry(_pose(21.0, 0.0), 200_000_000), lidar)

    assert node._consecutive_rejections == 0
    assert node._corrections_applied == 1
    assert node._reanchors == 0


def test_rotation_difference_is_sign_independent():
    """q and -q are the same rotation; a naive dot product would report 360 deg."""

    pose = _pose(yaw_deg=10.0)
    negated = Pose()
    negated.orientation.x = -pose.orientation.x
    negated.orientation.y = -pose.orientation.y
    negated.orientation.z = -pose.orientation.z
    negated.orientation.w = -pose.orientation.w

    assert rotation_difference_deg(pose, negated) == pytest.approx(0.0, abs=1e-6)
