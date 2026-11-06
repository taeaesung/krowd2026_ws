"""Fuse LiDAR odometry with the RTK/IMU global pose, outside MOLA.

Why this is a separate node rather than MOLA's StateEstimationSmoother: MOLA
accepts exactly one NavStateFilter, and that module also supplies LidarOdometry's
motion-model prior. The smoother's GNSS factors were measured to corrupt that
prior on the 0824_5 bag - ICP then converges confidently onto a wrong rotation
and the saved route degrades (the numbers are in config/mapping_system.yaml).
Fusing here instead keeps the two sensors' complementary strengths without any
path back into the mapping pipeline: this node subscribes only, and nothing in
the SLAM stack subscribes to it.

The two inputs fail in opposite ways, which is what makes fusing them worthwhile:

    /lidar_odometry/pose  odom->base_link.  Excellent over one 0.1 s step, but its
                          origin is arbitrary and it drifts without bound.
    /gnss_imu/pose        map->base_link.   Globally correct to RTK accuracy and
                          drift-free, but noisy sample to sample, only ~5 Hz, and
                          its heading comes from the GNSS ENU course, so it is
                          poor at low speed and meaningless at a standstill.

So LiDAR is used ONLY for increments and RTK supplies the absolute pose - plain
dead reckoning with an absolute correction:

    predict   T_fused <- T_fused * (T_odom_prev^-1 * T_odom_now)
    correct   T_fused <- blend(T_fused, T_reference, correction_filter_weight)

The correction is deliberately applied at the vehicle's current position, so no
estimated rotation is ever multiplied by anything longer than one 0.1 s increment
of about 0.3 m. An earlier version instead low-pass filtered the drift transform
itself and applied it globally,

    T_fused = filtered(T_map_base * T_odom_base^-1) * T_odom_base

which multiplies that filtered rotation by the LiDAR *position* - up to 75 m from
the odom origin here. It is kept out of this file on principle, but note it was
not what actually broke the measured z: see below.

TIMING IS THE PART THAT IS EASY TO GET WRONG. A reference pose is stamped at its
measurement instant, which is normally newer than the newest LiDAR sample buffered
when it arrives, so interpolate_pose_by_stamp() cannot bracket it *yet*. Correcting
only on arrival therefore throws nearly every correction away - measured at 4
applied against 340 dropped, which left this node dead-reckoning almost unaided and
its z swinging 3.9 m against 1.1 m for RTK alone. References are queued and applied
from _on_lidar(), because the new LiDAR sample is what completes the bracket.
map_odom_tf_node's _on_lidar() retries its buffered state poses for the same
reason; any node time-matching these two streams needs that retry.

The result follows RTK globally and LiDAR locally. Note this is deliberately not
what the blue /driven_path shows: that reprojects the whole history with a single
instantaneous correction, so it stays a rigid transform of the LiDAR path and
carries all of its drift. Where the purple and blue lines separate is where LiDAR
odometry has drifted.

WHEN LiDAR ODOMETRY FAILS, THIS MUST FALL BACK ON RTK. Measured on 0902_4:
ICP lost tracking 34 s in and jumped 19 m, after which every correction exceeded
max_correction_step_m and was rejected - 47 applied, then 36 rejected and none
applied for the rest of the run. The estimate had no way back, so the purple line
(and the route saved from it) carried the LiDAR failure to the end, 14 m off RTK.
A gate that compares the estimate with the reference cannot tell a bad reference
from a broken estimate, so it must not be the only word: a reference that passes
its OWN continuity check (reference_is_a_position_spike) and still disagrees for
max_consecutive_rejections samples in a row is evidence that dead reckoning
failed, and the estimate is re-anchored onto it. Degrading to the red line is the
correct failure mode; silently tracking a broken LiDAR is not.

THAT GATE CANNOT SEE A LiDAR ODOMETRY THAT SIMPLY STOPS MOVING. testweek_0905_3:
ICP lost the first 74 m (20 s of driving recorded as 2.3 m). Each correction
then closed only 30 % of a gap that RTK reopened by ~0.8 m per sample, so the
estimate trailed the vehicle by a steady 1.4-1.9 m and, with every step below
max_correction_step_m, the gate never fired. The tell is not the size of the
gap but the *increments*: between two consecutive references RTK moved 0.7 m
and the matched LiDAR poses moved 0.02 m. lidar_increment_disagrees() compares
exactly those two distances; after lidar_increment_mismatch_samples of them in
a row the correction is applied outright (weight 1) instead of blended, so the
estimate - and the route recorded from it - follows the red line through the
loss with no lag. Healthy driving differs by a few centimetres per sample and
never trips it; a single reference glitch cannot either - it disagrees once on
the way out and once on the way back, and three samples must agree.

Since 2026-09-03 this pose is also what waypoint_saver_node records the route and
corridor from (measured 0.26 m from RTK, against 1.4-1.6 m mean / 3.2 m worst for
raw LiDAR odometry). Because the route must start at the run's first sample and
the first reference only lands after the heading alignment, the LiDAR poses seen
before that are published retroactively at the first correction, rigidly attached
to it (attach_history_pose / _backfill_history).
"""

import math

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from rclpy.node import Node

from mola_slam_mapping.map_odom_tf_node import (
    _invert,
    _mat_from_pose,
    _mul,
    _quat_from_mat,
    _slerp,
    _unit_quaternion,
    interpolate_pose_by_stamp,
)


def pose_from_matrix(matrix) -> Pose:
    """Build a Pose from a 4x4 homogeneous transform."""

    pose = Pose()
    pose.position.x = matrix[0][3]
    pose.position.y = matrix[1][3]
    pose.position.z = matrix[2][3]
    (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ) = _quat_from_mat(matrix)
    return pose


def blend_poses(previous: Pose, measured: Pose, weight: float) -> Pose:
    """Move `previous` a `weight` fraction of the way towards `measured`.

    Translation is linear, rotation is SLERP. weight=1 adopts the measurement
    outright, which is what the very first correction must do.
    """

    if not 0.0 < weight <= 1.0:
        raise ValueError("blend weight must be in (0, 1]")

    blended = Pose()
    for axis in ("x", "y", "z"):
        start = getattr(previous.position, axis)
        end = getattr(measured.position, axis)
        setattr(blended.position, axis, start + weight * (end - start))
    quaternion = _slerp(previous.orientation, measured.orientation, weight)
    for axis, value in zip(("x", "y", "z", "w"), quaternion):
        setattr(blended.orientation, axis, value)
    return blended


def propagate_reference(
    reference: Pose, matched_lidar: Pose, latest_lidar: Pose
) -> Pose:
    """Carry a reference pose forward by the LiDAR motion since its timestamp.

    The ~5 Hz reference always describes an already-past instant, so correcting
    the current estimate towards it directly would drag the estimate backwards
    along the path by however far the vehicle moved in between. The LiDAR
    increment over that short interval is the best available description of that
    motion, so it is what carries the reference forward.
    """

    increment = _mul(_invert(_mat_from_pose(matched_lidar)), _mat_from_pose(latest_lidar))
    return pose_from_matrix(_mul(_mat_from_pose(reference), increment))


def attach_history_pose(history: Pose, lidar_latest: Pose, fused_latest: Pose) -> Pose:
    """Express a pre-correction LiDAR pose in the fused frame, rigidly.

    Before the first reference lands there is no fused estimate at all, only
    LiDAR odometry. Once it lands, the transform that carries the latest LiDAR
    pose onto the latest fused pose is the best available placement for the
    poses recorded before it - over the first few metres LiDAR drift is
    negligible, so a single rigid transform is exact enough.
    """

    attach = _mul(_mat_from_pose(fused_latest), _invert(_mat_from_pose(lidar_latest)))
    return pose_from_matrix(_mul(attach, _mat_from_pose(history)))


def pose_distance_m(first: Pose, second: Pose) -> float:
    return math.dist(
        (first.position.x, first.position.y, first.position.z),
        (second.position.x, second.position.y, second.position.z),
    )


def reference_is_a_position_spike(previous, current, max_speed_mps: float) -> bool:
    """Did the reference stream jump, as opposed to the estimate drifting away?

    Both arguments are (stamp_ns, Pose) of consecutive accepted references, or
    None before the first one. An RTK multipath spike moves the reference itself
    between two samples 0.2 s apart; no vehicle here covers 5 m in that time, so
    an implausible sample-to-sample speed is the signature of a bad fix. This is
    the only evidence that actually distinguishes a bad reference from a broken
    estimate, so only references failing it may be discarded outright.
    """

    if previous is None or current is None:
        return False
    previous_stamp, previous_pose = previous
    stamp, pose = current
    if stamp <= previous_stamp:
        return False
    interval_sec = (stamp - previous_stamp) * 1e-9
    return pose_distance_m(previous_pose, pose) / interval_sec > max_speed_mps


def lidar_increment_disagrees(
    previous_reference: Pose,
    reference: Pose,
    previous_lidar: Pose,
    lidar: Pose,
    min_mismatch_m: float,
    min_mismatch_ratio: float,
) -> bool:
    """Did LiDAR odometry move a different distance than the reference did?

    Both pairs span the same two reference instants (the LiDAR poses are the
    ones time-matched to those references). A LiDAR odometry that has stopped
    tracking reports ~0 m while RTK reports the real motion; a jump reports far
    more. Judged on distances, not directions, so a heading disagreement between
    the two frames does not count. Both thresholds must be exceeded: the
    absolute one keeps RTK noise at a standstill out, the relative one keeps a
    few centimetres of ordinary drift at speed out.
    """

    reference_step = pose_distance_m(previous_reference, reference)
    lidar_step = pose_distance_m(previous_lidar, lidar)
    mismatch = abs(reference_step - lidar_step)
    return mismatch > min_mismatch_m and mismatch > min_mismatch_ratio * reference_step


def rotation_difference_deg(first: Pose, second: Pose) -> float:
    """Absolute rotation angle between two orientations, in degrees."""

    a = _unit_quaternion(first.orientation)
    b = _unit_quaternion(second.orientation)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


class PoseFusionNode(Node):
    """Publish the fused global vehicle pose used by the purple RViz overlay."""

    def __init__(self):
        super().__init__("pose_fusion_node")

        self.declare_parameter("lidar_pose_topic", "/lidar_odometry/pose")
        self.declare_parameter("reference_pose_topic", "/gnss_imu/pose")
        self.declare_parameter("output_topic", "/fused_localization/pose")
        # Fraction of the gap to the reference that each correction closes, i.e.
        # how much the estimate trusts dead reckoning between references.
        #
        # Measured on 0824_5 (mean | max deviation from the reference, over the
        # whole run), which is what set the default:
        #     weight   z error          horizontal error   standing gap
        #     0.05     0.59 | 1.03 m    0.53 | 1.08 m      0.7-1.0 m
        #     0.30     0.14 | 0.35 m    0.23 | 0.72 m      0.08-0.11 m
        #     0.60     0.06 | 0.24 m    0.15 | 0.60 m      0.02-0.03 m
        # 0.05 is far too slow: this LiDAR odometry dead-reckons at roughly
        # 0.2 m/s of drift, so a 4 s time constant leaves most of a metre standing.
        # 0.30 removes ~75% of that error; 0.60 removes only ~0.08 m more while
        # halving how much reference noise gets averaged away, which is the wrong
        # trade the moment RTK is not 100% fixed. Per-sample smoothness is the same
        # either way (0.027 vs 0.028 m mean |dz|).
        self.declare_parameter("correction_filter_weight", 0.3)
        # An RTK multipath spike or an ICP failure shows up as a step between the
        # dead-reckoned estimate and the reference. Reject those outright instead
        # of letting the filter drag the whole trajectory towards them.
        self.declare_parameter("max_correction_step_m", 5.0)
        self.declare_parameter("max_correction_step_deg", 20.0)
        # ... but never forever: that gate alone cannot distinguish a spurious
        # reference from a LiDAR odometry failure, and treating every large step
        # as the former locks the estimate onto a broken trajectory permanently
        # (see the module docstring). This many consecutive rejections - one
        # second at the 5 Hz reference rate - is taken as proof that the estimate
        # is the broken one, and it is re-anchored onto the reference.
        self.declare_parameter("max_consecutive_rejections", 5)
        # Sample-to-sample speed above which the reference itself is the outlier.
        # The vehicle never approaches 25 m/s, so nothing legitimate is discarded,
        # while a 5 m spike between two 5 Hz samples reads as 25 m/s.
        self.declare_parameter("max_reference_speed_mps", 25.0)
        self.declare_parameter("max_pose_time_difference_sec", 0.15)
        # A LiDAR odometry that stops moving (or jumps) while the reference keeps
        # moving is caught by comparing the two distances covered between
        # consecutive references - see lidar_increment_disagrees() and the module
        # docstring. 0.15 m and 50 % of the reference step must both be exceeded,
        # on this many references in a row (0.6 s at 5 Hz), before corrections
        # are applied with weight 1 instead of correction_filter_weight. Three,
        # not two: one bad reference sample disagrees twice, out and back.
        self.declare_parameter("lidar_increment_mismatch_m", 0.15)
        self.declare_parameter("lidar_increment_mismatch_ratio", 0.5)
        self.declare_parameter("lidar_increment_mismatch_samples", 3)

        self._filter_weight = float(
            self.get_parameter("correction_filter_weight").value
        )
        if not 0.0 < self._filter_weight <= 1.0:
            raise ValueError("correction_filter_weight must be in (0, 1]")
        self._max_step_m = float(self.get_parameter("max_correction_step_m").value)
        self._max_step_deg = float(
            self.get_parameter("max_correction_step_deg").value
        )
        self._max_consecutive_rejections = int(
            self.get_parameter("max_consecutive_rejections").value
        )
        if self._max_consecutive_rejections < 1:
            raise ValueError("max_consecutive_rejections must be at least 1")
        self._max_reference_speed_mps = float(
            self.get_parameter("max_reference_speed_mps").value
        )
        self._max_pose_time_difference_ns = int(
            round(
                float(self.get_parameter("max_pose_time_difference_sec").value) * 1e9
            )
        )
        self._mismatch_min_m = float(
            self.get_parameter("lidar_increment_mismatch_m").value
        )
        self._mismatch_ratio = float(
            self.get_parameter("lidar_increment_mismatch_ratio").value
        )
        self._mismatch_samples = int(
            self.get_parameter("lidar_increment_mismatch_samples").value
        )
        if self._mismatch_samples < 1:
            raise ValueError("lidar_increment_mismatch_samples must be at least 1")

        self._lidar_poses = []
        # Every LiDAR pose received before the first correction. waypoint_saver_node
        # keeps the route from the very first sample, and the first reference only
        # lands after gnss_imu_pose_node's ~5 m heading alignment (23 m on a slow,
        # curving start), so those poses are published retroactively once the
        # first correction anchors them; see _backfill_history().
        self._history: list[Odometry] = []
        # A reference pose is timestamped at its measurement instant, which is
        # normally NEWER than the newest LiDAR sample buffered when it arrives -
        # so it cannot be bracketed yet, and correcting on arrival alone throws
        # almost every correction away (measured: 4 applied against 340 dropped).
        # References are therefore held until a later LiDAR sample brackets them.
        self._pending_references = []
        # The fused estimate and the LiDAR sample it is anchored to. These two are
        # always advanced together, so a correction can be propagated to exactly
        # the instant the estimate describes.
        self._fused: Pose | None = None
        self._latest_lidar: Odometry | None = None
        self._reference_frame: str | None = None
        self._published = False
        # The last reference accepted as self-consistent, as (stamp_ns, Pose).
        # Its only job is to judge the next one; see reference_is_a_position_spike.
        self._last_reference: tuple[int, Pose] | None = None
        # The LiDAR pose matched to that reference: the other end of the
        # increment comparison in lidar_increment_disagrees().
        self._last_matched_lidar: Pose | None = None
        self._increment_mismatch_run = 0
        self._consecutive_rejections = 0
        # A correction that silently never lands looks exactly like a correction
        # that lands and is outvoted by drift, so the two are counted apart.
        self._corrections_applied = 0
        self._corrections_unmatched = 0
        self._corrections_rejected = 0
        self._references_discarded = 0
        self._reanchors = 0
        self._lidar_overrides = 0
        self._last_correction_error_m = 0.0
        self.create_timer(10.0, self._report_correction_health)

        self._publisher = self.create_publisher(
            Odometry, self.get_parameter("output_topic").value, 20
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("lidar_pose_topic").value,
            self._on_lidar,
            20,
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("reference_pose_topic").value,
            self._on_reference,
            20,
        )

        self.get_logger().info(
            "pose_fusion_node: dead-reckoning "
            f"'{self.get_parameter('lidar_pose_topic').value}' increments, "
            f"corrected towards '{self.get_parameter('reference_pose_topic').value}' "
            f"-> '{self.get_parameter('output_topic').value}' "
            f"(correction weight {self._filter_weight})"
        )

    @staticmethod
    def _stamp_ns(msg: Odometry) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _reset(self) -> None:
        self._lidar_poses.clear()
        self._history.clear()
        self._pending_references.clear()
        self._fused = None
        self._latest_lidar = None
        self._last_reference = None
        self._last_matched_lidar = None
        self._increment_mismatch_run = 0
        self._consecutive_rejections = 0

    def _on_lidar(self, msg: Odometry) -> None:
        """Advance the estimate by one LiDAR increment and publish it."""

        if not msg.header.frame_id.strip():
            return
        previous = self._latest_lidar
        if previous is not None and (
            self._stamp_ns(msg) < self._stamp_ns(previous)
            or msg.header.frame_id != previous.header.frame_id
        ):
            self.get_logger().warning(
                "LiDAR odometry restarted; restarting the fused trajectory."
            )
            self._reset()
            previous = None

        self._lidar_poses.append((self._stamp_ns(msg), msg))
        # Two seconds at the ~10 Hz LiDAR rate is ample to bracket a reference
        # pose that arrives slightly late, and bounds the sort in
        # interpolate_pose_by_stamp().
        if len(self._lidar_poses) > 20:
            del self._lidar_poses[0]
        if self._fused is None:
            self._history.append(msg)
            # ~40 min at the LiDAR rate; a run that long without a single
            # reference has a bigger problem than a truncated route start.
            if len(self._history) > 20000:
                del self._history[0]

        if self._fused is not None and previous is not None:
            # Only this 0.1 s increment - never the pose itself - is transformed
            # by an estimated rotation. See the module docstring.
            self._fused = pose_from_matrix(
                _mul(
                    _mat_from_pose(self._fused),
                    _mul(
                        _invert(_mat_from_pose(previous.pose.pose)),
                        _mat_from_pose(msg.pose.pose),
                    ),
                )
            )
        self._latest_lidar = msg

        # This new sample is what brackets the references that arrived since the
        # previous one, so corrections are applied here rather than on arrival.
        self._apply_pending_references()

        if self._fused is None:
            self.get_logger().warning(
                "No fused pose yet: waiting for the first "
                f"'{self.get_parameter('reference_pose_topic').value}' that can "
                "be time-matched to a LiDAR pose.",
                throttle_duration_sec=10.0,
            )
            return
        self._publish(msg)

    def _on_reference(self, msg: Odometry) -> None:
        """Queue one global reference pose for the next possible correction."""

        reference_frame = msg.header.frame_id.strip()
        if not reference_frame:
            return
        if (
            self._reference_frame is not None
            and reference_frame != self._reference_frame
        ):
            self.get_logger().warning(
                f"Reference frame changed '{self._reference_frame}' -> "
                f"'{reference_frame}'; restarting the fused trajectory."
            )
            self._reset()
        self._reference_frame = reference_frame

        self._pending_references.append((self._stamp_ns(msg), msg))
        if len(self._pending_references) > 10:
            del self._pending_references[0]
        self._apply_pending_references()

    def _apply_pending_references(self) -> None:
        """Correct with every queued reference the LiDAR buffer can now bracket.

        A reference newer than the newest LiDAR sample stays queued; one older
        than the oldest is unbracketable forever and is dropped.
        """

        if self._latest_lidar is None or not self._pending_references:
            return
        if self._latest_lidar.header.frame_id.strip() == self._reference_frame:
            # In localization mode LiDAR odometry already publishes in map, so
            # there is no drift to correct and no separate fused estimate to make.
            self._pending_references.clear()
            return

        newest_lidar_stamp = self._stamp_ns(self._latest_lidar)
        still_pending = []
        for stamp_ns, reference in self._pending_references:
            try:
                matched = interpolate_pose_by_stamp(
                    self._lidar_poses,
                    stamp_ns,
                    self._max_pose_time_difference_ns,
                )
            except ValueError as error:
                self.get_logger().warn(
                    f"Not correcting the fused pose: {error}",
                    throttle_duration_sec=5.0,
                )
                continue
            if matched is None:
                if stamp_ns > newest_lidar_stamp:
                    still_pending.append((stamp_ns, reference))
                else:
                    # Already inside the buffered span, so no later sample can
                    # help: either too far from any sample or already aged out.
                    self._corrections_unmatched += 1
                continue
            self._correct(reference, matched[1])
        self._pending_references = still_pending

    def _correct(self, reference: Odometry, matched_lidar: Odometry) -> None:
        current = (self._stamp_ns(reference), reference.pose.pose)
        if reference_is_a_position_spike(
            self._last_reference, current, self._max_reference_speed_mps
        ):
            # Judged on the reference stream alone, so a broken estimate can
            # never be mistaken for this case and silence its own corrections.
            self._references_discarded += 1
            self.get_logger().warning(
                "Discarded a reference that moved faster than "
                f"{self._max_reference_speed_mps:.0f} m/s since the previous one",
                throttle_duration_sec=5.0,
            )
            return
        previous_reference = self._last_reference
        previous_lidar = self._last_matched_lidar
        self._last_reference = current
        self._last_matched_lidar = matched_lidar.pose.pose

        target = propagate_reference(
            reference.pose.pose,
            matched_lidar.pose.pose,
            self._latest_lidar.pose.pose,
        )

        if self._fused is None:
            self._fused = target
            self._corrections_applied += 1
            self._backfill_history()
            return

        step_m = pose_distance_m(self._fused, target)
        step_deg = rotation_difference_deg(self._fused, target)

        if previous_reference is not None and previous_lidar is not None and (
            lidar_increment_disagrees(
                previous_reference[1],
                reference.pose.pose,
                previous_lidar,
                matched_lidar.pose.pose,
                self._mismatch_min_m,
                self._mismatch_ratio,
            )
        ):
            self._increment_mismatch_run += 1
        else:
            self._increment_mismatch_run = 0
        if self._increment_mismatch_run >= self._mismatch_samples:
            # LiDAR odometry is not covering the distance the reference is: it
            # has stopped tracking (or jumped). Its increments are worthless
            # right now, so the reference is adopted outright rather than
            # blended, and the step gate below - which exists for a bad
            # reference - does not apply.
            self._fused = target
            self._consecutive_rejections = 0
            self._lidar_overrides += 1
            self._corrections_applied += 1
            self._last_correction_error_m = step_m
            self.get_logger().warning(
                "LiDAR odometry increments disagree with the reference for "
                f"{self._increment_mismatch_run} samples: following the reference "
                f"outright ({step_m:.2f} m / {step_deg:.1f} deg).",
                throttle_duration_sec=5.0,
            )
            return

        if step_m > self._max_step_m or step_deg > self._max_step_deg:
            self._consecutive_rejections += 1
            if self._consecutive_rejections < self._max_consecutive_rejections:
                self._corrections_rejected += 1
                self.get_logger().warning(
                    f"Rejected implausible correction: {step_m:.1f} m / "
                    f"{step_deg:.1f} deg (limits {self._max_step_m:.1f} m / "
                    f"{self._max_step_deg:.1f} deg)"
                )
                return
            # Self-consistent references have disagreed with the estimate for a
            # whole second, so it is the estimate that is wrong: dead reckoning
            # has lost the vehicle and only the reference still knows where it is.
            self._fused = target
            self._consecutive_rejections = 0
            self._reanchors += 1
            self._corrections_applied += 1
            self._last_correction_error_m = step_m
            self.get_logger().warning(
                f"LiDAR odometry appears to have failed: re-anchored the fused "
                f"pose {step_m:.1f} m / {step_deg:.1f} deg onto the reference "
                f"after {self._max_consecutive_rejections} consecutive "
                "rejections. The trajectory before this point is unreliable."
            )
            return

        self._consecutive_rejections = 0
        self._corrections_applied += 1
        self._last_correction_error_m = step_m
        self._fused = blend_poses(self._fused, target, self._filter_weight)

    def _report_correction_health(self) -> None:
        """Report whether corrections are landing, and how far they have to reach.

        A standing error much larger than one correction step means the filter is
        tracking a drift it cannot keep up with, not that it is idle.
        """

        if not self._corrections_applied and not self._corrections_unmatched:
            return
        self.get_logger().info(
            "fusion: %d corrections applied, %d unmatched, %d rejected, "
            "%d references discarded, %d re-anchors, %d LiDAR-increment overrides; "
            "latest gap to reference %.2f m"
            % (
                self._corrections_applied,
                self._corrections_unmatched,
                self._corrections_rejected,
                self._references_discarded,
                self._reanchors,
                self._lidar_overrides,
                self._last_correction_error_m,
            )
        )

    def _backfill_history(self) -> None:
        """Publish the poses from before the first correction, now anchored.

        Emitted oldest first, all older than the current LiDAR sample, which the
        normal flow publishes next - so consumers see one monotonic stream that
        starts at the run's first pose instead of at the first reference.
        """

        latest_stamp = self._stamp_ns(self._latest_lidar)
        count = 0
        for msg in self._history:
            if self._stamp_ns(msg) >= latest_stamp:
                continue
            self._publish_pose(
                attach_history_pose(
                    msg.pose.pose, self._latest_lidar.pose.pose, self._fused
                ),
                msg,
            )
            count += 1
        self._history.clear()
        if count:
            self.get_logger().info(
                f"Back-filled {count} fused poses from the start of the run, "
                "rigidly attached at the first correction."
            )

    def _publish(self, lidar_msg: Odometry) -> None:
        self._publish_pose(self._fused, lidar_msg)

    def _publish_pose(self, pose: Pose, lidar_msg: Odometry) -> None:
        fused = Odometry()
        fused.header.stamp = lidar_msg.header.stamp
        fused.header.frame_id = self._reference_frame
        fused.child_frame_id = lidar_msg.child_frame_id
        fused.pose.pose = pose
        self._publisher.publish(fused)
        if not self._published:
            self._published = True
            self.get_logger().info(
                f"First fused pose published in frame '{self._reference_frame}'."
            )


def main(args=None):
    rclpy.init(args=args)
    node = PoseFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
