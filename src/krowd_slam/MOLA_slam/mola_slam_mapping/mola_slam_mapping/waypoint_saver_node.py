import bisect
import csv
import math
import os
import sys
import time as wall_time
from collections import deque
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transform support
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from mrpt_nav_interfaces.msg import GeoreferencingMetadata
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import TransformException
from visualization_msgs.msg import MarkerArray

from mola_slam_mapping.route_visualization import (
    RoutePoint,
    build_route_marker_array,
    make_route_point,
    route_label_indices,
)
from mola_slam_mapping.traffic_detect_zones import (
    ZONE_KINDS,
    template_may_be_rewritten,
    zone_template_yaml,
)

# UTM 좌표계로 변환된 차량 pose의 quaternion orientation에서
# 평면상 진행 방향인 yaw(rad)를 추출하는 함수입니다.


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_from_quaternion(q) -> tuple[float, float, float]:
    """Return yaw, pitch, roll from a normalized-or-normalizable quaternion."""

    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm == 0.0:
        x = y = z = 0.0
        w = 1.0
    else:
        x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return yaw, pitch, roll


def _pose_to_dict(pose) -> dict:
    return {
        "translation": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "rotation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def _transform_to_dict(transform) -> dict:
    return {
        "source_frame": transform.child_frame_id,
        "target_frame": transform.header.frame_id,
        "source_to_target": {
            "translation": {
                "x": transform.transform.translation.x,
                "y": transform.transform.translation.y,
                "z": transform.transform.translation.z,
            },
            "rotation": {
                "x": transform.transform.rotation.x,
                "y": transform.transform.rotation.y,
                "z": transform.transform.rotation.z,
                "w": transform.transform.rotation.w,
            },
        },
        "stamp": {
            "sec": transform.header.stamp.sec,
            "nanosec": transform.header.stamp.nanosec,
        },
    }


def _yaml_builtin_scalars(value):
    """Recursively convert NumPy/ROS numeric scalars for PyYAML SafeDumper."""

    if isinstance(value, dict):
        return {key: _yaml_builtin_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_builtin_scalars(item) for item in value]
    # bool is an Integral, so preserve it before applying the numeric conversions.
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return value


def _mrpt_diagonal_covariance(diagonal: float) -> dict:
    """Return the map schema consumed by mrpt::containers::yaml::toMatrix()."""

    return {
        "rows": 6,
        "cols": 6,
        "data": [
            diagonal if row == column else 0.0
            for row in range(6)
            for column in range(6)
        ],
    }


def waypoint_step_distance_3d(previous: RoutePoint, current: RoutePoint) -> float:
    """Euclidean distance used to reject isolated estimator pose spikes."""

    return math.sqrt(
        (current.x - previous.x) ** 2
        + (current.y - previous.y) ** 2
        + (current.z - previous.z) ** 2
    )


def waypoint_jump_limit_m(
    base_jump_m: float, max_speed_mps: float, elapsed_sec: float
) -> float:
    """Allow longer steps after a real pose-publication gap."""

    return max(base_jump_m, max_speed_mps * max(0.0, elapsed_sec))


def angle_difference(first: float, second: float) -> float:
    '''Return the shortest signed planar angular difference.'''

    return math.atan2(math.sin(first - second), math.cos(first - second))


def circular_yaw_spread(yaws: list[float]) -> float:
    '''Return the small-arc spread, including samples around the pi wrap.'''

    if len(yaws) < 2:
        return 0.0
    reference = yaws[-1]
    unwrapped = [reference + angle_difference(yaw, reference) for yaw in yaws]
    return max(unwrapped) - min(unwrapped)


def straight_suffix_metrics(poses: list[PoseStamped], min_distance_m: float):
    '''Return travel and chord/travel ratio for the newest distance-long suffix.'''

    if not poses:
        return 0.0, 0.0
    if min_distance_m <= 0.0:
        return 0.0, 1.0

    end = poses[-1].pose.position
    travel = 0.0
    start = end
    for index in range(len(poses) - 1, 0, -1):
        current = poses[index].pose.position
        previous = poses[index - 1].pose.position
        travel += math.hypot(current.x - previous.x, current.y - previous.y)
        start = previous
        if travel >= min_distance_m:
            chord = math.hypot(end.x - start.x, end.y - start.y)
            return travel, chord / travel if travel > 0.0 else 0.0
    return travel, 0.0


def _quaternion_multiply(first, second) -> tuple[float, float, float, float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def transform_pose_stamped(pose: PoseStamped, transform) -> PoseStamped:
    '''Apply one TransformStamped without querying historical TF for each pose.'''

    rotation = transform.transform.rotation
    norm = math.sqrt(
        rotation.x * rotation.x
        + rotation.y * rotation.y
        + rotation.z * rotation.z
        + rotation.w * rotation.w
    )
    if norm < 1e-12:
        q_transform = (0.0, 0.0, 0.0, 1.0)
    else:
        q_transform = (
            rotation.x / norm,
            rotation.y / norm,
            rotation.z / norm,
            rotation.w / norm,
        )

    position = pose.pose.position
    rotated = _quaternion_multiply(
        _quaternion_multiply(
            q_transform, (position.x, position.y, position.z, 0.0)
        ),
        (-q_transform[0], -q_transform[1], -q_transform[2], q_transform[3]),
    )
    result = PoseStamped()
    result.header.stamp = pose.header.stamp
    result.header.frame_id = transform.header.frame_id
    result.pose.position.x = rotated[0] + transform.transform.translation.x
    result.pose.position.y = rotated[1] + transform.transform.translation.y
    result.pose.position.z = rotated[2] + transform.transform.translation.z

    source_rotation = pose.pose.orientation
    q_pose_norm = math.sqrt(
        source_rotation.x * source_rotation.x
        + source_rotation.y * source_rotation.y
        + source_rotation.z * source_rotation.z
        + source_rotation.w * source_rotation.w
    )
    if q_pose_norm < 1e-12:
        q_pose = (0.0, 0.0, 0.0, 1.0)
    else:
        q_pose = (
            source_rotation.x / q_pose_norm,
            source_rotation.y / q_pose_norm,
            source_rotation.z / q_pose_norm,
            source_rotation.w / q_pose_norm,
        )
    q_result = _quaternion_multiply(q_transform, q_pose)
    result.pose.orientation.x = q_result[0]
    result.pose.orientation.y = q_result[1]
    result.pose.orientation.z = q_result[2]
    result.pose.orientation.w = q_result[3]
    return result


def _identity_transform(parent_frame: str, child_frame: str) -> TransformStamped:
    """tf2 also answers a same-frame lookup with identity; this avoids the lookup."""

    transform = TransformStamped()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    transform.transform.rotation.w = 1.0
    return transform


def _matrix_from_transform(transform: TransformStamped) -> np.ndarray:
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    t = transform.transform.translation
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), t.x],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), t.y],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), t.z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _transform_from_matrix(
    matrix: np.ndarray, parent_frame: str, child_frame: str, stamp
) -> TransformStamped:
    transform = TransformStamped()
    transform.header.frame_id = parent_frame
    transform.header.stamp = stamp
    transform.child_frame_id = child_frame
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qx, qy, qz, qw = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qx, qy, qz, qw = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qx, qy, qz, qw = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qx, qy, qz, qw = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    transform.transform.translation.x = float(m[0, 3])
    transform.transform.translation.y = float(m[1, 3])
    transform.transform.translation.z = float(m[2, 3])
    transform.transform.rotation.x = float(qx)
    transform.transform.rotation.y = float(qy)
    transform.transform.rotation.z = float(qz)
    transform.transform.rotation.w = float(qw)
    return transform


def interpolate_position(stamps_ns, positions, stamp_ns: int, max_gap_ns: int):
    """Linear position at stamp_ns from a stamp-sorted history, or None.

    None when the stamp falls outside the history or the two bracketing samples
    are further apart than max_gap_ns - a LiDAR gap that long is not something a
    straight line should paper over.
    """

    index = bisect.bisect_left(stamps_ns, stamp_ns)
    if index < len(stamps_ns) and stamps_ns[index] == stamp_ns:
        return positions[index]
    if index == 0 or index >= len(stamps_ns):
        return None
    before, after = stamps_ns[index - 1], stamps_ns[index]
    if after - before > max_gap_ns:
        return None
    fraction = (stamp_ns - before) / (after - before)
    return tuple(
        a + fraction * (b - a)
        for a, b in zip(positions[index - 1], positions[index])
    )


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid 4x4 transform (no scale) taking source onto target."""

    s_mean, t_mean = source.mean(axis=0), target.mean(axis=0)
    h = (source - s_mean).T @ (target - t_mean)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = t_mean - rotation @ s_mean
    return matrix


def _residuals(matrix: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm((matrix[:3, :3] @ source.T).T + matrix[:3, 3] - target, axis=1)


def fit_rigid_transform(
    source_points, target_points, outlier_distance_m: float, ransac_iterations: int = 300
):
    """Robust rigid transform taking source points onto target points.

    A plain least-squares solve is steered by every pair, and a stretch where
    LiDAR odometry lost tracking contributes pairs that all map one LiDAR
    position onto metres of fused route (testweek_0905_3: 74 m of driving
    recorded as 2.3 m, i.e. ~20 % of the run's pairs). Measured live, a
    least-squares-then-reject loop started from that contaminated solve kept
    only 11 of 367 pairs. So the model is found by RANSAC first: repeated
    minimal (three-pair) Kabsch solves, keeping the one most pairs agree with
    within outlier_distance_m, then refined by least squares over those pairs,
    twice. Deterministic (fixed seed) so a run's sidecar is reproducible.

    Returns (4x4 matrix, {"pairs", "inliers", "residual_median_m",
    "residual_p95_m"}), or None with fewer than three pairs.
    """

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    count = len(source)
    if count < 3 or count != len(target):
        return None

    rng = np.random.default_rng(0)
    best_keep = np.zeros(count, dtype=bool)
    for _ in range(ransac_iterations if count > 3 else 1):
        sample = rng.choice(count, size=3, replace=False)
        # Three nearly collinear pairs pin the rotation badly; skip them.
        if np.linalg.matrix_rank(source[sample] - source[sample].mean(axis=0), tol=0.05) < 2:
            continue
        keep = _residuals(_kabsch(source[sample], target[sample]), source, target) <= outlier_distance_m
        if keep.sum() > best_keep.sum():
            best_keep = keep
    if best_keep.sum() < 3:
        # Degenerate (e.g. a straight line so far): fall back to all pairs.
        best_keep = np.ones(count, dtype=bool)

    keep = best_keep
    matrix = _kabsch(source[keep], target[keep])
    for _ in range(2):
        residual = _residuals(matrix, source, target)
        refined = residual <= outlier_distance_m
        if refined.sum() < 3:
            break
        keep = refined
        matrix = _kabsch(source[keep], target[keep])
    residual = _residuals(matrix, source, target)
    keep = residual <= outlier_distance_m
    inlier_residual = np.sort(residual[keep]) if keep.any() else np.sort(residual)
    stats = {
        "pairs": int(count),
        "inliers": int(keep.sum()),
        "residual_median_m": float(np.median(inlier_residual)),
        "residual_p95_m": float(inlier_residual[int(0.95 * (len(inlier_residual) - 1))]),
    }
    return matrix, stats


class WaypointSaverNode(Node):
    """Record the driven route and its drivable corridor in UTM.

    Input poses (gnss_imu_pose_node's RTK pose by default) are buffered in route_frame from the
    very first sample; the whole history is reprojected with one latest
    utm<-route_frame transform whenever it changes, so early global-yaw
    corrections cannot kink the route. georeference.yaml separately records
    where LidarOdometry's point-cloud map (pointcloud_map_frame) sits in UTM:
    a least-squares rigid fit of the raw LiDAR-odometry trajectory onto the
    route over the whole run (see _refresh_pointcloud_transform).
    """

    def __init__(self):
        super().__init__("waypoint_saver_node")

        # The route follows the RTK/IMU pose (gnss_imu_pose_node), with no LiDAR
        # odometry in it. Raw LiDAR was measured 1.4-1.6 m from RTK on average
        # and 3.2 m at worst over a 190 m loop, and a rigid reprojection cannot
        # remove drift that is internal to the trajectory. The fused pose
        # (pose_fusion_node) that replaced it still left routes up to 1.5 m off
        # the run's own RTK track where LiDAR lost tracking - and the path
        # trackers steer on /gnss_imu/pose. See config/waypoint_saver.yaml.
        self.declare_parameter("input_odom_topic", "/gnss_imu/pose")
        self.declare_parameter("utm_frame", "utm")
        self.declare_parameter("enu_frame", "enu")
        self.declare_parameter("visualization_frame", "map")
        # Frame the route history is buffered in and reprojected from. With the
        # RTK pose this is "map", whose UTM transform is the static
        # geo-reference, so no sample is ever re-expressed through a snapshot of
        # the dynamic map->odom edge. Raw LiDAR odometry input would use "odom".
        self.declare_parameter("route_frame", "map")
        # Frame of LidarOdometry's saved point-cloud map, for georeference.yaml
        # only; the route no longer lives in it.
        self.declare_parameter("pointcloud_map_frame", "odom")
        # Raw LiDAR-odometry pose stream, in pointcloud_map_frame. Not a route
        # source: it is the "source" side of the placement fit below, paired by
        # timestamp with the route samples ("target" side).
        self.declare_parameter("pointcloud_pose_topic", "/lidar_odometry/pose")
        # Placement fit of the point-cloud map (utm<-pointcloud_map_frame).
        # Any single instant of the live map->odom edge is a bet on LiDAR
        # odometry being right at that instant: captured at shutdown it carried
        # 3.5 m / 3.4 deg of loop-closure drift (0902_4); captured at the ~5 m
        # heading alignment it caught the heading from BEFORE a 20 s tracking loss
        # and placed a 516 m map 38 deg / 86 m off (testweek_0905_3). A rigid
        # least-squares fit over every timestamp-matched (LiDAR pose, route pose)
        # pair of the run is what both cases actually needed: 1.15 m median
        # residual on the latter. Pairs further than this from the fit are
        # dropped and the fit re-solved, so a stretch where odometry lost tracking
        # cannot steer the placement of the stretch that tracked.
        self.declare_parameter("placement_fit_min_pairs", 20)
        self.declare_parameter("placement_fit_outlier_m", 3.0)
        # A fit that explains only a minority of the pairs is a fit to the
        # broken stretch, not to the map: while LiDAR odometry is lost the
        # rejection loop can shrink the inlier set to a handful of samples that
        # happen to agree with each other (seen live: 3/47 pairs, yaw swinging
        # -108 -> +29 deg between refreshes). Such a solution is refused and the
        # last accepted placement, or the live TF edge, stays in force.
        self.declare_parameter("placement_fit_min_inlier_fraction", 0.5)
        self.declare_parameter("placement_fit_max_time_gap_sec", 0.5)
        self.declare_parameter(
            "georeference_topic", "/state_estimator/geo_ref_metadata"
        )
        self.declare_parameter(
            "map_georeference_topic", "/lidar_odometry/geo_ref_metadata"
        )
        self.declare_parameter("global_map_filename", "global_map.mm")
        self.declare_parameter("utm_zone_label", "52N")
        # Waypoint를 너무 촘촘하게 저장하지 않도록 하는 값.
        self.declare_parameter("min_waypoint_spacing_m", 1.0)
        self.declare_parameter("max_waypoint_jump_m", 5.0)
        self.declare_parameter("max_waypoint_speed_mps", 20.0)
        self.declare_parameter("max_position_sigma_m", 1.0)
        self.declare_parameter("corridor_half_width_m", 3.0)
        # RViz waypoint dots and their "#id  x, y" labels. Labelling the whole
        # route is unreadable (a 500 m route has ~400 waypoints) and costs one
        # RViz object each, so only the waypoints within this distance of the
        # vehicle are labelled; 0 disables labels.
        self.declare_parameter("waypoint_dot_scale_m", 0.35)
        self.declare_parameter("waypoint_label_radius_m", 12.0)
        self.declare_parameter("waypoint_label_text_size_m", 0.6)
        self.declare_parameter("initial_alignment_min_distance_m", 5.0)
        self.declare_parameter(
            "initial_alignment_min_straightness_ratio", 0.95
        )
        self.declare_parameter("initial_alignment_yaw_window_sec", 2.0)
        self.declare_parameter("initial_alignment_min_yaw_samples", 5)
        self.declare_parameter(
            "initial_alignment_max_yaw_spread_deg", 3.0
        )
        self.declare_parameter("autosave_interval_sec", 30.0)
        self.declare_parameter("markers_topic", "waypoint_markers")
        self.declare_parameter("output_dir", "~/krowd_maps")
        # Shared with mola-cli's MOLA_SIMPLEMAP_OUTPUT (set by mapping.launch.py) so the
        # waypoint/georeference files and the point-cloud map land in the same per-run
        # folder without the two separate processes needing a runtime handshake. Empty
        # (the default for standalone/manual runs, e.g. `ros2 run` without the launch
        # file) falls back to a timestamp generated here instead.
        self.declare_parameter("run_id", "")

        self._utm_frame = self.get_parameter("utm_frame").value
        self._enu_frame = self.get_parameter("enu_frame").value
        self._visualization_frame = self.get_parameter("visualization_frame").value
        self._route_frame = self.get_parameter("route_frame").value
        self._pointcloud_map_frame = self.get_parameter("pointcloud_map_frame").value
        self._placement_fit_min_pairs = int(
            self.get_parameter("placement_fit_min_pairs").value
        )
        self._placement_fit_outlier_m = float(
            self.get_parameter("placement_fit_outlier_m").value
        )
        self._placement_fit_min_inlier_fraction = float(
            self.get_parameter("placement_fit_min_inlier_fraction").value
        )
        self._placement_fit_max_gap_ns = int(
            round(float(self.get_parameter("placement_fit_max_time_gap_sec").value) * 1e9)
        )
        self._global_map_filename = self.get_parameter("global_map_filename").value
        self._utm_zone_label = self.get_parameter("utm_zone_label").value
        self._min_spacing = float(self.get_parameter("min_waypoint_spacing_m").value)
        self._max_waypoint_jump = float(
            self.get_parameter("max_waypoint_jump_m").value
        )
        self._max_waypoint_speed = float(
            self.get_parameter("max_waypoint_speed_mps").value
        )
        self._max_sigma = float(self.get_parameter("max_position_sigma_m").value)
        self._corridor_half_width = float(self.get_parameter("corridor_half_width_m").value)
        self._waypoint_dot_scale = float(
            self.get_parameter("waypoint_dot_scale_m").value
        )
        self._waypoint_label_radius = float(
            self.get_parameter("waypoint_label_radius_m").value
        )
        self._waypoint_label_text_size = float(
            self.get_parameter("waypoint_label_text_size_m").value
        )
        # How many text labels the last publication carried, so a shrinking
        # label set deletes the surplus instead of leaving orphan text in RViz.
        self._published_label_count = 0
        # Vehicle position in visualization_frame; centres the label window.
        self._label_center: tuple[float, float] | None = None
        self._alignment_min_distance = float(
            self.get_parameter("initial_alignment_min_distance_m").value
        )
        self._alignment_min_straightness = float(
            self.get_parameter(
                "initial_alignment_min_straightness_ratio"
            ).value
        )
        self._alignment_yaw_window_ns = int(
            round(
                float(
                    self.get_parameter(
                        "initial_alignment_yaw_window_sec"
                    ).value
                )
                * 1.0e9
            )
        )
        self._alignment_min_yaw_samples = int(
            self.get_parameter("initial_alignment_min_yaw_samples").value
        )
        self._alignment_max_yaw_spread = math.radians(
            float(
                self.get_parameter(
                    "initial_alignment_max_yaw_spread_deg"
                ).value
            )
        )

        output_root = Path(self.get_parameter("output_dir").value).expanduser()
        run_id = self.get_parameter("run_id").value or wall_time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = output_root / run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._waypoints_path = self._run_dir / "waypoints.csv"
        self._corridor_path = self._run_dir / "corridor.csv"
        self._georeference_path = self._run_dir / "georeference.yaml"
        self._native_georeference_path = self._run_dir / "global_map.georef.yaml"
        self._zone_paths = {
            kind: self._run_dir / kind.filename for kind in ZONE_KINDS
        }

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Every accepted sample remains in route_frame. UTM/map routes below are
        # derived views rebuilt from this complete history.
        self._raw_waypoints: list[PoseStamped] = []
        self._raw_waypoint_stamps_ns: list[int] = []
        self._last_local_position: tuple[float, float, float] | None = None
        self._last_local_stamp_ns: int | None = None
        self._latest_valid_local_pose: PoseStamped | None = None
        self._latest_valid_stamp_ns: int | None = None

        self._waypoints: list[RoutePoint] = []
        self._visualization_route: list[RoutePoint] = []
        # Latest single transforms used to rebuild the entire route. Previously
        # each sample was permanently transformed at its own callback time, which
        # bent the saved route whenever the smoother corrected global yaw.
        self._last_route_transform = None
        self._last_visualization_transform = None
        self._last_pointcloud_transform = None
        self._last_pointcloud_enu_transform = None
        # Raw LiDAR-odometry positions in pointcloud_map_frame, stamp-sorted, for
        # the placement fit (see the placement_fit_* parameters). ~7 Hz, so a
        # multi-hour run is still only a few hundred thousand tuples.
        self._pointcloud_pose_stamps_ns: list[int] = []
        self._pointcloud_positions: list[tuple[float, float, float]] = []
        # Statistics of the latest fit, written to georeference.yaml; None until
        # enough pairs exist and while the placement is a live TF fallback.
        self._placement_fit: dict | None = None
        self._alignment_ready = False
        self._alignment_yaw_samples = deque(maxlen=500)
        self._last_alignment_sample_stamp_ns: int | None = None
        self._georeference_msg = None
        # The smoother owns the final ENU->map estimate. LidarOdometry also publishes
        # metadata, but that stream can lag the smoother by one optimization update.
        # Use it only as a startup fallback so it cannot overwrite the authoritative
        # transform just before the map artifacts are saved.
        self._authoritative_georeference_received = False
        # Live RViz preview is kept in the small map frame; CSV artifacts remain UTM.
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._marker_pub = self.create_publisher(
            MarkerArray, self.get_parameter("markers_topic").value, latched_qos
        )

        topic = self.get_parameter("input_odom_topic").value
        self.create_subscription(Odometry, topic, self._on_odometry, 10)
        if self._pointcloud_map_frame != self._route_frame:
            self.create_subscription(
                Odometry,
                self.get_parameter("pointcloud_pose_topic").value,
                self._on_pointcloud_pose,
                20,
            )
        self.create_subscription(
            GeoreferencingMetadata,
            self.get_parameter("georeference_topic").value,
            self._on_georeference,
            latched_qos,
        )
        map_georeference_topic = self.get_parameter(
            "map_georeference_topic"
        ).value
        if map_georeference_topic != self.get_parameter(
            "georeference_topic"
        ).value:
            self.create_subscription(
                GeoreferencingMetadata,
                map_georeference_topic,
                self._on_map_georeference,
                latched_qos,
            )

        autosave_interval = float(self.get_parameter("autosave_interval_sec").value)
        if autosave_interval > 0.0:
            self.create_timer(autosave_interval, self._save_all)
        # Keep the projected route current even while the vehicle is stopped and
        # only the smoother's map/ENU/UTM alignment is still being refined.
        self.create_timer(0.5, self._refresh_route_projection)

        self.create_service(Trigger, "~/save_path", self._on_save_service)
        self.create_service(Trigger, "~/mark_waypoint", self._on_mark_waypoint_service)

        self.get_logger().info(
            f"waypoint_saver_node: listening on '{topic}', "
            f"buffering in '{self._route_frame}' and resolving the "
            f"whole route to '{self._utm_frame}', "
            f"will save to '{self._run_dir}'"
        )

    def _on_georeference(self, msg: GeoreferencingMetadata) -> None:
        if not msg.valid:
            return
        self._authoritative_georeference_received = True
        self._store_georeference(msg)

    def _on_map_georeference(self, msg: GeoreferencingMetadata) -> None:
        if not msg.valid or self._authoritative_georeference_received:
            return
        self._store_georeference(msg)

    def _store_georeference(self, msg: GeoreferencingMetadata) -> None:
        self._georeference_msg = msg
        self._refresh_pointcloud_transform()
        if self._last_pointcloud_enu_transform is not None:
            self._save_georeference()

    def _on_pointcloud_pose(self, msg: Odometry) -> None:
        """Keep the raw LiDAR-odometry position history for the placement fit."""

        if msg.header.frame_id.strip() != self._pointcloud_map_frame:
            return
        position = msg.pose.pose.position
        if not all(math.isfinite(v) for v in (position.x, position.y, position.z)):
            return
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if self._pointcloud_pose_stamps_ns and stamp_ns < self._pointcloud_pose_stamps_ns[-1]:
            # LidarOdometry restarted: its frame origin moved, so every earlier
            # sample describes a different odom and the fit must start over.
            self._pointcloud_pose_stamps_ns.clear()
            self._pointcloud_positions.clear()
            self._placement_fit = None
        self._pointcloud_pose_stamps_ns.append(stamp_ns)
        self._pointcloud_positions.append((position.x, position.y, position.z))

    def _placement_fit_pairs(self):
        """(LiDAR position, route position) for every route sample both saw."""

        source, target = [], []
        for pose, stamp_ns in zip(self._raw_waypoints, self._raw_waypoint_stamps_ns):
            lidar = interpolate_position(
                self._pointcloud_pose_stamps_ns,
                self._pointcloud_positions,
                stamp_ns,
                self._placement_fit_max_gap_ns,
            )
            if lidar is None:
                continue
            source.append(lidar)
            target.append((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z))
        return source, target

    def _on_odometry(self, msg: Odometry) -> None:
        cov = msg.pose.covariance
        # `>= 0.0` (not `> 0.0` with an `else 0.0`) so NaN and negative/"unknown-sentinel"
        # covariance values map to NaN here, not to 0.0 - otherwise NaN/negative would
        # silently read as "perfectly confident" and skip the isfinite() gate below.
        sigma_x = math.sqrt(cov[0]) if cov[0] >= 0.0 else math.nan
        sigma_y = math.sqrt(cov[7]) if cov[7] >= 0.0 else math.nan
        if not (math.isfinite(sigma_x) and math.isfinite(sigma_y)):
            return
        if sigma_x > self._max_sigma or sigma_y > self._max_sigma:
            return

        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose

        # The durable history lives in route_frame. No UTM transform is required
        # here, so every early pose survives while GNSS motion is still resolving
        # the initially weak global yaw.
        if not pose_in.header.frame_id:
            return
        if pose_in.header.frame_id != self._route_frame:
            try:
                pose_local = self._tf_buffer.transform(
                    pose_in,
                    self._route_frame,
                    timeout=Duration(seconds=0.0),
                )
            except TransformException:
                return
        else:
            pose_local = pose_in

        pose = pose_local.pose
        if not all(
            math.isfinite(value)
            for value in (
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        ):
            return

        stamp_ns = (
            int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec)
        )
        if (
            self._last_local_stamp_ns is not None
            and stamp_ns < self._last_local_stamp_ns
        ):
            self.get_logger().warning(
                "Odometry time moved backwards; starting a new waypoint history."
            )
            self._reset_route_history()

        local_position = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
        )
        if self._last_local_position is not None:
            dx = local_position[0] - self._last_local_position[0]
            dy = local_position[1] - self._last_local_position[1]
            dz = local_position[2] - self._last_local_position[2]
            step_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            elapsed_sec = 0.0
            if self._last_local_stamp_ns is not None:
                elapsed_sec = (
                    stamp_ns - self._last_local_stamp_ns
                ) * 1.0e-9
            jump_limit = waypoint_jump_limit_m(
                self._max_waypoint_jump,
                self._max_waypoint_speed,
                elapsed_sec,
            )
            if (
                self._max_waypoint_jump > 0.0
                and step_3d > jump_limit
            ):
                self.get_logger().warning(
                    "Rejected implausible waypoint pose jump: "
                    f"{step_3d:.1f} m > {jump_limit:.1f} m"
                )
                return

            self._latest_valid_local_pose = pose_local
            self._latest_valid_stamp_ns = stamp_ns
            if math.hypot(dx, dy) < self._min_spacing:
                self._refresh_route_projection(stamp_ns)
                return

        self._latest_valid_local_pose = pose_local
        self._latest_valid_stamp_ns = stamp_ns
        self._append_local_waypoint(pose_local, stamp_ns)

    def _append_local_waypoint(
        self, pose_local: PoseStamped, stamp_ns: int
    ) -> None:
        self._raw_waypoints.append(pose_local)
        self._raw_waypoint_stamps_ns.append(stamp_ns)
        position = pose_local.pose.position
        self._last_local_position = (position.x, position.y, position.z)
        self._last_local_stamp_ns = stamp_ns
        self._refresh_route_projection(stamp_ns)
        # This sidecar is tiny; refreshing it at route spacing keeps the map's
        # shutdown-time embedded alignment close to the final route estimate.
        self._save_georeference()

    def _reset_route_history(self) -> None:
        self._raw_waypoints.clear()
        self._raw_waypoint_stamps_ns.clear()
        self._last_local_position = None
        self._last_local_stamp_ns = None
        self._latest_valid_local_pose = None
        self._latest_valid_stamp_ns = None
        self._waypoints.clear()
        self._visualization_route.clear()
        self._label_center = None
        self._alignment_ready = False
        self._pointcloud_pose_stamps_ns.clear()
        self._pointcloud_positions.clear()
        self._placement_fit = None
        self._alignment_yaw_samples.clear()
        self._last_alignment_sample_stamp_ns = None
        self._publish_markers()

    def _record_alignment_yaw(self, transform, stamp_ns: int) -> None:
        if (
            self._last_alignment_sample_stamp_ns is not None
            and stamp_ns <= self._last_alignment_sample_stamp_ns
        ):
            return
        yaw = yaw_from_quaternion(transform.transform.rotation)
        if not math.isfinite(yaw):
            return
        self._alignment_yaw_samples.append((stamp_ns, yaw))
        self._last_alignment_sample_stamp_ns = stamp_ns

    def _alignment_conditions_met(self) -> tuple[bool, float, float, float]:
        travel, straightness = straight_suffix_metrics(
            self._raw_waypoints, self._alignment_min_distance
        )
        if travel < self._alignment_min_distance:
            return False, travel, straightness, math.inf
        if straightness < self._alignment_min_straightness:
            return False, travel, straightness, math.inf
        if not self._alignment_yaw_samples:
            return False, travel, straightness, math.inf

        newest_stamp = self._alignment_yaw_samples[-1][0]
        cutoff = newest_stamp - max(0, self._alignment_yaw_window_ns)
        if (
            self._alignment_yaw_window_ns > 0
            and newest_stamp - self._alignment_yaw_samples[0][0]
            < self._alignment_yaw_window_ns
        ):
            return False, travel, straightness, math.inf
        samples = [
            sample
            for sample in self._alignment_yaw_samples
            if sample[0] >= cutoff
        ]
        if len(samples) < max(1, self._alignment_min_yaw_samples):
            return False, travel, straightness, math.inf

        spread = circular_yaw_spread([sample[1] for sample in samples])
        return (
            spread <= self._alignment_max_yaw_spread,
            travel,
            straightness,
            spread,
        )

    def _on_mark_waypoint_service(self, request, response):
        # Manual override stores the current local pose even while global yaw is
        # pending. It is projected together with the complete history later.
        if self._latest_valid_local_pose is None:
            response.success = False
            response.message = "No valid pose yet; cannot mark a waypoint."
            return response
        self._append_local_waypoint(
            self._latest_valid_local_pose,
            self._latest_valid_stamp_ns,
        )
        response.success = True
        if self._alignment_ready and self._waypoints:
            global_point = self._waypoints[-1]
            response.message = (
                f"Marked waypoint #{len(self._raw_waypoints)} at "
                f"utm=({global_point.x:.3f}, {global_point.y:.3f})"
            )
        else:
            response.message = (
                f"Buffered local waypoint #{len(self._raw_waypoints)}; "
                "global yaw alignment is still pending."
            )
        return response

    def _refresh_route_projection(
        self,
        observation_stamp_ns: int | None = None,
        allow_unstable: bool = False,
    ) -> bool:
        if not self._raw_waypoints:
            return False

        try:
            route_transform = self._tf_buffer.lookup_transform(
                self._utm_frame,
                self._route_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
            self._last_route_transform = route_transform
        except TransformException:
            route_transform = self._last_route_transform

        if self._visualization_frame == self._route_frame:
            visualization_transform = _identity_transform(
                self._visualization_frame, self._route_frame
            )
            self._last_visualization_transform = visualization_transform
        else:
            try:
                visualization_transform = self._tf_buffer.lookup_transform(
                    self._visualization_frame,
                    self._route_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
                self._last_visualization_transform = visualization_transform
            except TransformException:
                visualization_transform = self._last_visualization_transform

        if route_transform is None or visualization_transform is None:
            # Kept as one combined guard: transform_pose_stamped(pose, None) below
            # would raise inside the shutdown save and destroy an otherwise
            # recoverable route.
            missing = (
                self._utm_frame if route_transform is None
                else self._visualization_frame
            )
            self.get_logger().warning(
                f"No '{missing}' <- '{self._route_frame}' transform yet: "
                f"{len(self._raw_waypoints)} waypoints are buffered but cannot be "
                "projected, so waypoints.csv/corridor.csv will NOT be written. "
                "utm<-map is gnss_imu_pose_node's static geo-reference (it needs "
                "both GNSS topics and ~5 m of straight driving); utm<-odom "
                "additionally needs map_odom_tf_node's map->odom.",
                throttle_duration_sec=10.0,
            )
            return False

        alignment_stamp_ns = observation_stamp_ns
        if alignment_stamp_ns is None:
            transform_stamp = route_transform.header.stamp
            alignment_stamp_ns = (
                int(transform_stamp.sec) * 1_000_000_000
                + int(transform_stamp.nanosec)
            )
            if alignment_stamp_ns == 0:
                alignment_stamp_ns = None
        if alignment_stamp_ns is not None:
            self._record_alignment_yaw(
                route_transform, alignment_stamp_ns
            )

        if not self._alignment_ready:
            ready, travel, straightness, spread = (
                self._alignment_conditions_met()
            )
            if ready:
                self._alignment_ready = True
                self.get_logger().info(
                    "Global yaw alignment ready: "
                    f"{travel:.1f} m straight-motion window, "
                    f"straightness={straightness:.3f}, "
                    f"yaw spread={math.degrees(spread):.2f} deg. "
                    f"Reprojecting all {len(self._raw_waypoints)} buffered "
                    "waypoints from the start."
                )
            elif not allow_unstable:
                return False

        projected_global: list[RoutePoint] = []
        projected_visualization: list[RoutePoint] = []
        for pose_local in self._raw_waypoints:
            pose_utm = transform_pose_stamped(
                pose_local, route_transform
            )
            pose_visualization = transform_pose_stamped(
                pose_local, visualization_transform
            )
            projected_global.append(
                make_route_point(
                    pose_utm.pose.position.x,
                    pose_utm.pose.position.y,
                    pose_utm.pose.position.z,
                    yaw_from_quaternion(pose_utm.pose.orientation),
                    self._corridor_half_width,
                )
            )
            projected_visualization.append(
                make_route_point(
                    pose_visualization.pose.position.x,
                    pose_visualization.pose.position.y,
                    pose_visualization.pose.position.z,
                    yaw_from_quaternion(
                        pose_visualization.pose.orientation
                    ),
                    self._corridor_half_width,
                )
            )

        self._waypoints = projected_global
        self._visualization_route = projected_visualization
        # The label window follows the pose, not the newest waypoint: waypoints
        # are only appended every min_waypoint_spacing_m, so the newest one sits
        # up to that far behind the vehicle.
        if self._latest_valid_local_pose is not None:
            vehicle = transform_pose_stamped(
                self._latest_valid_local_pose, visualization_transform
            ).pose.position
            self._label_center = (vehicle.x, vehicle.y)
        self._publish_markers()
        return True

    def _refresh_pointcloud_transform(self) -> None:
        """Place the point-cloud map (pointcloud_map_frame) in UTM and ENU.

        Sidecar only; the route transform is refreshed separately from
        route_frame and neither may alias the other. Preferred: a rigid fit of
        the whole LiDAR-odometry history onto the route (see the
        placement_fit_* parameters), composed with the static utm<-route_frame
        and enu<-route_frame transforms. Until enough pairs exist, the live TF
        edge is used as a stand-in.
        """

        if self._route_frame != self._pointcloud_map_frame:
            if self._fit_pointcloud_placement():
                return
            if self._placement_fit is not None:
                # A previously accepted fit outranks the live edge: the current
                # refresh was refused (too few pairs/inliers or no static edge
                # yet), not contradicted.
                return
        for attribute, parent in (
            ("_last_pointcloud_transform", self._utm_frame),
            ("_last_pointcloud_enu_transform", self._enu_frame),
        ):
            try:
                setattr(self, attribute, self._tf_buffer.lookup_transform(
                    parent,
                    self._pointcloud_map_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                ))
            except TransformException:
                pass

    def _fit_pointcloud_placement(self) -> bool:
        """True when the placement was set from the whole-run fit."""

        source, target = self._placement_fit_pairs()
        if len(source) < self._placement_fit_min_pairs:
            return False
        result = fit_rigid_transform(source, target, self._placement_fit_outlier_m)
        if result is None:
            return False
        route_to_map, stats = result
        if (
            stats["inliers"] < self._placement_fit_min_pairs
            or stats["inliers"] < self._placement_fit_min_inlier_fraction * stats["pairs"]
        ):
            self.get_logger().warning(
                "Point-cloud map placement fit refused: only "
                f"{stats['inliers']}/{stats['pairs']} LiDAR/route pairs agree within "
                f"{self._placement_fit_outlier_m:.1f} m - LiDAR odometry is not "
                "tracking the route over most of the run so far.",
                throttle_duration_sec=10.0,
            )
            return False

        placements = {}
        for attribute, parent in (
            ("_last_pointcloud_transform", self._utm_frame),
            ("_last_pointcloud_enu_transform", self._enu_frame),
        ):
            if parent == self._route_frame:
                parent_from_route = _identity_transform(parent, self._route_frame)
            else:
                try:
                    parent_from_route = self._tf_buffer.lookup_transform(
                        parent, self._route_frame, Time(), timeout=Duration(seconds=0.0)
                    )
                except TransformException:
                    return False
            placements[attribute] = _transform_from_matrix(
                _matrix_from_transform(parent_from_route) @ route_to_map,
                parent,
                self._pointcloud_map_frame,
                parent_from_route.header.stamp,
            )

        previous = self._placement_fit
        for attribute, transform in placements.items():
            setattr(self, attribute, transform)
        stats["method"] = "rigid_least_squares_fit_of_lidar_odometry_onto_route"
        stats["yaw_deg"] = math.degrees(
            yaw_from_quaternion(self._last_pointcloud_enu_transform.transform.rotation)
        )
        self._placement_fit = stats
        if previous is None or abs(stats["yaw_deg"] - previous["yaw_deg"]) > 0.5:
            self.get_logger().info(
                f"Point-cloud map placement from whole-run fit: {stats['inliers']}/"
                f"{stats['pairs']} pairs, residual median {stats['residual_median_m']:.2f} m "
                f"/ p95 {stats['residual_p95_m']:.2f} m, "
                f"enu<-{self._pointcloud_map_frame} yaw {stats['yaw_deg']:.2f} deg"
            )
        return True

    def _publish_markers(self) -> None:
        # rclpy's SIGINT handler can invalidate the context before spin()
        # returns.  The final best-effort save must still write CSV/YAML, but a
        # marker publication at that point raises RCLError and makes an
        # otherwise successful mapping shutdown report exit code 1.
        if not rclpy.ok():
            return
        self._marker_pub.publish(
            build_route_marker_array(
                self._visualization_route,
                self._visualization_frame,
                self.get_clock().now().to_msg(),
                namespace="waypoints",
                dot_scale_m=self._waypoint_dot_scale,
                # Labels here are the window around the vehicle only; the
                # whole-route every-n fallback belongs to viewers without a pose.
                label_every_n=0,
                label_text_size_m=self._waypoint_label_text_size,
                previous_label_count=self._published_label_count,
                label_center=self._label_center,
                label_radius_m=self._waypoint_label_radius,
            )
        )
        self._published_label_count = len(
            route_label_indices(
                self._visualization_route,
                center=self._label_center,
                radius_m=self._waypoint_label_radius,
            )
        )

    def _save_all(self, allow_unstable: bool = False) -> None:
        self._refresh_route_projection(allow_unstable=allow_unstable)
        self._save_waypoints()
        self._save_corridor()
        self._save_zone_templates()
        self._save_georeference()

    def _route_blocking_reason(self) -> str | None:
        """None when a route was written; otherwise why nothing could be.

        Several distinct conditions all present to the operator as "the run
        directory has no waypoints.csv". This names which one occurred.
        """

        if not self._raw_waypoints:
            return (
                "no pose was ever accepted (check that "
                f"'{self.get_parameter('input_odom_topic').value}' is published and "
                f"that its reported position sigma stays under "
                f"{self._max_sigma:.2f} m)"
            )
        if self._last_route_transform is None:
            return (
                f"no '{self._utm_frame}' <- '{self._route_frame}' transform "
                "was ever available (utm<-map is gnss_imu_pose_node's static "
                "geo-reference; utm<-odom additionally needs map_odom_tf_node)"
            )
        if self._last_visualization_transform is None:
            return (
                f"no '{self._visualization_frame}' <- "
                f"'{self._route_frame}' transform was ever available"
            )
        if not self._waypoints:
            return "the route projection produced no points"
        return None

    def _save_waypoints(self) -> None:
        # Centerline only - the drivable-area corridor boundary is saved
        # separately in corridor.csv (see _save_corridor) so a consumer that
        # only wants the reference path (e.g. the lane-detection/path-planning
        # team building their own in-corridor path) doesn't have to parse or
        # ignore boundary columns it doesn't need.
        if not self._waypoints:
            return
        tmp_path = self._waypoints_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["idx", "utm_zone", "utm_x", "utm_y", "utm_z", "yaw_rad"]
            )
            for idx, point in enumerate(self._waypoints):
                writer.writerow(
                    [
                        idx,
                        self._utm_zone_label,
                        f"{point.x:.3f}",
                        f"{point.y:.3f}",
                        f"{point.z:.3f}",
                        f"{point.yaw:.6f}",
                    ]
                )
        os.replace(tmp_path, self._waypoints_path)
        if rclpy.ok():
            self.get_logger().info(
                f"saved {len(self._waypoints)} waypoints to {self._waypoints_path}"
            )

    def _save_corridor(self) -> None:
        # 'idx' matches the same-index row in waypoints.csv (both are written
        # from the same self._waypoints list in the same order), so a consumer
        # can join the two files by idx instead of relying on row order alone.
        if not self._waypoints:
            return
        tmp_path = self._corridor_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "idx",
                    "utm_zone",
                    "left_x",
                    "left_y",
                    "left_z",
                    "right_x",
                    "right_y",
                    "right_z",
                ]
            )
            for idx, point in enumerate(self._waypoints):
                writer.writerow(
                    [
                        idx,
                        self._utm_zone_label,
                        f"{point.left_x:.3f}",
                        f"{point.left_y:.3f}",
                        f"{point.z:.3f}",
                        f"{point.right_x:.3f}",
                        f"{point.right_y:.3f}",
                        f"{point.z:.3f}",
                    ]
                )
        os.replace(tmp_path, self._corridor_path)

    def _save_zone_templates(self) -> None:
        """Leave an empty per-map zone file of each kind beside waypoints.csv:
        traffic_detect_zones.yaml, slow_down_zones.yaml, avoidance_zones.yaml.

        Each is rewritten on every save while it still lists no zones, so the
        waypoint range in its header keeps up with the growing route and is
        final once the run ends. As soon as an operator fills zones in, that
        file is never touched again - including by a later re-run of this node
        on the same directory.
        """

        if not self._waypoints:
            return
        for kind, path in self._zone_paths.items():
            if not template_may_be_rewritten(path, kind):
                continue
            tmp_path = path.with_suffix(".yaml.tmp")
            with open(tmp_path, "w") as stream:
                stream.write(
                    zone_template_yaml(
                        self._run_dir.name, len(self._waypoints), kind=kind
                    )
                )
            os.replace(tmp_path, path)

    def _save_georeference(self) -> None:
        self._refresh_pointcloud_transform()
        if (
            self._last_route_transform is None
            and self._last_pointcloud_transform is None
            and self._last_pointcloud_enu_transform is None
            and self._georeference_msg is None
        ):
            return

        data = {
            "schema_version": 2,
            "global_frame": self._utm_frame,
            "utm_zone_label": self._utm_zone_label,
            "artifacts": {
                "pointcloud_map": {
                    "file": self._global_map_filename,
                    "coordinate_frame": self._pointcloud_map_frame,
                    "georeference_file": self._native_georeference_path.name,
                    "contains_full_accumulated_run": True,
                },
                "simplemap": {
                    "file": "map.simplemap",
                    "coordinate_frame": self._pointcloud_map_frame,
                },
                "route": {
                    "file": "waypoints.csv",
                    "coordinate_frame": self._utm_frame,
                    "source_frame": self._route_frame,
                    "source_topic": self.get_parameter("input_odom_topic").value,
                    "buffered_from_first_pose": True,
                    "global_yaw_stability_gate_passed": self._alignment_ready,
                },
                "drivable_corridor": {
                    "file": "corridor.csv",
                    "coordinate_frame": self._utm_frame,
                },
            },
            "note": (
                "Point coordinates stay near the run origin for numeric precision. "
                "Use pointcloud_map_to_global to place global_map.mm in UTM. The "
                "route is rebuilt in one operation from the complete source-pose "
                "history using the latest single global transform, so early global "
                "yaw corrections cannot kink it. A rigid transform cannot remove "
                "drift internal to the source trajectory, which is why the source "
                "is the RTK/IMU pose rather than LiDAR odometry: "
                "the point-cloud map still carries LiDAR drift, so expect the route "
                "and LiDAR-only localization to disagree by that drift. "
                "pointcloud_map_to_global/enu come from a rigid least-squares fit "
                "of the whole LiDAR-odometry history onto the route "
                "(pointcloud_map_placement holds its residuals); a large residual "
                "there means the odometry itself is deformed and the map should "
                "not be trusted for localization."
            ),
            "pointcloud_map_placement": self._placement_fit or {
                "method": "live_tf_fallback_too_few_pairs"
            },
            "route_alignment": {
                "ready": self._alignment_ready,
                "min_straight_distance_m": self._alignment_min_distance,
                "min_straightness_ratio": self._alignment_min_straightness,
                "yaw_window_sec": self._alignment_yaw_window_ns * 1.0e-9,
                "min_yaw_samples": self._alignment_min_yaw_samples,
                "max_yaw_spread_deg": math.degrees(
                    self._alignment_max_yaw_spread
                ),
                "raw_waypoint_count": len(self._raw_waypoints),
            },
        }

        if self._last_route_transform is not None:
            route_transform_data = _transform_to_dict(
                self._last_route_transform
            )
            data["route_local_frame_to_global"] = route_transform_data
            # Schema-v2 compatibility for existing artifact readers. The source
            # frame field inside the value remains authoritative (route_frame).
            data["localization_frame_to_global"] = _transform_to_dict(
                self._last_route_transform
            )
        if self._last_pointcloud_transform is not None:
            data["pointcloud_map_to_global"] = _transform_to_dict(
                self._last_pointcloud_transform
            )
        if self._last_pointcloud_enu_transform is not None:
            data["pointcloud_map_to_enu"] = _transform_to_dict(
                self._last_pointcloud_enu_transform
            )

        if self._georeference_msg is not None:
            msg = self._georeference_msg
            data["geodetic_origin_wgs84"] = {
                "latitude": msg.latitude,
                "longitude": msg.longitude,
                "height": msg.height,
                "utm_zone": msg.utm_zone,
                "utm_band": msg.utm_band,
            }
            data["enu_to_localization_frame"] = _pose_to_dict(
                msg.t_enu_to_map.pose
            )
            data["enu_to_localization_frame"]["covariance"] = list(
                msg.t_enu_to_map.covariance
            )
            data["enu_to_utm"] = _pose_to_dict(msg.t_enu_to_utm)

        tmp_path = self._georeference_path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as f:
            yaml.safe_dump(_yaml_builtin_scalars(data), f, sort_keys=False)
        os.replace(tmp_path, self._georeference_path)
        self._save_native_georeference()

    def _save_native_georeference(self) -> None:
        """Write the schema consumed by mp2p_icp::FromYAML/mm-georef."""

        if (
            self._georeference_msg is None
            or self._last_pointcloud_enu_transform is None
        ):
            return

        msg = self._georeference_msg
        transform = self._last_pointcloud_enu_transform.transform
        yaw, pitch, roll = euler_from_quaternion(transform.rotation)
        native = {
            "type": "mp2p_icp::Georeferencing",
            "defined": True,
            "geo_coord": {
                "lon": msg.longitude,
                "lat": msg.latitude,
                "altitude": msg.height,
            },
            "T_enu_to_map": {
                # In MOLA notation this is the pose of the saved map frame
                # (mapping-run odom) with respect to ENU.
                "mean": {
                    "x": transform.translation.x,
                    "y": transform.translation.y,
                    "z": transform.translation.z,
                    "yaw_deg": math.degrees(yaw),
                    "pitch_deg": math.degrees(pitch),
                    "roll_deg": math.degrees(roll),
                },
                "cov": _mrpt_diagonal_covariance(1e-6),
            },
        }
        tmp_path = self._native_georeference_path.with_suffix(".yaml.tmp")
        with open(tmp_path, "w") as stream:
            yaml.safe_dump(_yaml_builtin_scalars(native), stream, sort_keys=False)
        os.replace(tmp_path, self._native_georeference_path)

    def _on_save_service(self, request, response):
        # Explicit operator save writes a best-effort route even when a short
        # run has not yet satisfied the configurable yaw-stability gate.
        self._save_all(allow_unstable=True)
        reason = self._route_blocking_reason()
        # Previously this always reported success, even when nothing was written.
        response.success = reason is None
        alignment = (
            "stable" if self._alignment_ready else "best-effort/unverified"
        )
        if response.success:
            response.message = (
                f"{self._run_dir}: {len(self._waypoints)} waypoints "
                f"(global yaw: {alignment})"
            )
        else:
            response.message = (
                f"NOTHING SAVED to {self._run_dir}: {reason} "
                f"({len(self._raw_waypoints)} buffered local poses)"
            )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = WaypointSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Never discard the buffered beginning of a short/interrupted run.
        # georeference.yaml records whether the normal stability gate passed.
        node._save_all(allow_unstable=True)
        reason = node._route_blocking_reason()
        if reason is not None:
            # rclpy's SIGINT handler may already have invalidated the context, so
            # get_logger() is unusable here (cf. the rclpy.ok() guards above). Go
            # straight to stderr - this is the last chance to tell the operator that
            # the run produced no route, which was previously silent.
            print(
                f"[waypoint_saver_node] NO ROUTE SAVED to {node._run_dir}: {reason} "
                f"({len(node._raw_waypoints)} buffered local poses)",
                file=sys.stderr,
                flush=True,
            )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
