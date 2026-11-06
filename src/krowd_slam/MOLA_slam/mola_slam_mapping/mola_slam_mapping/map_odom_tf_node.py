import math
from collections import deque
from copy import deepcopy
import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker

from mola_slam_mapping.waypoint_saver_node import transform_pose_stamped


def _mat_from_pose(p):
    """4x4 homogeneous matrix from a geometry_msgs Pose, as nested lists."""
    q = p.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = x / n, y / n, z / n, w / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p.position.x],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p.position.y],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p.position.z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _invert(m):
    """Inverse of a rigid 4x4 transform: R^T and -R^T t."""
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    t = [m[i][3] for i in range(3)]
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    ti = [-sum(rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return [[rt[i][0], rt[i][1], rt[i][2], ti[i]] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _quat_from_mat(m):
    """Rotation matrix -> quaternion (x, y, z, w), branch-selected for stability."""
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s, 0.25 * s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        return (0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s,
                (m[2][1] - m[1][2]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        return ((m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s,
                (m[0][2] - m[2][0]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
    return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s,
            (m[1][0] - m[0][1]) / s)


def _unit_quaternion(orientation):
    values = [
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(value / norm for value in values)


def _slerp(first, second, fraction):
    first = _unit_quaternion(first)
    second = _unit_quaternion(second)
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = tuple(-value for value in second)
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        mixed = tuple(
            a + fraction * (b - a) for a, b in zip(first, second)
        )
        norm = math.sqrt(sum(value * value for value in mixed))
        return tuple(value / norm for value in mixed)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_weight = math.sin((1.0 - fraction) * theta) / sin_theta
    second_weight = math.sin(fraction * theta) / sin_theta
    return tuple(
        first_weight * a + second_weight * b
        for a, b in zip(first, second)
    )


def interpolate_pose_by_stamp(samples, target_stamp_ns: int, max_delta_ns: int):
    """Interpolate a LiDAR pose at target time from two bounded samples."""

    if not samples:
        return None
    ordered = sorted(samples, key=lambda item: item[0])
    for stamp_ns, message in reversed(ordered):
        if stamp_ns == target_stamp_ns:
            return stamp_ns, message

    before = next(
        (item for item in reversed(ordered) if item[0] < target_stamp_ns),
        None,
    )
    after = next(
        (item for item in ordered if item[0] > target_stamp_ns),
        None,
    )
    if before is None or after is None:
        return None
    if (
        target_stamp_ns - before[0] > max_delta_ns
        or after[0] - target_stamp_ns > max_delta_ns
    ):
        return None

    before_msg, after_msg = before[1], after[1]
    if (
        before_msg.header.frame_id != after_msg.header.frame_id
        or before_msg.child_frame_id != after_msg.child_frame_id
    ):
        raise ValueError("cannot interpolate LiDAR poses with different frames")

    fraction = (target_stamp_ns - before[0]) / (after[0] - before[0])
    pose_before = before_msg.pose.pose
    pose_after = after_msg.pose.pose
    interpolated = Odometry()
    interpolated.header.frame_id = before_msg.header.frame_id
    interpolated.header.stamp.sec = target_stamp_ns // 1_000_000_000
    interpolated.header.stamp.nanosec = target_stamp_ns % 1_000_000_000
    interpolated.child_frame_id = before_msg.child_frame_id
    for axis in ("x", "y", "z"):
        start = getattr(pose_before.position, axis)
        end = getattr(pose_after.position, axis)
        setattr(
            interpolated.pose.pose.position,
            axis,
            start + fraction * (end - start),
        )
    quaternion = _slerp(
        pose_before.orientation,
        pose_after.orientation,
        fraction,
    )
    for axis, value in zip(("x", "y", "z", "w"), quaternion):
        setattr(interpolated.pose.pose.orientation, axis, value)
    return target_stamp_ns, interpolated


def path_jump_limit_m(
    base_jump_m: float, max_speed_mps: float, elapsed_sec: float
) -> float:
    """Maximum plausible path step, accounting for a real publication gap."""

    return max(base_jump_m, max_speed_mps * max(0.0, elapsed_sec))


def same_frame_pose_distance_xy(first: Odometry, second: Odometry):
    """Planar distance, or None when two poses do not share a parent frame."""

    if first.header.frame_id.strip() != second.header.frame_id.strip():
        return None
    return math.hypot(
        first.pose.pose.position.x - second.pose.pose.position.x,
        first.pose.pose.position.y - second.pose.pose.position.y,
    )


def compose_map_to_odom(state_msg: Odometry, lidar_msg: Odometry):
    """Build parent<-LO-frame from two parent<-vehicle poses.

    None is returned when both sources already use the same parent frame; in
    localization mode the loaded point cloud is deliberately published as map
    and broadcasting a map->map self transform would be invalid.
    """

    parent_frame = state_msg.header.frame_id
    lidar_frame = lidar_msg.header.frame_id
    if not parent_frame or not lidar_frame:
        raise ValueError("state and lidar poses must have non-empty frame IDs")
    if (
        state_msg.child_frame_id
        and lidar_msg.child_frame_id
        and state_msg.child_frame_id != lidar_msg.child_frame_id
    ):
        raise ValueError(
            "state and lidar poses use different child frames "
            f"('{state_msg.child_frame_id}' vs '{lidar_msg.child_frame_id}')"
        )
    if parent_frame == lidar_frame:
        return None

    matrix = _mul(
        _mat_from_pose(state_msg.pose.pose),
        _invert(_mat_from_pose(lidar_msg.pose.pose)),
    )
    qx, qy, qz, qw = _quat_from_mat(matrix)

    transform = TransformStamped()
    transform.header.stamp = state_msg.header.stamp
    transform.header.frame_id = parent_frame
    transform.child_frame_id = lidar_frame
    transform.transform.translation.x = matrix[0][3]
    transform.transform.translation.y = matrix[1][3]
    transform.transform.translation.z = matrix[2][3]
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


BASE_LINK_TF_SOURCES = ("none", "lidar", "state", "auto")
"""Which pose, if any, this node publishes as the TF edge ending at the vehicle
frame.

One value, not two booleans, because `base_link` may have exactly one parent:
publishing both the LiDAR and the state pose would give TF two contradictory
answers for where the vehicle is, and tf2 resolves that by whichever arrived
last. "none" is for a system where something else (a vehicle ESKF) already owns
that edge. "auto" publishes the state (vehicle) pose only while no other node
does - see AutoBaseLinkTfGate.
"""


class AutoBaseLinkTfGate:
    """Decides, for base_link_tf_source "auto", whether this node may own the
    TF edge ending at the vehicle frame.

    On the live vehicle the ESKF already publishes odom->base_link; in a bag
    replay (recorded /tf excluded) nothing does. The gate watches /tf for a
    transform into the vehicle frame that this node did not send, holds its own
    edge back for a grace period so a slower owner can show up first, and gives
    the edge up for good once one does. This node's own transforms come back on
    its /tf subscription too, so they are recognised by (parent, child, stamp).
    """

    def __init__(self, grace_ns: int, start_ns: int, own_history: int = 256):
        if grace_ns < 0:
            raise ValueError("base_link_tf_auto_grace_sec must be non-negative")
        self._ready_ns = start_ns + grace_ns
        self._own_history = own_history
        self._own_order: deque = deque()
        self._own: set = set()
        self._published_children: set = set()
        self._foreign_parents: dict = {}

    def note_published(self, parent: str, child: str, stamp_ns: int) -> None:
        key = (parent, child, stamp_ns)
        self._published_children.add(child)
        if key in self._own:
            return
        self._own.add(key)
        self._own_order.append(key)
        while len(self._own_order) > self._own_history:
            self._own.discard(self._own_order.popleft())

    def observe(self, parent: str, child: str, stamp_ns: int) -> bool:
        """Record one /tf transform.

        Returns True only when another node takes over an edge this node had
        already been publishing.
        """

        if (parent, child, stamp_ns) in self._own or child in self._foreign_parents:
            return False
        self._foreign_parents[child] = parent
        return child in self._published_children

    def foreign_parent(self, child: str):
        return self._foreign_parents.get(child)

    def may_publish(self, child: str, now_ns: int) -> bool:
        return child not in self._foreign_parents and now_ns >= self._ready_ns


def validate_base_link_tf_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in BASE_LINK_TF_SOURCES:
        raise ValueError(
            f"base_link_tf_source must be one of {BASE_LINK_TF_SOURCES}, "
            f"got '{value}'"
        )
    return source


def odometry_to_transform(message: Odometry):
    """Convert an Odometry pose to its equivalent parent->child TF."""

    parent_frame = message.header.frame_id.strip()
    child_frame = message.child_frame_id.strip()
    if not parent_frame or not child_frame:
        raise ValueError(
            "odometry pose must have non-empty parent and child frame IDs"
        )
    if parent_frame == child_frame:
        raise ValueError(
            f"odometry pose cannot publish the self transform '{parent_frame}'"
        )

    pose = message.pose.pose
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("odometry pose contains a non-finite transform")

    qx, qy, qz, qw = _unit_quaternion(pose.orientation)
    transform = TransformStamped()
    transform.header = message.header
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    transform.transform.translation.x = pose.position.x
    transform.transform.translation.y = pose.position.y
    transform.transform.translation.z = pose.position.z
    transform.transform.rotation.x = qx
    transform.transform.rotation.y = qy
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


VEHICLE_MARKER_COLOR_RGBA = (0.0, 0.24, 1.0, 0.9)
"""Vehicle cuboid colour, RGBA in 0..1. A deeper blue than the Driven Path
(0;170;255) and the corrected-map preview (130;200;255) so the three stay
distinguishable."""


def vehicle_marker_from_odometry(
    message: Odometry,
    length_m: float,
    width_m: float,
    height_m: float,
    z_offset_m: float,
    color_rgba=VEHICLE_MARKER_COLOR_RGBA,
) -> Marker:
    """Create a vehicle cuboid at a fused odometry pose."""

    if not message.header.frame_id.strip():
        raise ValueError("vehicle marker pose must have a non-empty frame ID")
    dimensions = (length_m, width_m, height_m)
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        raise ValueError("vehicle marker dimensions must be finite and positive")
    if not math.isfinite(z_offset_m):
        raise ValueError("vehicle marker z offset must be finite")
    if len(color_rgba) != 4 or not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in color_rgba
    ):
        raise ValueError("vehicle marker colour must be 4 values in 0..1 (RGBA)")

    marker = Marker()
    marker.header = message.header
    marker.ns = "vehicle"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    # Re-placed by RViz every render frame rather than once per message, so the
    # vehicle sits still in a vehicle-frame view instead of twitching at the
    # pose rate. See the note in route_visualization.initialize_marker().
    marker.frame_locked = True
    marker.pose = deepcopy(message.pose.pose)
    marker.pose.position.z += z_offset_m
    marker.scale.x = length_m
    marker.scale.y = width_m
    marker.scale.z = height_m
    (
        marker.color.r,
        marker.color.g,
        marker.color.b,
        marker.color.a,
    ) = (float(value) for value in color_rgba)
    return marker


class MapFramePath:
    """Decimated RViz Path for poses that already arrive in the map frame.

    Two trajectories share this logic and differ only in where they come from:
    the GNSS/IMU reference pose (RTK position + course, no LiDAR) and the fused
    state-estimator pose (LiDAR odometry + IMU + GNSS in one factor graph). They
    are accumulated as separate lines on purpose - overlaying them is how a run
    is judged, so neither may be substituted for the other.

    Unlike the LiDAR-odometry path these need no reprojection, so a pose is
    stored exactly as received once it passes the spacing and jump filters.
    """

    def __init__(
        self,
        node: Node,
        label: str,
        frame: str,
        topic: str,
        qos: QoSProfile,
        min_spacing_m: float,
        max_jump_m: float,
        max_speed_mps: float,
        max_points: int,
    ):
        self._logger = node.get_logger()
        self._label = label
        self._frame = frame
        self._min_spacing = min_spacing_m
        self._max_jump = max_jump_m
        self._max_speed = max_speed_mps
        self._max_points = max_points

        self._publisher = node.create_publisher(Path, topic, qos)
        self._path = Path()
        self._path.header.frame_id = frame
        self._poses: list[PoseStamped] = []
        self._last_xy = None
        self._last_position = None
        self._last_stamp_ns = None

    def append(self, msg: Odometry) -> None:
        source_frame = msg.header.frame_id.strip()
        if not source_frame:
            return
        if source_frame != self._frame:
            self._logger.warning(
                f"Not appending {self._label} path pose: expected frame "
                f"'{self._frame}', got '{source_frame}'",
                throttle_duration_sec=5.0,
            )
            return

        pose = msg.pose.pose
        pose_values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in pose_values):
            return
        position = pose_values[:3]
        stamp_ns = (
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )
        if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
            self._reset()

        if self._last_position is not None:
            dx = position[0] - self._last_position[0]
            dy = position[1] - self._last_position[1]
            dz = position[2] - self._last_position[2]
            step_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            elapsed_sec = 0.0
            if self._last_stamp_ns is not None:
                elapsed_sec = (stamp_ns - self._last_stamp_ns) * 1.0e-9
            jump_limit = path_jump_limit_m(
                self._max_jump, self._max_speed, elapsed_sec
            )
            if self._max_jump > 0.0 and step_3d > jump_limit:
                self._logger.warning(
                    f"Rejected implausible {self._label}-path pose jump: "
                    f"{step_3d:.1f} m > {jump_limit:.1f} m"
                )
                return

        if self._last_xy is not None:
            if (
                math.hypot(
                    position[0] - self._last_xy[0],
                    position[1] - self._last_xy[1],
                )
                < self._min_spacing
            ):
                return

        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header
        pose_stamped.pose = pose
        self._poses.append(pose_stamped)
        if self._max_points > 0 and len(self._poses) > self._max_points:
            del self._poses[0]

        self._last_xy = (position[0], position[1])
        self._last_position = position
        self._last_stamp_ns = stamp_ns
        self._path.poses = list(self._poses)
        self._path.header.frame_id = self._frame
        self._path.header.stamp = msg.header.stamp
        self._publisher.publish(self._path)

    def _reset(self) -> None:
        self._poses.clear()
        self._last_xy = None
        self._last_position = None
        self._last_stamp_ns = None


class MapOdomTfNode(Node):
    """Visualization support for the map->odom->base_link TF chain and Path.

    LidarOdometry publishes its accumulated point-cloud map in its own "odom"
    frame, but nothing here links "odom" to the rest of the TF tree: the vehicle's
    onboard ESKF owns base_link (enu->base_link) and MOLA contributes only the
    static enu->map and enu->utm geo-reference. Without map->odom the map point
    cloud cannot be drawn in the same view as the utm-frame waypoints.

    Both halves needed to close that gap are already published as Odometry:
        /gnss_imu/pose          : map  -> base_link   (RTK position + course)
        /lidar_odometry/pose    : odom -> base_link   (raw LiDAR odometry)
    so  map->odom = (map->base_link) * (odom->base_link)^-1.

    Three trajectories are drawn, and comparing them is the point:
        driven_path              (blue)   LiDAR odometry, reprojected into map
                                          with the single latest correction
        /state_estimation/path   (red)    the map->odom source above, unfused
        /fused_localization/path (purple) pose_fusion_node's fused pose
    Blue and purple differ by more than smoothing: blue applies one instantaneous
    correction to the whole history, so it stays a rigid transform of the LiDAR
    path and keeps all its drift, while purple applies a time-varying filtered
    correction. Where the two separate is where LiDAR odometry has drifted.

    The LiDAR pose can optionally be republished as odom->base_link. This must be
    enabled only when the launch file excludes every competing base_link parent.
    The bag wrapper does that by excluding the recorded ESKF /tf stream; live
    launch keeps the option disabled by default (see AI_CONTEXT.md).
    """

    def __init__(self):
        super().__init__("map_odom_tf_node")

        self.declare_parameter("state_pose_topic", "/state_estimation/pose")
        self.declare_parameter("lidar_pose_topic", "/lidar_odometry/pose")
        self.declare_parameter("path_frame", "map")
        self.declare_parameter("path_topic", "driven_path")
        self.declare_parameter("fused_path_topic", "/state_estimation/path")
        # Separate source and line for the genuinely fused estimate, so it can be
        # held against the unfused GNSS/IMU one instead of replacing it. The
        # source is pose_fusion_node, not MOLA's state estimator - see
        # config/mapping_system.yaml for why the smoother is not used here.
        self.declare_parameter(
            "fused_localization_pose_topic", "/fused_localization/pose"
        )
        self.declare_parameter(
            "fused_localization_path_topic", "/fused_localization/path"
        )
        self.declare_parameter("vehicle_marker_topic", "/vehicle_marker")
        self.declare_parameter("vehicle_marker_length_m", 2.5)
        self.declare_parameter("vehicle_marker_width_m", 1.2)
        self.declare_parameter("vehicle_marker_height_m", 0.8)
        self.declare_parameter("vehicle_marker_z_offset_m", 0.4)
        # RGBA in 0..1; see VEHICLE_MARKER_COLOR_RGBA for why this blue.
        self.declare_parameter(
            "vehicle_marker_color_rgba", list(VEHICLE_MARKER_COLOR_RGBA)
        )
        # Path is rebuilt for every pose, so cap it: at 20 Hz an unbounded Path
        # would grow without limit and RViz redraws the whole strip each time.
        self.declare_parameter("path_min_spacing_m", 0.5)
        # Reject isolated estimator/ICP spikes so one bad sample cannot leave a
        # permanent hundreds-of-metres line in RViz and collapse the useful view.
        self.declare_parameter("path_max_jump_m", 5.0)
        self.declare_parameter("path_max_speed_mps", 20.0)
        # 0 means the whole mapping/localization run. The route must not silently
        # lose its beginning on a long course.
        self.declare_parameter("path_max_points", 0)
        self.declare_parameter("max_pose_time_difference_sec", 0.15)
        # A loaded localization map and ESKF map pose are already usable when
        # LiDAR is absent. Mapping mode still requires LiDAR to establish the
        # map->odom edge, so its default remains true.
        self.declare_parameter("require_lidar_pose_for_map_odom", True)
        # In localization MOLA starts at the saved-map origin and may not be
        # reseeded until a trustworthy pose arrives. Do not draw that temporary
        # origin as driven motion; begin only when it agrees with the reference.
        # Zero keeps mapping-mode behavior unchanged.
        self.declare_parameter("lidar_path_state_proximity_m", 0.0)
        # Localization can use scan-to-map only as a one-shot stationary yaw
        # verifier. Its later pose is diagnostic and must not become the driven
        # path on a repetitive road; the RTK/IMU state path remains authoritative.
        self.declare_parameter("publish_lidar_path", True)
        # Which pose becomes the TF edge ending at the vehicle frame - see
        # BASE_LINK_TF_SOURCES. Needed for any RViz view whose Fixed Frame is
        # base_link: with "none" the TF tree is two disconnected islands
        # (map/utm/enu and base_link/sensors) and only a map-fixed view can
        # resolve anything. In localization "state" (the RTK pose) and "lidar"
        # (the ICP pose) both give map->base_link directly; in mapping the LiDAR
        # pose is in odom, so "lidar" gives odom->base_link under this node's
        # own map->odom.
        self.declare_parameter("base_link_tf_source", "none")
        # "auto" follows the state pose, i.e. the pose drawn as the vehicle.
        # "auto" only: how long to listen on /tf for another owner of the
        # vehicle frame before publishing this node's own edge.
        self.declare_parameter("base_link_tf_auto_grace_sec", 2.0)

        self._path_frame = self.get_parameter("path_frame").value
        self._min_spacing = float(self.get_parameter("path_min_spacing_m").value)
        self._max_jump = float(self.get_parameter("path_max_jump_m").value)
        self._max_speed = float(self.get_parameter("path_max_speed_mps").value)
        self._max_points = int(self.get_parameter("path_max_points").value)
        self._require_lidar_pose_for_map_odom = bool(
            self.get_parameter("require_lidar_pose_for_map_odom").value
        )
        self._lidar_path_state_proximity = float(
            self.get_parameter("lidar_path_state_proximity_m").value
        )
        self._publish_lidar_path = bool(
            self.get_parameter("publish_lidar_path").value
        )
        if self._lidar_path_state_proximity < 0.0:
            raise ValueError("lidar_path_state_proximity_m must be non-negative")
        self._lidar_path_ready = self._lidar_path_state_proximity == 0.0
        self._latest_state_pose = None
        self._base_link_tf_source = validate_base_link_tf_source(
            self.get_parameter("base_link_tf_source").value
        )
        self._effective_base_link_tf_source = self._base_link_tf_source
        self._auto_base_link_gate = None
        self._auto_base_link_announced = False
        self._auto_base_link_yielded = False
        if self._base_link_tf_source == "auto":
            # The state pose, not the LiDAR one: a base_link-fixed view has to
            # ride on the pose drawn as the vehicle. Following the LiDAR pose on
            # 20260914_220126 put the vehicle marker 1.4-11 m from the view's
            # origin, because scan matching against that map was off by that much.
            self._effective_base_link_tf_source = "state"
            grace_sec = float(
                self.get_parameter("base_link_tf_auto_grace_sec").value
            )
            self._auto_base_link_gate = AutoBaseLinkTfGate(
                int(round(grace_sec * 1e9)), time.monotonic_ns()
            )
        self._max_pose_time_difference_ns = int(
            round(
                float(
                    self.get_parameter("max_pose_time_difference_sec").value
                )
                * 1e9
            )
        )

        # Keep enough timestamped LO poses to match each smoother update instead
        # of combining it with an unrelated "latest" pose while the car moves.
        self._lidar_poses = deque(maxlen=100)
        self._state_poses = deque(maxlen=100)
        self._last_transform_stamp_ns = None
        # map->odom is the edge the whole UTM route projection hangs off. When it is
        # never published the run silently ends with an empty waypoints.csv, so the
        # two "give up quietly" paths in _broadcast_map_odom() report themselves.
        self._map_odom_published = False

        self._bc = tf2_ros.TransformBroadcaster(self)

        self._path = Path()
        self._path.header.frame_id = self._path_frame
        self._raw_path_poses: list[PoseStamped] = []
        self._raw_path_frame: str | None = None
        self._latest_path_transform = None
        self._last_local_xy = None
        self._last_local_path_position = None
        self._last_local_path_stamp_ns = None
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._path_pub = self.create_publisher(
            Path, self.get_parameter("path_topic").value, latched_qos
        )

        # Keep the global vehicle estimates as separate diagnostic trajectories.
        # Neither is substituted for the LiDAR-driven path above: the lines exist
        # to expose disagreement between local LiDAR motion, the unfused RTK/IMU
        # reference, and the fused GNSS/IMU/LiDAR estimate.
        path_limits = dict(
            frame=self._path_frame,
            qos=latched_qos,
            min_spacing_m=self._min_spacing,
            max_jump_m=self._max_jump,
            max_speed_mps=self._max_speed,
            max_points=self._max_points,
        )
        self._state_path = MapFramePath(
            self,
            label="GNSS/IMU",
            topic=self.get_parameter("fused_path_topic").value,
            **path_limits,
        )
        self._fused_localization_path = MapFramePath(
            self,
            label="fused",
            topic=self.get_parameter("fused_localization_path_topic").value,
            **path_limits,
        )
        self._vehicle_marker_dimensions = (
            float(self.get_parameter("vehicle_marker_length_m").value),
            float(self.get_parameter("vehicle_marker_width_m").value),
            float(self.get_parameter("vehicle_marker_height_m").value),
        )
        self._vehicle_marker_z_offset = float(
            self.get_parameter("vehicle_marker_z_offset_m").value
        )
        self._vehicle_marker_color = tuple(
            float(value)
            for value in self.get_parameter("vehicle_marker_color_rgba").value
        )
        self._vehicle_marker_pub = self.create_publisher(
            Marker, self.get_parameter("vehicle_marker_topic").value, 1
        )

        self.create_subscription(
            Odometry, self.get_parameter("lidar_pose_topic").value, self._on_lidar, 20
        )
        self.create_subscription(
            Odometry, self.get_parameter("state_pose_topic").value, self._on_state, 20
        )
        if self._auto_base_link_gate is not None:
            self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.create_subscription(
            Odometry,
            self.get_parameter("fused_localization_pose_topic").value,
            self._on_fused_localization,
            20,
        )

        lidar_tf_status = (
            ""
            if self._base_link_tf_source == "none"
            else (
                f", owning the TF edge to the vehicle frame from the "
                f"{self._effective_base_link_tf_source} pose"
                + (
                    " unless another node already publishes it (auto)"
                    if self._auto_base_link_gate is not None
                    else ""
                )
            )
        )
        self.get_logger().info(
            "map_odom_tf_node: bridging "
            f"'{self.get_parameter('state_pose_topic').value}' + "
            f"'{self.get_parameter('lidar_pose_topic').value}' -> map->odom TF, "
            f"publishing the complete driven path in '{self._path_frame}', "
            "overlaying the fused estimate from "
            f"'{self.get_parameter('fused_localization_pose_topic').value}' on "
            f"'{self.get_parameter('fused_localization_path_topic').value}'"
            f"{lidar_tf_status}"
        )

    def _broadcast_base_link_tf(self, source: str, msg: Odometry) -> None:
        """Publish this pose as TF when it is the configured vehicle-frame source."""

        if self._effective_base_link_tf_source != source:
            return
        try:
            transform = odometry_to_transform(msg)
        except ValueError as error:
            self.get_logger().warn(
                f"Not broadcasting the {source} pose as TF: {error}",
                throttle_duration_sec=5.0,
            )
            return
        gate = self._auto_base_link_gate
        if gate is not None:
            parent = transform.header.frame_id
            child = transform.child_frame_id
            if not gate.may_publish(child, time.monotonic_ns()):
                owner = gate.foreign_parent(child)
                if owner is not None and not self._auto_base_link_yielded:
                    self._auto_base_link_yielded = True
                    self.get_logger().info(
                        f"base_link_tf_source=auto: '{owner}'->'{child}' is "
                        "already published by another node (e.g. the vehicle "
                        f"ESKF); not publishing the {source} pose as TF."
                    )
                return
            if not self._auto_base_link_announced:
                self._auto_base_link_announced = True
                self.get_logger().info(
                    f"base_link_tf_source=auto: no other owner of '{child}' on "
                    f"/tf; publishing the {source} pose as '{parent}'->'{child}'."
                )
            gate.note_published(
                parent,
                child,
                transform.header.stamp.sec * 1_000_000_000
                + transform.header.stamp.nanosec,
            )
        self._bc.sendTransform(transform)

    def _on_tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            parent = transform.header.frame_id.strip()
            child = transform.child_frame_id.strip()
            stamp_ns = (
                transform.header.stamp.sec * 1_000_000_000
                + transform.header.stamp.nanosec
            )
            if self._auto_base_link_gate.observe(parent, child, stamp_ns):
                self._auto_base_link_yielded = True
                self.get_logger().warn(
                    f"base_link_tf_source=auto: another node now publishes "
                    f"'{parent}'->'{child}'; stopped publishing this node's "
                    f"edge ('{child}' may have exactly one parent)."
                )

    def _on_lidar(self, msg: Odometry) -> None:
        self._broadcast_base_link_tf("lidar", msg)

        self._lidar_poses.append((self._stamp_ns(msg), msg))
        if self._publish_lidar_path and self._lidar_path_is_trustworthy(msg):
            self._append_path(msg)
        # A state estimate may arrive ahead of the next LiDAR sample. Retry the
        # newest pending estimates now that a bracketing sample is available.
        for _, state_msg in reversed(self._state_poses):
            if self._broadcast_map_odom(state_msg):
                break

    def _on_state(self, msg: Odometry) -> None:
        self._latest_state_pose = msg
        self._broadcast_base_link_tf("state", msg)
        self._state_poses.append((self._stamp_ns(msg), msg))
        try:
            marker = vehicle_marker_from_odometry(
                msg,
                *self._vehicle_marker_dimensions,
                self._vehicle_marker_z_offset,
                self._vehicle_marker_color,
            )
        except ValueError as error:
            self.get_logger().warn(
                f"Not publishing vehicle marker: {error}",
                throttle_duration_sec=5.0,
            )
        else:
            self._vehicle_marker_pub.publish(marker)
        self._state_path.append(msg)
        self._broadcast_map_odom(msg)

    def _on_fused_localization(self, msg: Odometry) -> None:
        """Draw the fused estimator pose only; map->odom keeps its own source.

        This trajectory is a comparison overlay, so it deliberately drives no TF
        and no waypoint projection: switching the state estimator must not
        silently move the route that the rest of the run is saved against.
        """

        self._fused_localization_path.append(msg)

    @staticmethod
    def _stamp_ns(msg: Odometry) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    def _broadcast_map_odom(self, msg: Odometry) -> bool:
        if not self._lidar_poses:
            if not self._require_lidar_pose_for_map_odom:
                # Nothing needs bridging yet: the state, loaded map, route and
                # vehicle marker are all already expressed in map. If LiDAR
                # starts later, its callback retries this method normally.
                return True
            # Only worth reporting while map->odom has never been established. Once
            # it has, _on_lidar() retries the buffered state poses on every new LiDAR
            # sample, so an individual miss is self-healing and not operator-relevant.
            if not self._map_odom_published:
                self.get_logger().warning(
                    "map->odom not established yet: no LiDAR-odometry pose received "
                    f"on '{self.get_parameter('lidar_pose_topic').value}'. Until this "
                    "resolves, no route can be projected into map/utm.",
                    throttle_duration_sec=10.0,
                )
            return False

        state_stamp_ns = self._stamp_ns(msg)
        if state_stamp_ns == self._last_transform_stamp_ns:
            return True
        # Localization-mode MOLA already publishes map<-base. No timestamp
        # interpolation is needed to discover that map<-map is identity, and
        # requiring a 0.15 s match here previously prevented this branch from
        # being reached at all when the two publishers used different stamps.
        latest_lidar = self._lidar_poses[-1][1]
        if latest_lidar.header.frame_id == msg.header.frame_id:
            try:
                transform = compose_map_to_odom(msg, latest_lidar)
            except ValueError as error:
                self.get_logger().warn(
                    f"Not accepting same-frame localization poses: {error}",
                    throttle_duration_sec=5.0,
                )
                return False
            assert transform is None
            self._latest_path_transform = None
            self._last_transform_stamp_ns = state_stamp_ns
            if not self._map_odom_published:
                self._map_odom_published = True
                self.get_logger().info(
                    "State and LiDAR poses are already in map; no map->map TF "
                    "will be published."
                )
            self._rebuild_path()
            return True
        try:
            matched = interpolate_pose_by_stamp(
                self._lidar_poses,
                state_stamp_ns,
                self._max_pose_time_difference_ns,
            )
        except ValueError as error:
            self.get_logger().warn(
                f"Not updating map->odom: {error}",
                throttle_duration_sec=5.0,
            )
            return False
        if matched is None:
            # A state pose newer than the newest buffered LiDAR pose has no bracketing
            # pair yet; _on_lidar() retries it on the next LiDAR sample, so in steady
            # state this is routine and must not be reported. It is only a problem
            # while map->odom has never been established at all - sustained, that is
            # what silently produces a run with an empty waypoints.csv.
            if not self._map_odom_published:
                oldest = self._lidar_poses[0][0]
                newest = self._lidar_poses[-1][0]
                self.get_logger().warning(
                    "map->odom not established yet: state pose at t=%.3f has no "
                    "LiDAR pose within %.3f s (buffered LiDAR span %.3f..%.3f). "
                    "Check that both streams use the same clock."
                    % (
                        state_stamp_ns * 1.0e-9,
                        self._max_pose_time_difference_ns * 1.0e-9,
                        oldest * 1.0e-9,
                        newest * 1.0e-9,
                    ),
                    throttle_duration_sec=10.0,
                )
            return False
        _, lidar_msg = matched

        try:
            transform = compose_map_to_odom(msg, lidar_msg)
        except ValueError as error:
            self.get_logger().warn(
                f"Not updating map->odom: {error}",
                throttle_duration_sec=5.0,
            )
            return False
        if transform is not None:
            self._bc.sendTransform(transform)
            self._latest_path_transform = transform
            if not self._map_odom_published:
                self._map_odom_published = True
                self.get_logger().info(
                    "First map->odom transform published; the UTM route projection "
                    "can now proceed."
                )
        else:
            self._latest_path_transform = None
        self._last_transform_stamp_ns = state_stamp_ns
        self._rebuild_path()
        return True

    def _lidar_path_is_trustworthy(self, msg: Odometry) -> bool:
        if self._lidar_path_state_proximity == 0.0:
            return True
        if self._latest_state_pose is None:
            return False
        distance = same_frame_pose_distance_xy(msg, self._latest_state_pose)
        if distance is None or distance > self._lidar_path_state_proximity:
            if self._lidar_path_ready:
                self.get_logger().warning(
                    "Pausing driven path: MOLA pose differs from the vehicle "
                    "pose by %s"
                    % (
                        "different frames"
                        if distance is None
                        else f"{distance:.1f} m"
                    )
                )
                self._clear_raw_path(msg)
            self._lidar_path_ready = False
            return False
        if not self._lidar_path_ready:
            self._clear_raw_path(msg)
            self._lidar_path_ready = True
            self.get_logger().info(
                "MOLA pose now agrees with the vehicle pose; starting driven path"
            )
        return True

    def _clear_raw_path(self, msg: Odometry) -> None:
        self._raw_path_poses.clear()
        self._raw_path_frame = None
        self._last_local_xy = None
        self._last_local_path_position = None
        self._last_local_path_stamp_ns = None
        self._path.poses = []
        self._path.header.frame_id = self._path_frame
        self._path.header.stamp = msg.header.stamp
        self._path_pub.publish(self._path)

    def _append_path(self, msg: Odometry) -> None:
        source_frame = msg.header.frame_id.strip()
        if not source_frame:
            return
        if self._raw_path_frame is None:
            self._raw_path_frame = source_frame
        elif self._raw_path_frame != source_frame:
            self.get_logger().warning(
                "LiDAR odometry path frame changed from "
                f"'{self._raw_path_frame}' to '{source_frame}'; "
                "starting a new driven-path history."
            )
            self._raw_path_poses.clear()
            self._raw_path_frame = source_frame
            self._last_local_xy = None
            self._last_local_path_position = None
            self._last_local_path_stamp_ns = None

        pose = msg.pose.pose
        x, y = pose.position.x, pose.position.y
        z = pose.position.z
        stamp_ns = self._stamp_ns(msg)
        if (
            self._last_local_path_stamp_ns is not None
            and stamp_ns < self._last_local_path_stamp_ns
        ):
            self._raw_path_poses.clear()
            self._last_local_xy = None
            self._last_local_path_position = None
            self._last_local_path_stamp_ns = None

        if self._last_local_path_position is not None:
            dx = x - self._last_local_path_position[0]
            dy = y - self._last_local_path_position[1]
            dz = z - self._last_local_path_position[2]
            step_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            elapsed_sec = 0.0
            if self._last_local_path_stamp_ns is not None:
                elapsed_sec = (
                    stamp_ns - self._last_local_path_stamp_ns
                ) * 1.0e-9
            jump_limit = path_jump_limit_m(
                self._max_jump, self._max_speed, elapsed_sec
            )
            if self._max_jump > 0.0 and step_3d > jump_limit:
                self.get_logger().warning(
                    "Rejected implausible driven-path pose jump: "
                    f"{step_3d:.1f} m > {jump_limit:.1f} m"
                )
                return

        if self._last_local_xy is not None:
            if (
                math.hypot(
                    x - self._last_local_xy[0],
                    y - self._last_local_xy[1],
                )
                < self._min_spacing
            ):
                return
        self._last_local_xy = (x, y)
        self._last_local_path_position = (x, y, z)
        self._last_local_path_stamp_ns = stamp_ns

        pose_local = PoseStamped()
        pose_local.header = msg.header
        pose_local.pose = pose
        self._raw_path_poses.append(pose_local)
        if (
            self._max_points > 0
            and len(self._raw_path_poses) > self._max_points
        ):
            del self._raw_path_poses[0]
        self._rebuild_path()

    def _rebuild_path(self) -> None:
        if not self._raw_path_poses or self._raw_path_frame is None:
            return

        if self._raw_path_frame == self._path_frame:
            projected = list(self._raw_path_poses)
        else:
            transform = self._latest_path_transform
            if (
                transform is None
                or transform.header.frame_id != self._path_frame
                or transform.child_frame_id != self._raw_path_frame
            ):
                return
            projected = [
                transform_pose_stamped(pose, transform)
                for pose in self._raw_path_poses
            ]

        self._path.poses = projected
        self._path.header.frame_id = self._path_frame
        self._path.header.stamp = self._raw_path_poses[-1].header.stamp
        self._path_pub.publish(self._path)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomTfNode()
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
