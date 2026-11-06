"""Put the live lane detections on the localization map, and keep 10 s of them.

han_krowd (GLane3D) publishes the lanes it sees on /glane3d/lane_markers as
LINE_STRIPs in base_link, stamped with the camera image time. Drawn as they
are, RViz would place them through TF - and on the vehicle the TF edge to
base_link belongs to the sensor-stack ESKF, whose ENU origin is not the map's:
map->base_link through TF was 5.7 m away from /gnss_imu/pose on 2026-09-19.
So this node places every detection itself, with the SAME pose path tracking
steers on (/gnss_imu/pose), interpolated to the image time (the detections
arrive ~150 ms after the image; 0.6 m at 15 km/h).

Outputs (none of them depends on TF):
    /lane_map/current        MarkerArray, map        lanes of the latest frame
    /lane_map/history        MarkerArray, map        every frame of the last
                                                     history_sec, fading with age
    /lane_map/current_base   MarkerArray, base_link  the same two, re-expressed
    /lane_map/history_base   MarkerArray, base_link  in the vehicle's CURRENT
                                                     localized pose - for an
                                                     RViz view with Fixed Frame
                                                     base_link
    /lane_map/history_points PointCloud2, map        the history as data, for
                                                     algorithms: x y z class_id
                                                     track_id age_s stamp

Only the lanes are carried over; han_krowd's own path (/planning/lane_path)
and stop lines are not subscribed to. The detections' z (a fixed ~0.8 m) is
discarded: lanes are drawn on the plane lane_z_offset_m below base_link.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

# han_krowd's lane classes; the marker namespace is "<class>_track_<id>" with
# an optional "_2d_fallback" suffix (yolo_depth_improved_node._lane_marker_array).
LANE_CLASS_IDS = {"leftSolid": 0, "leftDashed": 1, "rightSolid": 2, "rightDashed": 3}
UNKNOWN_CLASS_ID = 9
FALLBACK_SUFFIX = "_2d_fallback"


@dataclass(frozen=True)
class Lane:
    """One detected lane line, in base_link or map depending on the stage."""

    class_name: str
    track_id: int
    fallback: bool
    color: tuple[float, float, float]
    width_m: float
    points: np.ndarray  # (N, 2) x, y


@dataclass(frozen=True)
class Pose2D:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float
    speed_mps: float = 0.0
    yaw_rate: float = 0.0


def parse_namespace(ns: str) -> tuple[str, int, bool]:
    """(class name, track id, 2D fallback) from a han_krowd lane namespace."""

    fallback = ns.endswith(FALLBACK_SUFFIX)
    if fallback:
        ns = ns[: -len(FALLBACK_SUFFIX)]
    name, _, track = ns.partition("_track_")
    try:
        track_id = int(track) if track else -1
    except ValueError:
        track_id = -1
    return name, track_id, fallback


def lanes_from_markers(message: MarkerArray) -> tuple[int | None, list[Lane]]:
    """(stamp_ns, lanes) of one /glane3d/lane_markers publication."""

    stamp_ns = None
    lanes = []
    for marker in message.markers:
        if marker.action != Marker.ADD or marker.type != Marker.LINE_STRIP:
            continue
        if len(marker.points) < 2:
            continue
        stamp_ns = marker.header.stamp.sec * 1_000_000_000 + marker.header.stamp.nanosec
        name, track_id, fallback = parse_namespace(marker.ns)
        lanes.append(
            Lane(
                class_name=name,
                track_id=track_id,
                fallback=fallback,
                color=(marker.color.r, marker.color.g, marker.color.b),
                width_m=float(marker.scale.x),
                points=np.array([[p.x, p.y] for p in marker.points], dtype=float),
            )
        )
    return stamp_ns, lanes


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class PoseBuffer:
    """Recent localized poses, answered at an arbitrary stamp."""

    def __init__(self, max_gap_ns: int, max_extrapolation_ns: int, maxlen: int = 200):
        self._poses: deque[Pose2D] = deque(maxlen=maxlen)
        self._max_gap_ns = max_gap_ns
        self._max_extrapolation_ns = max_extrapolation_ns

    def add(self, pose: Pose2D) -> None:
        if self._poses and pose.stamp_ns <= self._poses[-1].stamp_ns:
            if pose.stamp_ns + 1_000_000_000 < self._poses[-1].stamp_ns:
                self._poses.clear()  # time went backwards: a bag restarted
            else:
                return
        self._poses.append(pose)

    @property
    def latest(self) -> Pose2D | None:
        return self._poses[-1] if self._poses else None

    def at(self, stamp_ns: int) -> Pose2D | None:
        """Interpolated inside the buffer, extrapolated a little past its end."""

        poses = self._poses
        if not poses:
            return None
        last = poses[-1]
        if stamp_ns >= last.stamp_ns:
            dt_ns = stamp_ns - last.stamp_ns
            if dt_ns > self._max_extrapolation_ns:
                return None
            dt = dt_ns * 1e-9
            yaw = last.yaw + last.yaw_rate * dt
            mid = last.yaw + 0.5 * last.yaw_rate * dt
            return Pose2D(
                stamp_ns,
                last.x + last.speed_mps * dt * math.cos(mid),
                last.y + last.speed_mps * dt * math.sin(mid),
                last.z,
                wrap_angle(yaw),
                last.speed_mps,
                last.yaw_rate,
            )
        if stamp_ns < poses[0].stamp_ns:
            return None
        for older, newer in zip(reversed(list(poses)[:-1]), reversed(list(poses)[1:])):
            if older.stamp_ns <= stamp_ns <= newer.stamp_ns:
                span = newer.stamp_ns - older.stamp_ns
                if span > self._max_gap_ns:
                    return None
                t = 0.0 if span == 0 else (stamp_ns - older.stamp_ns) / span
                return Pose2D(
                    stamp_ns,
                    older.x + t * (newer.x - older.x),
                    older.y + t * (newer.y - older.y),
                    older.z + t * (newer.z - older.z),
                    wrap_angle(older.yaw + t * wrap_angle(newer.yaw - older.yaw)),
                    newer.speed_mps,
                    newer.yaw_rate,
                )
        return None


def base_to_map(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return np.column_stack(
        (
            pose.x + c * points[:, 0] - s * points[:, 1],
            pose.y + s * points[:, 0] + c * points[:, 1],
        )
    )


def map_to_base(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    dx = points[:, 0] - pose.x
    dy = points[:, 1] - pose.y
    return np.column_stack((c * dx + s * dy, -s * dx + c * dy))


@dataclass(frozen=True)
class LaneFrame:
    stamp_ns: int
    z_map: float
    lanes: list[Lane]  # points in map


class LaneMapOverlayNode(Node):
    def __init__(self):
        super().__init__("lane_map_overlay_node")
        self.declare_parameter("lane_topic", "/glane3d/lane_markers")
        self.declare_parameter("pose_topic", "/gnss_imu/pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("history_sec", 10.0)
        self.declare_parameter("history_publish_rate_hz", 2.0)
        self.declare_parameter("history_line_width_m", 0.08)
        self.declare_parameter("history_min_alpha", 0.15)
        self.declare_parameter("max_pose_gap_sec", 0.5)
        self.declare_parameter("max_pose_extrapolation_sec", 0.5)
        self.declare_parameter("lane_z_offset_m", -0.2)
        self.declare_parameter("current_topic", "/lane_map/current")
        self.declare_parameter("history_topic", "/lane_map/history")
        self.declare_parameter("current_base_topic", "/lane_map/current_base")
        self.declare_parameter("history_base_topic", "/lane_map/history_base")
        self.declare_parameter("history_points_topic", "/lane_map/history_points")

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self._map_frame = str(g("map_frame"))
        self._base_frame = str(g("base_frame"))
        self._history_ns = int(float(g("history_sec")) * 1e9)
        self._history_width = float(g("history_line_width_m"))
        self._min_alpha = float(g("history_min_alpha"))
        self._z_offset = float(g("lane_z_offset_m"))
        self._poses = PoseBuffer(
            int(float(g("max_pose_gap_sec")) * 1e9),
            int(float(g("max_pose_extrapolation_sec")) * 1e9),
        )
        self._frames: deque[LaneFrame] = deque()
        self._dropped = 0

        self._current_pub = self.create_publisher(MarkerArray, str(g("current_topic")), 10)
        self._history_pub = self.create_publisher(MarkerArray, str(g("history_topic")), 10)
        self._current_base_pub = self.create_publisher(
            MarkerArray, str(g("current_base_topic")), 10
        )
        self._history_base_pub = self.create_publisher(
            MarkerArray, str(g("history_base_topic")), 10
        )
        self._points_pub = self.create_publisher(
            PointCloud2, str(g("history_points_topic")), 10
        )
        self.create_subscription(
            Odometry, str(g("pose_topic")), self._on_pose, qos_profile_sensor_data
        )
        self.create_subscription(MarkerArray, str(g("lane_topic")), self._on_lanes, 10)
        rate = float(g("history_publish_rate_hz"))
        if rate > 0.0:
            self.create_timer(1.0 / rate, self._publish_history)
        self.get_logger().info(
            f"lane_map_overlay_node: '{g('lane_topic')}' (base_link) placed with "
            f"'{g('pose_topic')}' at the image time -> '{g('current_topic')}' / "
            f"'{g('history_topic')}' ({g('history_sec')} s) in '{self._map_frame}', "
            f"'{g('current_base_topic')}' / '{g('history_base_topic')}' in "
            f"'{self._base_frame}', data on '{g('history_points_topic')}'"
        )

    # ---- inputs ----------------------------------------------------------
    def _on_pose(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._poses.add(
            Pose2D(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(msg.pose.pose.position.z),
                yaw,
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.angular.z),
            )
        )

    def _on_lanes(self, msg: MarkerArray) -> None:
        stamp_ns, lanes = lanes_from_markers(msg)
        if stamp_ns is None:
            # No lane in this frame: the current view empties, history stays.
            self._publish_current(None)
            return
        pose = self._poses.at(stamp_ns)
        if pose is None:
            self._dropped += 1
            self.get_logger().warning(
                f"no localized pose at the lane image time ({self._dropped} frames "
                "dropped so far) - is /gnss_imu/pose publishing, heading acquired?",
                throttle_duration_sec=5.0,
            )
            return
        frame = LaneFrame(
            stamp_ns,
            pose.z + self._z_offset,
            [
                Lane(
                    lane.class_name, lane.track_id, lane.fallback, lane.color,
                    lane.width_m, base_to_map(lane.points, pose),
                )
                for lane in lanes
            ],
        )
        if self._frames and stamp_ns + 1_000_000_000 < self._frames[-1].stamp_ns:
            self._frames.clear()  # bag restarted
        self._frames.append(frame)
        self._prune(stamp_ns)
        self._publish_current(frame)

    def _prune(self, newest_ns: int) -> None:
        while self._frames and newest_ns - self._frames[0].stamp_ns > self._history_ns:
            self._frames.popleft()

    # ---- outputs ---------------------------------------------------------
    def _header(self, frame_id: str, stamp_ns: int) -> Header:
        header = Header()
        header.frame_id = frame_id
        header.stamp.sec = stamp_ns // 1_000_000_000
        header.stamp.nanosec = stamp_ns % 1_000_000_000
        return header

    def _line(self, header, ns, marker_id, lane, xy, z, width, alpha) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = width
        marker.color = ColorRGBA(r=lane.color[0], g=lane.color[1], b=lane.color[2], a=alpha)
        marker.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y in xy]
        return marker

    def _deleteall(self, header) -> Marker:
        marker = Marker()
        marker.header = header
        marker.action = Marker.DELETEALL
        return marker

    def _publish_current(self, frame: LaneFrame | None) -> None:
        latest = self._poses.latest
        stamp_ns = frame.stamp_ns if frame is not None else (latest.stamp_ns if latest else 0)
        map_header = self._header(self._map_frame, stamp_ns)
        base_header = self._header(self._base_frame, latest.stamp_ns if latest else stamp_ns)
        in_map = MarkerArray(markers=[self._deleteall(map_header)])
        in_base = MarkerArray(markers=[self._deleteall(base_header)])
        if frame is not None:
            for index, lane in enumerate(frame.lanes):
                ns = f"{lane.class_name}_track_{lane.track_id}"
                if lane.fallback:
                    ns += FALLBACK_SUFFIX
                in_map.markers.append(
                    self._line(map_header, ns, index, lane, lane.points, frame.z_map,
                               lane.width_m, 1.0)
                )
                if latest is not None:
                    in_base.markers.append(
                        self._line(base_header, ns, index, lane,
                                   map_to_base(lane.points, latest), self._z_offset,
                                   lane.width_m, 1.0)
                    )
        self._current_pub.publish(in_map)
        self._current_base_pub.publish(in_base)

    def _publish_history(self) -> None:
        latest = self._poses.latest
        if not self._frames:
            return
        newest_ns = self._frames[-1].stamp_ns
        self._prune(newest_ns)
        map_header = self._header(self._map_frame, newest_ns)
        base_header = self._header(self._base_frame, latest.stamp_ns if latest else newest_ns)
        in_map = MarkerArray(markers=[self._deleteall(map_header)])
        in_base = MarkerArray(markers=[self._deleteall(base_header)])
        rows = []
        marker_id = 0
        for frame in self._frames:
            age_s = (newest_ns - frame.stamp_ns) * 1e-9
            fade = 1.0 - min(1.0, age_s / max(self._history_ns * 1e-9, 1e-6))
            alpha = self._min_alpha + (1.0 - self._min_alpha) * fade
            for lane in frame.lanes:
                ns = f"history_{lane.class_name}"
                in_map.markers.append(
                    self._line(map_header, ns, marker_id, lane, lane.points, frame.z_map,
                               self._history_width, alpha)
                )
                if latest is not None:
                    in_base.markers.append(
                        self._line(base_header, ns, marker_id, lane,
                                   map_to_base(lane.points, latest), self._z_offset,
                                   self._history_width, alpha)
                    )
                marker_id += 1
                class_id = LANE_CLASS_IDS.get(lane.class_name, UNKNOWN_CLASS_ID)
                stamp_s = frame.stamp_ns * 1e-9
                for x, y in lane.points:
                    rows.append((x, y, frame.z_map, class_id, lane.track_id, age_s, stamp_s))
        self._history_pub.publish(in_map)
        self._history_base_pub.publish(in_base)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="class_id", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="track_id", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="age_s", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="stamp", offset=24, datatype=PointField.FLOAT64, count=1),
        ]
        self._points_pub.publish(point_cloud2.create_cloud(map_header, fields, rows))


def main(args=None):
    rclpy.init(args=args)
    node = LaneMapOverlayNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
