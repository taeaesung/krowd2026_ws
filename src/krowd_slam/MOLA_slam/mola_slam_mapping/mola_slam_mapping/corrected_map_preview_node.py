"""Live RViz preview of the map placed on the RTK-corrected trajectory.

The accumulated cloud RViz shows during mapping is LidarOdometry's rolling
local map. That is the map ICP registers against, so it lives in the `odom`
frame and carries LO's drift - locally excellent, globally displaced and, on a
flat site, visibly tilted. Nothing may move it: correcting the pose that feeds
ICP is exactly what was measured to break registration (see the smoother A/B
in config/mapping_system.yaml).

So this node answers "what would the map look like if it were built on the red
line?" while the run is still going, without touching anything the SLAM stack
uses. It subscribes only - to the raw scans and to /fused_localization/pose -
keeps its own voxelized copy placed by the fused pose, and publishes it in
`map` on its own topic. Nothing in MOLA consumes it.

IT IS A PREVIEW, NOT THE PRODUCT. Each scan is placed at a single pose (no
deskew), and a keyframe is never revisited once accumulated, so what it shows
is the quality of the fused (purple) trajectory at the instant each scan
arrived. The map to localize against is still the one scripts/retraject_map.sh
rebuilds after the run, where the whole trajectory is optimized at once and
every scan is the deskewed copy the simplemap stored.

Timing is the same trap as everywhere else in this package: the fused pose for
a scan's instant is not necessarily published when the scan arrives, so scans
are queued and drained once a pose brackets them (see pose_fusion_node's
docstring). Scans that age out of the queue unmatched are dropped - a preview
must never stall the run to wait for them.
"""

import numpy as np
import rclpy
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformException

from mola_slam_mapping.map_odom_tf_node import _mat_from_pose, interpolate_pose_by_stamp

# 21 bits per axis, so a voxel index must stay inside +/-2^20 - at 0.25 m that
# is +/-262 km from the map origin, far beyond any course.
_VOXEL_BITS = 21
_VOXEL_MASK = (1 << _VOXEL_BITS) - 1
_VOXEL_LIMIT = 1 << (_VOXEL_BITS - 1)


def voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    """One int64 identifying each point's voxel, for fast de-duplication."""

    if size <= 0.0:
        raise ValueError("voxel size must be positive")
    indices = np.floor(points / size).astype(np.int64)
    if len(indices) and np.abs(indices).max() >= _VOXEL_LIMIT:
        raise ValueError("point too far from the map origin to index")
    return (
        ((indices[:, 0] & _VOXEL_MASK) << (2 * _VOXEL_BITS))
        | ((indices[:, 1] & _VOXEL_MASK) << _VOXEL_BITS)
        | (indices[:, 2] & _VOXEL_MASK)
    )


def deduplicate_voxels(points: np.ndarray, size: float) -> np.ndarray:
    """Keep the first point of each occupied voxel, in first-seen order."""

    if len(points) == 0:
        return points
    _, index = np.unique(voxel_keys(points, size), return_index=True)
    return points[np.sort(index)]


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) array."""

    if len(points) == 0:
        return points
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def cloud_from_points(points: np.ndarray, frame: str, stamp) -> PointCloud2:
    """XYZ float32 PointCloud2, built without a per-point Python loop."""

    data = np.ascontiguousarray(points, dtype=np.float32)
    message = PointCloud2()
    message.header.frame_id = frame
    message.header.stamp = stamp
    message.height = 1
    message.width = len(data)
    message.fields = [
        PointField(name=name, offset=4 * index, datatype=PointField.FLOAT32, count=1)
        for index, name in enumerate(("x", "y", "z"))
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12 * len(data)
    message.is_dense = True
    message.data = data.tobytes()
    return message


class CorrectedMapPreviewNode(Node):
    def __init__(self):
        super().__init__("corrected_map_preview_node")

        self.declare_parameter("pointcloud_topic", "/ouster/points")
        self.declare_parameter("pose_topic", "/fused_localization/pose")
        self.declare_parameter("output_topic", "/mola/viz/corrected_map_points")
        self.declare_parameter("base_frame", "base_link")
        # Same resolution as the RViz voxel filter applied to MOLA's own maps,
        # so the two views are visually comparable.
        self.declare_parameter("voxel_size_m", 0.25)
        # Mirrors mapping.launch.py's MOLA_MAXIMUM_RANGE_FILTER / the rebuild
        # pipeline's FilterByRange, so the preview covers the same returns.
        self.declare_parameter("min_range_m", 2.0)
        self.declare_parameter("max_range_m", 75.0)
        # Each publication copies the whole accumulated cloud, so this is a
        # cost/latency trade, not a data one. 40 scans is ~4 s.
        self.declare_parameter("publish_every_n_scans", 40)
        # More generous than pose_fusion_node's 0.15 s on purpose. A scan needs
        # a fused pose on each side of its stamp, and /lidar_odometry/pose -
        # which is what paces the fused pose - runs at a median 0.100 s but a
        # p95 of 0.2-0.3 s, so 0.15 s would silently drop whole stretches. A
        # preview already places each scan at a single pose (no deskew), so
        # interpolating across a slightly longer gap costs it nothing it has not
        # already paid.
        self.declare_parameter("max_pose_time_difference_sec", 0.30)
        # ~20 s of scans: long enough for pose_fusion_node's back-fill at the
        # first correction to arrive, short enough to bound RAM.
        self.declare_parameter("max_pending_scans", 200)
        # A preview that has eaten all the RAM is worse than a coarse one.
        self.declare_parameter("max_points", 6_000_000)
        self.declare_parameter("time_rewind_tolerance_sec", 0.1)

        self._base_frame = str(self.get_parameter("base_frame").value)
        self._voxel = float(self.get_parameter("voxel_size_m").value)
        self._min_range = float(self.get_parameter("min_range_m").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._publish_every = int(self.get_parameter("publish_every_n_scans").value)
        self._max_delta_ns = int(
            round(float(self.get_parameter("max_pose_time_difference_sec").value) * 1e9)
        )
        self._max_pending = int(self.get_parameter("max_pending_scans").value)
        self._max_points = int(self.get_parameter("max_points").value)
        self._rewind_tolerance_ns = int(
            round(float(self.get_parameter("time_rewind_tolerance_sec").value) * 1e9)
        )

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._map_frame: str | None = None
        self._poses: list[tuple[int, Odometry]] = []
        self._pending: list[tuple[int, np.ndarray]] = []
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._last_scan_stamp_ns: int | None = None
        self._scans_accumulated = 0
        self._scans_dropped = 0
        self._since_publish = 0
        self._capacity_warned = False

        # Transient-local so an RViz started mid-run gets the whole preview.
        self._publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        # Best-effort: the LiDAR publisher may be either, and a best-effort
        # subscription is compatible with both.
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("pointcloud_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("pose_topic").value), self._on_pose, 50
        )
        self.create_timer(10.0, self._report)
        self.get_logger().info(
            f"corrected_map_preview_node: accumulating "
            f"'{self.get_parameter('pointcloud_topic').value}' placed by "
            f"'{self.get_parameter('pose_topic').value}' -> "
            f"'{self.get_parameter('output_topic').value}' "
            f"({self._voxel} m voxels). Preview only; nothing in MOLA reads it."
        )

    @staticmethod
    def _stamp_ns(message) -> int:
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def _reset(self) -> None:
        self._poses.clear()
        self._pending.clear()
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._scans_accumulated = 0
        self._scans_dropped = 0

    def _on_pose(self, message: Odometry) -> None:
        frame = message.header.frame_id.strip()
        if not frame:
            return
        if self._map_frame is not None and frame != self._map_frame:
            self.get_logger().warning(
                f"Pose frame changed '{self._map_frame}' -> '{frame}'; restarting the preview."
            )
            self._reset()
        self._map_frame = frame
        self._poses.append((self._stamp_ns(message), message))
        # pose_fusion_node back-fills the start of the run at its first
        # correction, so poses do not always arrive in order.
        if len(self._poses) > self._max_pending + 50:
            self._poses = sorted(self._poses, key=lambda item: item[0])[-self._max_pending:]
        self._drain_pending()

    def _on_cloud(self, message: PointCloud2) -> None:
        stamp_ns = self._stamp_ns(message)
        if (
            self._last_scan_stamp_ns is not None
            and stamp_ns < self._last_scan_stamp_ns - self._rewind_tolerance_ns
        ):
            self.get_logger().info("Scan time moved backwards; restarting the preview.")
            self._reset()
        self._last_scan_stamp_ns = stamp_ns

        try:
            points = self._scan_in_base_link(message)
        except (TransformException, ValueError) as error:
            self.get_logger().warning(
                f"Skipping scan: {error}", throttle_duration_sec=5.0
            )
            return

        self._pending.append((stamp_ns, points))
        if len(self._pending) > self._max_pending:
            del self._pending[0]
            self._scans_dropped += 1
        self._drain_pending()

    def _scan_in_base_link(self, message: PointCloud2) -> np.ndarray:
        source = message.header.frame_id.strip()
        if not source:
            raise ValueError("PointCloud2 frame_id is empty")
        values = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.column_stack((values["x"], values["y"], values["z"])).astype(float)
        distance = np.linalg.norm(points, axis=1)
        points = points[(distance >= self._min_range) & (distance <= self._max_range)]
        # Decimate once, here, so the queue and every later step stay cheap.
        points = deduplicate_voxels(points, self._voxel)
        if source == self._base_frame:
            return points
        # The sensor edge is static, so the latest transform is the right one
        # and asking for it at the scan stamp would only race /tf_static.
        transform = self._tf_buffer.lookup_transform(
            self._base_frame, source, Time(), timeout=Duration(seconds=0.05)
        ).transform
        pose = Odometry().pose.pose
        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z
        pose.orientation = transform.rotation
        return transform_points(points, np.asarray(_mat_from_pose(pose)))

    def _drain_pending(self) -> None:
        if not self._poses or not self._pending:
            return
        newest_pose_stamp = max(stamp for stamp, _ in self._poses)
        oldest_pose_stamp = min(stamp for stamp, _ in self._poses)
        still_pending = []
        for stamp_ns, points in self._pending:
            try:
                matched = interpolate_pose_by_stamp(
                    self._poses, stamp_ns, self._max_delta_ns
                )
            except ValueError as error:
                self.get_logger().warning(
                    f"Not placing a scan: {error}", throttle_duration_sec=5.0
                )
                continue
            if matched is None:
                if stamp_ns > newest_pose_stamp or stamp_ns < oldest_pose_stamp:
                    # A later pose may still bracket it, or the back-fill may
                    # still reach back to it.
                    still_pending.append((stamp_ns, points))
                else:
                    self._scans_dropped += 1
                continue
            self._accumulate(transform_points(points, np.asarray(_mat_from_pose(matched[1].pose.pose))))
        self._pending = still_pending

    def _accumulate(self, points_in_map: np.ndarray) -> None:
        if len(self._points) >= self._max_points:
            if not self._capacity_warned:
                self._capacity_warned = True
                self.get_logger().warning(
                    f"Preview reached max_points ({self._max_points}); it stops growing "
                    "from here. Raise max_points or voxel_size_m for a longer course."
                )
            return
        self._points = np.vstack(
            (self._points, np.asarray(points_in_map, dtype=np.float32))
        )
        self._scans_accumulated += 1
        self._since_publish += 1
        if self._since_publish >= self._publish_every:
            self._since_publish = 0
            self._publish()

    def _publish(self) -> None:
        if self._map_frame is None or len(self._points) == 0:
            return
        # De-duplicating only here keeps the per-scan path a cheap append.
        self._points = deduplicate_voxels(self._points, self._voxel)
        self._publisher.publish(
            cloud_from_points(self._points, self._map_frame, self.get_clock().now().to_msg())
        )

    def _report(self) -> None:
        # Dropped scans have to be in the trigger: a preview that matches
        # nothing at all drops every scan and would otherwise say nothing,
        # which is exactly the case an operator needs told.
        if not (self._scans_accumulated or self._pending or self._scans_dropped):
            return
        message = (
            "corrected preview: %d scans placed, %d points, %d queued for a pose, %d dropped"
            % (
                self._scans_accumulated,
                len(self._points),
                len(self._pending),
                self._scans_dropped,
            )
        )
        if self._scans_accumulated:
            self.get_logger().info(message)
        else:
            self.get_logger().warning(
                message
                + " - no scan could be placed yet. Every scan needs a "
                f"'{self.get_parameter('pose_topic').value}' on each side of its "
                "stamp, within max_pose_time_difference_sec."
            )


def main(args=None):
    rclpy.init(args=args)
    node = CorrectedMapPreviewNode()
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
