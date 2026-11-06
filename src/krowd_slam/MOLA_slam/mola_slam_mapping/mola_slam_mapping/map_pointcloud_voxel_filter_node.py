"""Voxel-filter MOLA map clouds in a process isolated from SLAM.

The LiDAR odometry map intentionally retains overlapping keyframe observations
for registration.  RViz does not need that density.  This node preserves the
first complete PointCloud2 record in each spatial voxel and republishes the
result, without changing the ICP map, SimpleMap, or saved global map.
"""

from copy import copy

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


MAP_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def republish_is_due(
    last_publish_ns: int | None, now_ns: int, interval_ns: int
) -> bool:
    """Should the last cloud be sent again to keep an RViz display alive?

    Changing RViz's Fixed Frame calls reset() on every topic display
    (rviz_common/ros_topic_display.hpp) and does NOT resubscribe, so a
    transient-local cloud that was published once - the loaded map in
    localization - is wiped and never redelivered. A slow repeat brings it back.

    Only when nothing was published recently: while mapping, MOLA sends map
    updates on its own and the timer must stay out of the way.
    """

    if interval_ns <= 0 or last_publish_ns is None:
        return False
    return now_ns - last_publish_ns >= interval_ns


def voxel_filter_cloud(message: PointCloud2, voxel_size_m: float) -> PointCloud2:
    """Return an unorganized cloud containing the first point in each voxel."""
    if (
        voxel_size_m <= 0.0
        or message.is_bigendian
        or message.point_step <= 0
        or message.width * message.height == 0
    ):
        return message

    fields = {field.name: field for field in message.fields}
    xyz = [fields.get(axis) for axis in ("x", "y", "z")]
    if any(
        field is None
        or field.datatype != PointField.FLOAT32
        or field.count != 1
        or field.offset + 4 > message.point_step
        for field in xyz
    ):
        return message

    point_count = int(message.width) * int(message.height)
    packed_row_step = int(message.width) * int(message.point_step)
    source = np.frombuffer(message.data, dtype=np.uint8)

    if message.row_step == packed_row_step:
        records = source[: point_count * message.point_step].reshape(
            point_count, message.point_step
        )
    else:
        records = np.empty((point_count, message.point_step), dtype=np.uint8)
        destination = 0
        for row in range(message.height):
            begin = row * message.row_step
            end = begin + packed_row_step
            records[destination : destination + message.width] = source[begin:end].reshape(
                message.width, message.point_step
            )
            destination += message.width

    coordinates = np.empty((point_count, 3), dtype=np.float32)
    byte_view = records.reshape(-1)
    for column, field in enumerate(xyz):
        coordinates[:, column] = np.ndarray(
            shape=(point_count,),
            dtype="<f4",
            buffer=byte_view,
            offset=field.offset,
            strides=(message.point_step,),
        )

    valid_indices = np.flatnonzero(np.isfinite(coordinates).all(axis=1))
    if valid_indices.size == 0:
        selected = valid_indices
    else:
        voxels = np.floor(coordinates[valid_indices] / voxel_size_m).astype(np.int32)
        keys = np.ascontiguousarray(voxels).view(
            np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])
        ).reshape(-1)
        _, first = np.unique(keys, return_index=True)
        selected = np.sort(valid_indices[first])

    output = copy(message)
    output.height = 1
    output.width = int(selected.size)
    output.row_step = output.width * output.point_step
    output.data = records[selected].tobytes()
    output.is_dense = True
    return output


class MapPointCloudVoxelFilterNode(Node):
    def __init__(self) -> None:
        super().__init__("map_pointcloud_voxel_filter_node")
        self.declare_parameter("voxel_size_m", 0.25)
        self.declare_parameter("local_input_topic", "/lidar_odometry/localmap_points")
        self.declare_parameter("local_output_topic", "/mola/viz/localmap_points")
        self.declare_parameter("global_input_topic", "/lidar_odometry/globalmap_points")
        self.declare_parameter("global_output_topic", "/mola/viz/globalmap_points")
        # Republish with stamp 0 = "use the latest transform". The accumulated
        # map is transient-local and updated rarely (in localization, the loaded
        # cloud is sent once), so its real stamp soon falls out of RViz's ~10 s
        # TF buffer and the map disappears in any view whose Fixed Frame is not
        # the cloud's own frame - base_link, say. A map-fixed view needs no
        # lookup, so this costs nothing there. Only this RViz-only copy is
        # affected; the ICP map, simplemap and saved map keep their stamps.
        self.declare_parameter("publish_with_latest_transform_stamp", True)
        # Repeat the last filtered cloud this often when nothing new arrived.
        # OFF by default, and deliberately so. RViz positions a point cloud once,
        # when the message is processed - unlike markers there is no frame_locked
        # equivalent and no per-frame retransform (PointCloudCommon only
        # retransforms when needs_retransform_ is set). So in a view whose Fixed
        # Frame moves, a static map cloud is only correct at the instant it
        # arrives, and keeping it correct means re-sending hundreds of thousands
        # of points over and over - RViz rebuilds the whole cloud each time,
        # which is exactly the stutter that costs. Enable it (2.0 is a
        # reasonable value) only for a vehicle-frame view, and prefer Fixed
        # Frame `map` with a camera that follows base_link, which needs no
        # retransform at all. After a Fixed Frame change the map can also be
        # brought back by hand: untick and re-tick the display, which
        # resubscribes and redelivers the latched cloud.
        self.declare_parameter("republish_interval_sec", 0.0)

        self._voxel_size_m = float(self.get_parameter("voxel_size_m").value)
        if self._voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")

        local_input = str(self.get_parameter("local_input_topic").value)
        local_output = str(self.get_parameter("local_output_topic").value)
        global_input = str(self.get_parameter("global_input_topic").value)
        global_output = str(self.get_parameter("global_output_topic").value)

        self._local_publisher = self.create_publisher(PointCloud2, local_output, MAP_QOS)
        self._global_publisher = self.create_publisher(PointCloud2, global_output, MAP_QOS)
        self._local_subscription = self.create_subscription(
            PointCloud2,
            local_input,
            lambda message: self._publish_filtered(message, self._local_publisher),
            MAP_QOS,
        )
        self._global_subscription = self.create_subscription(
            PointCloud2,
            global_input,
            lambda message: self._publish_filtered(message, self._global_publisher),
            MAP_QOS,
        )

        self._latest_transform_stamp = bool(
            self.get_parameter("publish_with_latest_transform_stamp").value
        )
        self._republish_interval_ns = int(
            round(float(self.get_parameter("republish_interval_sec").value) * 1e9)
        )
        # publisher -> (last filtered cloud, when it was last sent).
        self._last_sent: dict[object, tuple[PointCloud2, int]] = {}
        if self._republish_interval_ns > 0:
            self.create_timer(
                self._republish_interval_ns * 1e-9 / 2.0, self._repeat_stale_clouds
            )

        self.get_logger().info(
            f"RViz-only map voxel filter: {self._voxel_size_m:.3f} m; "
            f"'{local_input}' -> '{local_output}', '{global_input}' -> '{global_output}'"
        )

    def _publish_filtered(self, message: PointCloud2, publisher) -> None:
        filtered = voxel_filter_cloud(message, self._voxel_size_m)
        if self._latest_transform_stamp:
            filtered.header.stamp = Time()
        self._send(filtered, publisher)

    def _send(self, cloud: PointCloud2, publisher) -> None:
        publisher.publish(cloud)
        self._last_sent[publisher] = (
            cloud,
            self.get_clock().now().nanoseconds,
        )

    def _repeat_stale_clouds(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        for publisher, (cloud, last_publish_ns) in list(self._last_sent.items()):
            if republish_is_due(last_publish_ns, now_ns, self._republish_interval_ns):
                self._send(cloud, publisher)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapPointCloudVoxelFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
