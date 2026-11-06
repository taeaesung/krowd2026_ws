"""Record the run's stamped pose streams to CSV for offline map re-trajectory.

The saved route (waypoints.csv) is 1 m samples in UTM without timestamps, and
the simplemap keyframes carry stamps but LiDAR-odometry poses. Bringing the
point-cloud map onto the RTK trajectory afterwards (scripts/retraject_map.sh ->
map_retraject) needs the reference pose *with its stamps*, which nothing else
keeps. This node does only that: one line per message, flushed immediately, so
an interrupted run still leaves everything received so far. Subscribe-only.

Files, in the run directory shared with mola-cli and waypoint_saver_node:
    trajectory_reference.csv        /gnss_imu/pose          map <- base_link
    trajectory_fused.csv            /fused_localization/pose
    trajectory_lidar_odometry.csv   /lidar_odometry/pose    odom <- base_link
Columns: stamp_ns, frame, x, y, z, qx, qy, qz, qw (see trajectory_io.py).
"""

import csv
import time as wall_time
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from mola_slam_mapping.trajectory_io import TRAJECTORY_COLUMNS, trajectory_row


class _CsvStream:
    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TRAJECTORY_COLUMNS)
        self._file.flush()
        self.rows = 0

    def append(self, msg: Odometry) -> None:
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self._writer.writerow(
            trajectory_row(stamp_ns, msg.header.frame_id, msg.pose.pose)
        )
        self._file.flush()
        self.rows += 1

    def close(self) -> None:
        self._file.close()


class TrajectoryRecorderNode(Node):
    def __init__(self):
        super().__init__("trajectory_recorder_node")

        self.declare_parameter("output_dir", "~/krowd_maps")
        # Shared with waypoint_saver_node by mapping.launch.py so every artifact
        # of one run lands in one folder.
        self.declare_parameter("run_id", "")
        self.declare_parameter("reference_pose_topic", "/gnss_imu/pose")
        self.declare_parameter("fused_pose_topic", "/fused_localization/pose")
        self.declare_parameter("lidar_pose_topic", "/lidar_odometry/pose")

        run_id = self.get_parameter("run_id").value or wall_time.strftime(
            "%Y%m%d_%H%M%S"
        )
        run_dir = Path(self.get_parameter("output_dir").value).expanduser() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._streams = {}
        for parameter, filename in (
            ("reference_pose_topic", "trajectory_reference.csv"),
            ("fused_pose_topic", "trajectory_fused.csv"),
            ("lidar_pose_topic", "trajectory_lidar_odometry.csv"),
        ):
            topic = self.get_parameter(parameter).value
            stream = _CsvStream(run_dir / filename)
            self._streams[topic] = stream
            self.create_subscription(
                Odometry, topic, lambda msg, s=stream: s.append(msg), 100
            )
        self.get_logger().info(
            "trajectory_recorder_node: recording "
            + ", ".join(f"'{topic}' -> {s.path.name}" for topic, s in self._streams.items())
            + f" in '{run_dir}'"
        )

    def close(self) -> None:
        for topic, stream in self._streams.items():
            stream.close()
            self.get_logger().info(f"{stream.path.name}: {stream.rows} poses from '{topic}'")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
