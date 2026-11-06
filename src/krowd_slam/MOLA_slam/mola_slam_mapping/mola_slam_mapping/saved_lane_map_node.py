"""Publish the lane boundaries and stop lines saved with the selected map.

A map run may carry <map_dir>/lane_map.csv: lanes and stop lines already in
this map's frame (krowd_control/tools lane map pipeline, one row per point:
segment_id, point_index, stamp_ns, lane_class, namespace, quality, map_x,
map_y, map_z). It lives next to waypoints.csv rather than inside the .mm,
because the .mm is what MOLA scan-matches against and lane points are not
LiDAR geometry - and because every controller can read a CSV.

Output, latched (transient local, depth 1) so a late RViz or controller gets it:
    /lane_map/saved   MarkerArray, map   one LINE_STRIP per CSV segment:
                                         ns = CSV namespace, id = segment_id,
                                         points in point_index order

Stamped 0 so RViz draws it with the latest TF under either wall or sim time.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

REQUIRED_COLUMNS = (
    "segment_id", "point_index", "lane_class", "namespace", "map_x", "map_y", "map_z",
)
# Same colours as krowd_control/tools/lane_map_accumulator.py.
CLASS_COLORS = {
    "leftSolid": (1.0, 0.85, 0.05),
    "leftDashed": (1.0, 0.35, 0.05),
    "rightSolid": (0.05, 0.85, 1.0),
    "rightDashed": (0.25, 0.40, 1.0),
    "stopLine": (1.0, 0.05, 0.05),
}
UNKNOWN_COLOR = (0.8, 0.8, 0.8)


@dataclass(frozen=True)
class SavedSegment:
    segment_id: int
    lane_class: str
    namespace: str
    points: np.ndarray  # (N, 3) map x, y, z in point_index order


def load_lane_map(path) -> list[SavedSegment]:
    """Segments of a lane_map.csv, ordered by segment_id; raises when unusable."""

    path = Path(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
        grouped: dict[int, dict] = {}
        for line, row in enumerate(reader, start=2):
            try:
                segment_id = int(row["segment_id"])
                point_index = int(row["point_index"])
                xyz = (float(row["map_x"]), float(row["map_y"]), float(row["map_z"]))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            segment = grouped.setdefault(
                segment_id,
                {"lane_class": row["lane_class"], "namespace": row["namespace"], "rows": []},
            )
            if (row["lane_class"], row["namespace"]) != (
                segment["lane_class"], segment["namespace"]
            ):
                raise ValueError(
                    f"{path}:{line}: segment {segment_id} mixes classes/namespaces"
                )
            segment["rows"].append((point_index, xyz))

    segments = []
    for segment_id in sorted(grouped):
        segment = grouped[segment_id]
        rows = sorted(segment["rows"])
        indices = [index for index, _ in rows]
        if len(set(indices)) != len(indices):
            raise ValueError(f"{path}: segment {segment_id} repeats a point_index")
        if len(rows) < 2:
            raise ValueError(f"{path}: segment {segment_id} has fewer than 2 points")
        segments.append(
            SavedSegment(
                segment_id,
                segment["lane_class"],
                segment["namespace"],
                np.array([xyz for _, xyz in rows], dtype=float),
            )
        )
    if not segments:
        raise ValueError(f"{path}: no lane segments")
    return segments


def lane_map_markers(
    segments, frame_id: str, lane_width_m: float, stopline_width_m: float
) -> MarkerArray:
    clear = Marker()
    clear.header.frame_id = frame_id
    clear.action = Marker.DELETEALL
    markers = [clear]
    for segment in segments:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.ns = segment.namespace
        marker.id = segment.segment_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = stopline_width_m if segment.lane_class == "stopLine" else lane_width_m
        r, g, b = CLASS_COLORS.get(segment.lane_class, UNKNOWN_COLOR)
        marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        marker.points = [Point(x=x, y=y, z=z) for x, y, z in segment.points.tolist()]
        markers.append(marker)
    return MarkerArray(markers=markers)


class SavedLaneMapNode(Node):
    """Load lane_map.csv once and publish it latched."""

    def __init__(self):
        super().__init__("saved_lane_map_node")

        self.declare_parameter("lane_map_csv", "")
        self.declare_parameter("topic", "/lane_map/saved")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("lane_width_m", 0.18)
        self.declare_parameter("stopline_width_m", 0.35)
        g = lambda name: self.get_parameter(name).value  # noqa: E731

        csv_path = Path(str(g("lane_map_csv"))).expanduser()
        try:
            segments = load_lane_map(csv_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"saved_lane_map_node: {error}") from error

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        topic = str(g("topic"))
        self._publisher = self.create_publisher(MarkerArray, topic, latched)
        self._publisher.publish(
            lane_map_markers(
                segments, str(g("frame_id")),
                float(g("lane_width_m")), float(g("stopline_width_m")),
            )
        )
        counts: dict[str, int] = {}
        for segment in segments:
            counts[segment.lane_class] = counts.get(segment.lane_class, 0) + 1
        points = sum(len(segment.points) for segment in segments)
        self.get_logger().info(
            f"saved_lane_map_node: '{csv_path}' -> '{topic}' (latched, "
            f"'{g('frame_id')}'): {len(segments)} segments {counts}, {points} points"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SavedLaneMapNode()
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
