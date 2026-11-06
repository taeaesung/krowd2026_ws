"""Announce which mapping run the localization stack is using.

Path tracking needs the route of the SAME run localization loaded its map from,
and typing that directory into both launches is exactly how the two end up on
different maps. Localization already knows it, so it says so here: one latched
std_msgs/String with the resolved run directory. route_publisher_node subscribes
to it when it was not given a map_dir of its own.

Latched (transient local, depth 1) rather than periodic: path tracking is
normally started after localization, and a late subscriber still gets the value.
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


def resolved_map_dir(map_dir) -> str:
    """The run directory as an absolute path; raises when it is unusable.

    Resolved because map_dir is often the `latest` symlink: consumers should be
    told the run that was actually loaded, not a name that moves under them.
    """

    text = str(map_dir).strip()
    if not text:
        raise ValueError("map_dir is empty")
    path = Path(text).expanduser()
    if not path.is_dir():
        raise ValueError(f"map_dir is not a directory: {path}")
    return str(path.resolve())


class SelectedMapNode(Node):
    """Publish the selected mapping run directory, once, latched."""

    def __init__(self):
        super().__init__("selected_map_node")

        self.declare_parameter("map_dir", "")
        self.declare_parameter("map_dir_topic", "/localization/map_dir")

        try:
            map_dir = resolved_map_dir(self.get_parameter("map_dir").value)
        except ValueError as error:
            raise RuntimeError(f"selected_map_node: {error}") from error

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        topic = str(self.get_parameter("map_dir_topic").value)
        self._publisher = self.create_publisher(String, topic, latched)
        self._publisher.publish(String(data=map_dir))
        self.get_logger().info(
            f"selected_map_node: localizing against '{map_dir}'; announced on "
            f"'{topic}' so path tracking follows the same run"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SelectedMapNode()
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
