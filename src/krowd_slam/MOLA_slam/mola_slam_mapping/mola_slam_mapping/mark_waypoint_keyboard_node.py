import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


def _read_key(original_settings) -> str:
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
    return key


class MarkWaypointKeyboardNode(Node):
    """Interactive terminal tool: SPACE calls waypoint_saver_node's
    ~/mark_waypoint service to force-record a waypoint right now, on top of
    (not instead of) its normal automatic 1m-spacing sampling - e.g. to pin a
    cone, the start/finish line, or any point of interest. 'q' or Ctrl+C quits.

    Reads raw stdin (same technique as teleop_twist_keyboard), so this only
    works run interactively in its own terminal with keyboard focus - it is
    not wired into mapping.launch.py.
    """

    def __init__(self):
        super().__init__("mark_waypoint_keyboard_node")
        self.declare_parameter("service_name", "/waypoint_saver_node/mark_waypoint")
        service_name = self.get_parameter("service_name").value
        self._client = self.create_client(Trigger, service_name)
        self.get_logger().info(f"Waiting for service '{service_name}'...")
        self._client.wait_for_service()
        self.get_logger().info("Ready - press SPACE to mark a waypoint, 'q' to quit.")

    def mark(self) -> None:
        future = self._client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if future.done() and future.result() is not None:
            result = future.result()
            log = self.get_logger().info if result.success else self.get_logger().warn
            log(result.message)
        else:
            self.get_logger().warn("mark_waypoint service call timed out")


def main(args=None):
    rclpy.init(args=args)
    node = MarkWaypointKeyboardNode()
    original_settings = termios.tcgetattr(sys.stdin)
    try:
        while rclpy.ok():
            key = _read_key(original_settings)
            if key == " ":
                node.mark()
            elif key in ("q", "\x03"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
