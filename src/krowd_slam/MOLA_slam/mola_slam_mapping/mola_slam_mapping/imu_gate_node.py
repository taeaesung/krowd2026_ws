import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import NavSatFix


class ImuGateNode(Node):
    """TEST ONLY. Republishes IMU messages unchanged, but only after the first GNSS
    fix has been seen - see wheel_odom_from_bag_node.py's docstring for why: in
    URBAN_C0, /ublox_gps/fix's header.stamp runs ~37s ahead of every other topic's (an
    unsynced GPS receiver clock baked into the recording, not a slow fix), and
    mola_state_estimation_smoother's sliding window marginalizes the early IMU-only
    keyframes long before GNSS's skewed timestamp catches up, corrupting GTSAM's
    internal state once GNSS fusion tries to re-link back to them. NOTE: this alone did
    NOT fully fix the crash in testing - LidarOdometry's own initial-relocalization
    creates an early anchor keyframe from the ungated LiDAR topic regardless. Not used
    on the real vehicle (only wired in for wheel_odom_source:=bag) - all sensors are
    expected to share one clock and come online together there, so this defect
    shouldn't occur in the first place.
    """

    def __init__(self):
        super().__init__("imu_gate_node")

        self.declare_parameter("input_imu_topic", "/imu/data")
        self.declare_parameter("output_imu_topic", "/imu/data_gated")
        self.declare_parameter("gnss_topic", "/ublox_gps/fix")

        self._gnss_seen = False

        self._imu_pub = self.create_publisher(
            Imu, self.get_parameter("output_imu_topic").value, 10
        )
        self.create_subscription(
            Imu, self.get_parameter("input_imu_topic").value, self._on_imu, 10
        )
        gnss_topic = self.get_parameter("gnss_topic").value
        self.create_subscription(NavSatFix, gnss_topic, self._on_gnss, 10)

        self.get_logger().warn(
            "imu_gate_node is TEST ONLY: withholding "
            f"'{self.get_parameter('input_imu_topic').value}' -> "
            f"'{self.get_parameter('output_imu_topic').value}' until first fix on "
            f"'{gnss_topic}'"
        )

    def _on_gnss(self, msg: NavSatFix) -> None:
        if not self._gnss_seen:
            self._gnss_seen = True
            self.get_logger().info("First GNSS fix seen - IMU output enabled")

    def _on_imu(self, msg: Imu) -> None:
        if self._gnss_seen:
            self._imu_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuGateNode()
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
