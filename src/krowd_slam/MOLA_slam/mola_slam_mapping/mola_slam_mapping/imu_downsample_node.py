"""Timestamp-based IMU downsampler for the state-estimation smoother."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


class TimestampRateGate:
    """Select at most one sample per period using sensor timestamps."""

    def __init__(self, target_rate_hz: float):
        if not math.isfinite(target_rate_hz) or target_rate_hz <= 0.0:
            raise ValueError("target_rate_hz must be finite and greater than zero")
        self.period_ns = max(1, int(round(1.0e9 / target_rate_hz)))
        self.last_emitted_ns = None

    def accept(self, stamp_ns: int) -> bool:
        stamp_ns = int(stamp_ns)
        if (
            self.last_emitted_ns is None
            or stamp_ns < self.last_emitted_ns
            or stamp_ns - self.last_emitted_ns >= self.period_ns
        ):
            self.last_emitted_ns = stamp_ns
            return True
        return False


def stamp_nanoseconds(msg: Imu) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(
        msg.header.stamp.nanosec
    )


class ImuDownsampleNode(Node):
    def __init__(self):
        super().__init__("imu_downsample_node")
        self.declare_parameter("input_topic", "/imu/data_ekf")
        self.declare_parameter("output_topic", "/mola/imu/data_ekf_10hz")
        self.declare_parameter("target_rate_hz", 10.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        target_rate_hz = float(self.get_parameter("target_rate_hz").value)
        if not input_topic or not output_topic:
            raise ValueError("input_topic and output_topic must not be empty")
        if input_topic == output_topic:
            raise ValueError("input_topic and output_topic must be different")

        self._gate = TimestampRateGate(target_rate_hz)
        self._publisher = self.create_publisher(
            Imu, output_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            Imu, input_topic, self._on_imu, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Downsampling IMU '{input_topic}' -> '{output_topic}' at "
            f"{target_rate_hz:g} Hz using header timestamps"
        )

    def _on_imu(self, msg: Imu) -> None:
        if self._gate.accept(stamp_nanoseconds(msg)):
            self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuDownsampleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # A launch-wide SIGINT may invalidate the ROS context between taking a
        # queued sensor callback and publishing it. Suppress only that shutdown
        # race; a RuntimeError while ROS is live remains a real failure.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
