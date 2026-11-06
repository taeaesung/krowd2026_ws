"""Relay NavSatFix while enforcing realistic covariance lower bounds."""

from copy import deepcopy
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix


def _floored_variance(value: float, minimum_sigma: float) -> float:
    minimum_variance = minimum_sigma * minimum_sigma
    if not math.isfinite(value) or value < minimum_variance:
        return minimum_variance
    return value


def floor_navsat_covariance(
    msg: NavSatFix, min_xy_sigma_m: float, min_z_sigma_m: float
) -> NavSatFix:
    """Return a copy with finite diagonal covariance at or above the floors."""

    for name, value in (
        ("min_xy_sigma_m", min_xy_sigma_m),
        ("min_z_sigma_m", min_z_sigma_m),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")

    result = deepcopy(msg)
    covariance = list(result.position_covariance)
    if msg.position_covariance_type == NavSatFix.COVARIANCE_TYPE_UNKNOWN:
        covariance = [0.0] * 9
        result.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
    else:
        covariance = [value if math.isfinite(value) else 0.0 for value in covariance]

    covariance[0] = _floored_variance(covariance[0], min_xy_sigma_m)
    covariance[4] = _floored_variance(covariance[4], min_xy_sigma_m)
    covariance[8] = _floored_variance(covariance[8], min_z_sigma_m)
    result.position_covariance = covariance
    return result


class GnssCovarianceFloorNode(Node):
    def __init__(self):
        super().__init__("gnss_covariance_floor_node")
        self.declare_parameter("input_topic", "/ublox_gps_node/fix")
        self.declare_parameter("output_topic", "/mola/gnss/fix_sanitized")
        self.declare_parameter("min_xy_sigma_m", 0.05)
        self.declare_parameter("min_z_sigma_m", 1.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self._min_xy_sigma_m = float(
            self.get_parameter("min_xy_sigma_m").value
        )
        self._min_z_sigma_m = float(self.get_parameter("min_z_sigma_m").value)
        if not input_topic or not output_topic:
            raise ValueError("input_topic and output_topic must not be empty")
        if input_topic == output_topic:
            raise ValueError("input_topic and output_topic must be different")
        # Validate at startup, before any sensor data arrives.
        floor_navsat_covariance(
            NavSatFix(), self._min_xy_sigma_m, self._min_z_sigma_m
        )

        self._publisher = self.create_publisher(
            NavSatFix, output_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            NavSatFix, input_topic, self._on_fix, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Relaying GNSS '{input_topic}' -> '{output_topic}' with covariance "
            f"sigma floors XY={self._min_xy_sigma_m:g} m, "
            f"Z={self._min_z_sigma_m:g} m"
        )

    def _on_fix(self, msg: NavSatFix) -> None:
        self._publisher.publish(
            floor_navsat_covariance(
                msg, self._min_xy_sigma_m, self._min_z_sigma_m
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = GnssCovarianceFloorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # See imu_downsample_node: launch can stop the context while a queued
        # relay callback is publishing. Do not turn that normal shutdown race
        # into a node error.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
