"""Select a trustworthy map-frame pose for MOLA's one-shot seed.

The stationary LiDAR map initializer publishes candidate poses while it tests
the 360-degree yaw hypotheses. Those candidates must not seed MOLA until the
initializer has declared the heading verified.
Until then, the existing GNSS/course pose remains a fallback for deployments
that drive far enough to establish a course heading.
"""

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool


class VerifiedPoseSelector:
    """Latch verification and return only poses safe to forward."""

    def __init__(self) -> None:
        self.verified = False
        self.latest_primary: Odometry | None = None

    def primary(self, message: Odometry) -> Odometry | None:
        self.latest_primary = message
        return message if self.verified else None

    def verification(self, verified: bool) -> Odometry | None:
        if self.verified or not verified:
            return None
        self.verified = True
        return self.latest_primary

    def fallback(self, message: Odometry) -> Odometry | None:
        return None if self.verified else message


class LocalizationReferencePoseNode(Node):
    def __init__(self) -> None:
        super().__init__("localization_reference_pose_node")

        self.declare_parameter(
            "primary_pose_topic", "/mola/map_localization/odometry"
        )
        self.declare_parameter(
            "primary_verified_topic", "/mola/map_localization/heading_verified"
        )
        self.declare_parameter("fallback_pose_topic", "/gnss_imu/pose")
        self.declare_parameter("output_pose_topic", "/localization/mola_seed_pose")
        self.declare_parameter("expected_frame", "map")

        self._selector = VerifiedPoseSelector()
        self._expected_frame = str(self.get_parameter("expected_frame").value)
        self._publisher = self.create_publisher(
            Odometry, self.get_parameter("output_pose_topic").value, 20
        )
        self.create_subscription(
            Odometry,
            self.get_parameter("primary_pose_topic").value,
            self._on_primary,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            self.get_parameter("primary_verified_topic").value,
            self._on_verification,
            10,
        )
        fallback_topic = str(self.get_parameter("fallback_pose_topic").value)
        if fallback_topic:
            self.create_subscription(
                Odometry,
                fallback_topic,
                self._on_fallback,
                qos_profile_sensor_data,
            )

        self.get_logger().info(
            "Waiting for verified stationary LiDAR map alignment on "
            f"'{self.get_parameter('primary_pose_topic').value}'; "
            f"falling back to '{fallback_topic}' if its drive-based heading "
            "becomes available first."
        )

    def _publish(self, message: Odometry | None) -> None:
        if message is None:
            return
        frame = message.header.frame_id.strip()
        if frame != self._expected_frame:
            self.get_logger().warn(
                f"Rejected localization pose in frame '{frame}'; expected "
                f"'{self._expected_frame}'.",
                throttle_duration_sec=5.0,
            )
            return
        self._publisher.publish(message)

    def _on_primary(self, message: Odometry) -> None:
        self._publish(self._selector.primary(message))

    def _on_verification(self, message: Bool) -> None:
        was_verified = self._selector.verified
        selected = self._selector.verification(message.data)
        if not was_verified and self._selector.verified:
            self.get_logger().info(
                "Stationary LiDAR map alignment verified; using it for MOLA "
                "relocalization."
            )
        self._publish(selected)

    def _on_fallback(self, message: Odometry) -> None:
        self._publish(self._selector.fallback(message))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizationReferencePoseNode()
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
