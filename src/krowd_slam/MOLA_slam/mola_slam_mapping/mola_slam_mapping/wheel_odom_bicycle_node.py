import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from ros_can_bridge_msgs.msg import EncoderAngleStamped, WheelSpeedStamped


class WheelOdomBicycleNode(Node):
    """Converts ros_can_bridge's rear wheel speed + front steering angle into
    nav_msgs/Odometry via a bicycle model, for mola_bridge_ros2's wheel_odom input.
    """

    def __init__(self):
        super().__init__("wheel_odom_bicycle_node")

        self.declare_parameter("wheel_speed_topic", "/wheel_sensor")
        self.declare_parameter("steering_angle_topic", "/encoder")
        self.declare_parameter("output_odom_topic", "/wheel_odom")
        self.declare_parameter("wheelbase_m", 1.340)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("max_input_age_sec", 0.5)

        self._wheelbase = float(self.get_parameter("wheelbase_m").value)
        if self._wheelbase <= 0.0:
            raise ValueError("wheelbase_m must be greater than 0")
        self._odom_frame = self.get_parameter("odom_frame_id").value
        self._base_frame = self.get_parameter("base_frame_id").value
        self._max_input_age = float(self.get_parameter("max_input_age_sec").value)

        self._speed_mps = None
        self._speed_stamp = None
        self._steer_rad = None
        self._steer_stamp = None

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_pub_time = None

        self.create_subscription(
            WheelSpeedStamped,
            self.get_parameter("wheel_speed_topic").value,
            self._on_speed,
            10,
        )
        self.create_subscription(
            EncoderAngleStamped,
            self.get_parameter("steering_angle_topic").value,
            self._on_steering,
            10,
        )
        self._odom_pub = self.create_publisher(
            Odometry, self.get_parameter("output_odom_topic").value, 10
        )

        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        if rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be greater than 0")
        self.create_timer(1.0 / rate_hz, self._on_timer)

        self.get_logger().info(
            f"wheel_odom_bicycle_node: {self.get_parameter('wheel_speed_topic').value} + "
            f"{self.get_parameter('steering_angle_topic').value} -> "
            f"{self.get_parameter('output_odom_topic').value} "
            f"(wheelbase={self._wheelbase}m)"
        )

    def _on_speed(self, msg: WheelSpeedStamped) -> None:
        self._speed_mps = msg.speed_mps
        # header.stamp (set by can_to_ros_node from the actual SocketCAN receive time),
        # not this callback's processing time - freshness should reflect how old the CAN
        # reading itself is, not added ROS scheduling/queuing latency.
        self._speed_stamp = Time.from_msg(msg.header.stamp)

    def _on_steering(self, msg: EncoderAngleStamped) -> None:
        self._steer_rad = msg.angle_rad
        self._steer_stamp = Time.from_msg(msg.header.stamp)

    def _inputs_fresh(self, now) -> bool:
        if self._speed_stamp is None or self._steer_stamp is None:
            return False
        speed_age = (now - self._speed_stamp).nanoseconds * 1e-9
        steer_age = (now - self._steer_stamp).nanoseconds * 1e-9
        return speed_age <= self._max_input_age and steer_age <= self._max_input_age

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        if not self._inputs_fresh(now):
            # No fresh CAN data yet (or the bridge went quiet) - don't dead-reckon on
            # stale/missing readings, and don't advance the integrator's clock either
            # (next fresh sample would otherwise see a large, bogus dt).
            self._last_pub_time = None
            return

        if self._last_pub_time is None:
            self._last_pub_time = now
            return
        dt = (now - self._last_pub_time).nanoseconds * 1e-9
        self._last_pub_time = now
        if dt <= 0.0:
            return

        v = self._speed_mps
        delta = self._steer_rad
        yaw_rate = v * math.tan(delta) / self._wheelbase

        # Forward-Euler integration of the bicycle model. Fine for odometry-quality
        # dead reckoning at ~50Hz; not meant for large dt or high-speed cornering.
        self._x += v * math.cos(self._yaw) * dt
        self._y += v * math.sin(self._yaw) * dt
        self._yaw += yaw_rate * dt

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        half_yaw = self._yaw * 0.5
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)

        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = yaw_rate

        # Rough placeholder sigmas (diagonal only) - not derived from measured sensor
        # noise. Nonzero on purpose: an all-zero covariance reads as "perfectly
        # confident" to consumers (mola_slam_mapping's own waypoint_saver_node included),
        # which is wrong for open-loop dead-reckoned odometry. Tune once real wheel
        # speed/encoder noise is characterized.
        #
        # z/roll/pitch use 100.0 (sigma ~10 m / ~573 deg), not e.g. 1e6: still reads as
        # "essentially unconstrained" to any consumer, but avoids stacking a ~1e6-scale
        # covariance on top of mola_state_estimation_smoother's own extreme-but-legitimate
        # sigmas elsewhere in the same factor graph (first-pose-anchor prior at 1e-6, new
        # odometry-frame prior at 1e3) - that combination spans ~12 orders of magnitude,
        # which pushed GTSAM's Cholesky factorization into IndeterminantLinearSystemException
        # ("underconstrained") on the very first wheel_odom reading, before GNSS/georef had
        # even started. Reproduced and confirmed while replaying an old bag through
        # wheel_odom_from_bag_node.py, which used the same 1e6 placeholder.
        msg.pose.covariance[0] = 0.05**2  # x
        msg.pose.covariance[7] = 0.05**2  # y
        msg.pose.covariance[14] = 100.0  # z (unmeasured, planar vehicle)
        msg.pose.covariance[21] = 100.0  # roll (unmeasured)
        msg.pose.covariance[28] = 100.0  # pitch (unmeasured)
        msg.pose.covariance[35] = 0.02**2  # yaw

        msg.twist.covariance[0] = 0.05**2  # vx
        msg.twist.covariance[7] = 0.05**2  # vy (bicycle model assumes 0, not measured)
        msg.twist.covariance[14] = 100.0  # vz
        msg.twist.covariance[21] = 100.0  # wx
        msg.twist.covariance[28] = 100.0  # wy
        msg.twist.covariance[35] = 0.02**2  # wz (yaw rate)

        self._odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdomBicycleNode()
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
