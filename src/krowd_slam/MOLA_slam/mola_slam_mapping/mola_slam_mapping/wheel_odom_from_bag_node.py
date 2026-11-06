import math

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import NavSatFix


class WheelOdomFromBagNode(Node):
    """TEST ONLY. Converts a recorded bag's independent left/right wheel
    geometry_msgs/TwistStamped topics into nav_msgs/Odometry via differential-drive
    kinematics, so the mapping pipeline can be validated before the real vehicle's
    CAN wheel-speed + steering-encoder sensors are wired up. Not used on the actual
    vehicle - see wheel_odom_bicycle_node.py for the production CAN-based version.

    Only linear.x from /wheel_left and /wheel_right is used. Their angular.z is NOT
    vehicle yaw rate - in this bag it's a fixed multiple of linear.x (each wheel's own
    spin rate), so it carries no independent turning information and must not be fed
    into yaw_rate.

    Event-driven on the wheel messages themselves (not a wall-clock timer): dt and the
    published header.stamp both come from the incoming messages' own header.stamp, not
    self.get_clock().now(). This matters when replaying an old bag - ros2 bag play
    republishes every OTHER topic with its original recorded stamp, so a wall-clock
    stamp here would desync this topic from the rest by however old the bag is (the
    downstream state estimator would see a multi-year time jump between this topic and
    lidar/imu/gnss, and blow up its constant-velocity kinematics factor). Using each
    message's own stamp keeps this node correct in both live (real CAN, current time)
    and bag-replay use.

    Also withholds publishing until the first GNSS fix is seen (see gnss_topic param).
    In URBAN_C0, /ublox_gps/fix's header.stamp runs ~37s ahead of every other topic's -
    not a slow GNSS fix (the whole bag is only ~18.4s long), but an unsynced GPS receiver
    clock baked into the recording. mola_state_estimation_smoother's sliding window (a
    couple seconds) marginalizes the early wheel/IMU-only keyframes long before GNSS's
    (skewed) timestamp catches up, and fusing GNSS then tries to re-link back to an
    already-discarded keyframe, corrupting GTSAM's internal state
    (IndeterminantLinearSystemException / SIGSEGV). NOTE: withholding wheel/IMU alone did
    NOT fully fix this in testing - LidarOdometry's own initial-relocalization creates an
    early anchor keyframe from the ungated LiDAR topic regardless, so the same
    marginalization problem can still recur through that path. Not needed for the
    production CAN path: on the real vehicle all sensors are expected to share one clock
    and come online together, so this timestamp-skew defect shouldn't occur there in the
    first place.
    """

    def __init__(self):
        super().__init__("wheel_odom_from_bag_node")

        self.declare_parameter("left_wheel_topic", "/wheel_left")
        self.declare_parameter("right_wheel_topic", "/wheel_right")
        self.declare_parameter("output_odom_topic", "/wheel_odom")
        self.declare_parameter("track_width_m", 1.080)
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("max_side_skew_sec", 0.1)
        self.declare_parameter("gnss_topic", "/ublox_gps/fix")

        self._track_width = float(self.get_parameter("track_width_m").value)
        if self._track_width <= 0.0:
            raise ValueError("track_width_m must be greater than 0")
        self._odom_frame = self.get_parameter("odom_frame_id").value
        self._base_frame = self.get_parameter("base_frame_id").value
        # Max allowed gap between the left and right wheel's own timestamps before a
        # pair is considered stale/mismatched (they arrive on independent topics, not
        # already synchronized).
        self._max_side_skew = float(self.get_parameter("max_side_skew_sec").value)

        self._left_v = None
        self._left_stamp = None
        self._right_v = None
        self._right_stamp = None

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._last_stamp = None
        self._gnss_seen = False

        self.create_subscription(
            TwistStamped,
            self.get_parameter("left_wheel_topic").value,
            self._on_left,
            10,
        )
        self.create_subscription(
            TwistStamped,
            self.get_parameter("right_wheel_topic").value,
            self._on_right,
            10,
        )
        gnss_topic = self.get_parameter("gnss_topic").value
        self.create_subscription(NavSatFix, gnss_topic, self._on_gnss, 10)
        self._odom_pub = self.create_publisher(
            Odometry, self.get_parameter("output_odom_topic").value, 10
        )

        self.get_logger().warn(
            "wheel_odom_from_bag_node is TEST ONLY (differential-drive from a "
            "recorded bag's left/right wheel topics), not the production CAN path: "
            f"{self.get_parameter('left_wheel_topic').value} + "
            f"{self.get_parameter('right_wheel_topic').value} -> "
            f"{self.get_parameter('output_odom_topic').value} "
            f"(track_width={self._track_width}m); withholding output until first "
            f"fix on '{gnss_topic}'"
        )

    def _on_gnss(self, msg: NavSatFix) -> None:
        if not self._gnss_seen:
            self._gnss_seen = True
            self.get_logger().info("First GNSS fix seen - wheel_odom output enabled")

    def _on_left(self, msg: TwistStamped) -> None:
        self._left_v = msg.twist.linear.x
        self._left_stamp = Time.from_msg(msg.header.stamp)
        self._try_integrate(self._left_stamp)

    def _on_right(self, msg: TwistStamped) -> None:
        self._right_v = msg.twist.linear.x
        self._right_stamp = Time.from_msg(msg.header.stamp)
        self._try_integrate(self._right_stamp)

    def _try_integrate(self, stamp: Time) -> None:
        if not self._gnss_seen:
            return
        if self._left_stamp is None or self._right_stamp is None:
            return
        skew = abs((self._left_stamp - self._right_stamp).nanoseconds) * 1e-9
        if skew > self._max_side_skew:
            return

        if self._last_stamp is None:
            self._last_stamp = stamp
            return
        dt = (stamp - self._last_stamp).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self._last_stamp = stamp

        # Differential-drive kinematic identity: outer/inner wheel speed difference
        # during a turn relates to yaw rate via track width. This holds geometrically
        # for any vehicle regardless of steering mechanism (Ackermann included), so
        # it's a valid way to recover yaw rate from independent left/right wheel
        # speeds even though the real vehicle steers via a single front axle rather
        # than by driving the wheels at different speeds.
        v = (self._left_v + self._right_v) * 0.5
        yaw_rate = (self._right_v - self._left_v) / self._track_width

        # Forward-Euler integration, same approach as wheel_odom_bicycle_node.py.
        self._x += v * math.cos(self._yaw) * dt
        self._y += v * math.sin(self._yaw) * dt
        self._yaw += yaw_rate * dt

        msg = Odometry()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id = self._base_frame

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        half_yaw = self._yaw * 0.5
        msg.pose.pose.orientation.z = math.sin(half_yaw)
        msg.pose.pose.orientation.w = math.cos(half_yaw)

        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = yaw_rate

        # Rough placeholder sigmas, not measured - same rationale as
        # wheel_odom_bicycle_node.py: nonzero on purpose so this doesn't read as
        # "perfectly confident" to downstream consumers.
        #
        # z/roll/pitch use 100.0 (sigma ~10 m / ~573 deg), not e.g. 1e6: still reads as
        # "essentially unconstrained" to any consumer, but avoids stacking a ~1e6-scale
        # covariance on top of mola_state_estimation_smoother's own extreme-but-legitimate
        # sigmas elsewhere in the same factor graph (first-pose-anchor prior at 1e-6, new
        # odometry-frame prior at 1e3) - that combination spans ~12 orders of magnitude,
        # which pushed GTSAM's Cholesky factorization into IndeterminantLinearSystemException
        # ("underconstrained") on the very first wheel_odom reading, before GNSS/georef had
        # even started (reproduced replaying an old bag through this node).
        msg.pose.covariance[0] = 0.05**2  # x
        msg.pose.covariance[7] = 0.05**2  # y
        msg.pose.covariance[14] = 100.0  # z (unmeasured, planar vehicle)
        msg.pose.covariance[21] = 100.0  # roll (unmeasured)
        msg.pose.covariance[28] = 100.0  # pitch (unmeasured)
        msg.pose.covariance[35] = 0.02**2  # yaw

        msg.twist.covariance[0] = 0.05**2  # vx
        msg.twist.covariance[7] = 0.05**2  # vy (assumed 0, not measured)
        msg.twist.covariance[14] = 100.0  # vz
        msg.twist.covariance[21] = 100.0  # wx
        msg.twist.covariance[28] = 100.0  # wy
        msg.twist.covariance[35] = 0.02**2  # wz (yaw rate)

        self._odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdomFromBagNode()
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
