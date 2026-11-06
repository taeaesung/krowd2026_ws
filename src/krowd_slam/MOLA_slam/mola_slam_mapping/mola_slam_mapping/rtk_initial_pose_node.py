"""Seed MOLA's relocalization with the first RTK pose of a localization run.

localization.launch.py starts LidarOdometry with InitLocalization::FixedPose at
the saved map origin and covariance 1e-12 - "certain" - which is right only
when the run starts exactly where mapping started. MOLA's own global layer (the
smoother's GNSS factors) is switched off in this package because it measurably
damaged ICP, and with it went the only caller of MOLA's relocalization. This
node fills that one gap: the first heading-committed RTK pose (/gnss_imu/pose,
already map->base_link through the saved georeference) is published once on
/initialpose, the topic BridgeROS2 forwards to
LidarOdometry::relocalize_near_pose_pdf(). ICP then confirms and refines around
the RTK prior - RTK decides where, ICP decides exactly where.

The covariance is not cosmetic. MOLA sizes its ICP correspondence search from
it: adapt_thres_sigma = sqrt(sigma_x^2 + sigma_y^2). It therefore has to cover
the real error of the prior, not RTK's own 1.5 cm: the pose that reaches us is
up to one RTK period old (~0.8 m at 4 m/s), and it carries the map's
georeference and the heading alignment. 1.0 m per axis gives a 1.41 m search,
inside the pipeline's maximum_sigma of 2.0 m, and the adaptive threshold
shrinks it again as ICP converges.

gnss_imu_pose_node holds its poses back until the heading is committed (~5 m
of straight driving) and then releases them all at once, oldest first. The
first message is therefore ~5 m stale; the node waits a short settle time and
seeds from the newest one instead.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger


def initial_pose_from_odometry(
    odometry: Odometry,
    position_sigma_m: float,
    z_sigma_m: float,
    yaw_sigma_deg: float,
    roll_pitch_sigma_deg: float,
) -> PoseWithCovarianceStamped:
    """The /initialpose message for an RTK pose, with the prior's real uncertainty.

    The covariance is the ROS 6x6 row-major (x, y, z, roll, pitch, yaw); the
    bridge reorders it for MRPT. Only the diagonal is set.
    """

    for name, value in (
        ("position_sigma_m", position_sigma_m),
        ("z_sigma_m", z_sigma_m),
        ("yaw_sigma_deg", yaw_sigma_deg),
        ("roll_pitch_sigma_deg", roll_pitch_sigma_deg),
    ):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive, got {value}")
    frame = odometry.header.frame_id.strip()
    if not frame:
        raise ValueError("RTK pose has no frame_id; cannot seed relocalization")
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    if not all(
        math.isfinite(v)
        for v in (
            position.x, position.y, position.z,
            orientation.x, orientation.y, orientation.z, orientation.w,
        )
    ):
        raise ValueError("RTK pose contains a non-finite value")

    seed = PoseWithCovarianceStamped()
    seed.header.stamp = odometry.header.stamp
    seed.header.frame_id = frame
    seed.pose.pose = odometry.pose.pose
    roll_pitch = math.radians(roll_pitch_sigma_deg) ** 2
    diagonal = (
        position_sigma_m ** 2,
        position_sigma_m ** 2,
        z_sigma_m ** 2,
        roll_pitch,
        roll_pitch,
        math.radians(yaw_sigma_deg) ** 2,
    )
    for axis, variance in enumerate(diagonal):
        seed.pose.covariance[axis * 7] = variance
    return seed


def icp_search_sigma_m(seed: PoseWithCovarianceStamped) -> float:
    """What LidarOdometry::relocalize_near_pose_pdf() will use as thres_sigma."""

    return math.sqrt(seed.pose.covariance[0] + seed.pose.covariance[7])


class RtkInitialPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("rtk_initial_pose_node")

        self.declare_parameter("rtk_pose_topic", "/gnss_imu/pose")
        # BridgeROS2's relocalize_from_topic; RViz's "2D Pose Estimate" uses
        # the same one, so the two seeds are interchangeable.
        self.declare_parameter("initial_pose_topic", "/initialpose")
        # gnss_imu_pose_node releases ~30 buffered poses in one burst when the
        # heading commits. Wait this long after the first one and seed from the
        # newest, not from the ~5 m stale first message.
        self.declare_parameter("settle_time_sec", 0.5)
        # Prior uncertainty, see the module docstring. sqrt(2) * position_sigma_m
        # becomes ICP's correspondence search radius; keep it under the
        # pipeline's maximum_sigma (2.0 m in lidar3d-dlio-like.yaml).
        self.declare_parameter("position_sigma_m", 1.0)
        self.declare_parameter("z_sigma_m", 1.0)
        self.declare_parameter("yaw_sigma_deg", 3.0)
        self.declare_parameter("roll_pitch_sigma_deg", 2.0)
        # How long to keep retrying while nothing subscribes to the seed topic
        # (MOLA still loading the map). 0 = forever.
        self.declare_parameter("subscriber_wait_timeout_sec", 60.0)

        self._settle_time = float(self.get_parameter("settle_time_sec").value)
        self._sigmas = {
            "position_sigma_m": float(self.get_parameter("position_sigma_m").value),
            "z_sigma_m": float(self.get_parameter("z_sigma_m").value),
            "yaw_sigma_deg": float(self.get_parameter("yaw_sigma_deg").value),
            "roll_pitch_sigma_deg": float(
                self.get_parameter("roll_pitch_sigma_deg").value
            ),
        }
        self._subscriber_wait_timeout = float(
            self.get_parameter("subscriber_wait_timeout_sec").value
        )

        self._latest: Odometry | None = None
        self._seeded = False
        self._settle_timer = None
        self._retry_timer = None
        self._waiting_since_ns: int | None = None

        self._seed_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            self.get_parameter("initial_pose_topic").value,
            10,
        )
        self.create_subscription(
            Odometry, self.get_parameter("rtk_pose_topic").value, self._on_rtk, 20
        )
        # Re-seed from the newest RTK pose on demand - e.g. after ICP has
        # visibly lost the map. Same path as the automatic first seed.
        self.create_service(Trigger, "~/reseed", self._on_reseed)

        self.get_logger().info(
            "rtk_initial_pose_node: will seed MOLA relocalization on "
            f"'{self._seed_pub.topic_name}' from the first heading-committed "
            f"'{self.get_parameter('rtk_pose_topic').value}' pose "
            f"(ICP search radius {math.sqrt(2.0) * self._sigmas['position_sigma_m']:.2f} m)"
        )

    def _on_rtk(self, msg: Odometry) -> None:
        self._latest = msg
        if self._seeded or self._settle_timer is not None:
            return
        # First heading-committed pose: let the burst finish, then seed.
        self._settle_timer = self.create_timer(self._settle_time, self._on_settled)

    def _on_settled(self) -> None:
        self._settle_timer.cancel()
        self._settle_timer = None
        self._try_seed(reason="first RTK pose")

    def _try_seed(self, reason: str) -> bool:
        if self._latest is None:
            return False
        try:
            seed = initial_pose_from_odometry(self._latest, **self._sigmas)
        except ValueError as error:
            self.get_logger().error(f"Not seeding relocalization: {error}")
            return False

        if self._seed_pub.get_subscription_count() == 0:
            # MOLA has not subscribed yet (still loading the map). Retry; the
            # seed is rebuilt from the newest pose each time so it never goes
            # stale while waiting.
            now_ns = self.get_clock().now().nanoseconds
            if self._waiting_since_ns is None:
                self._waiting_since_ns = now_ns
                self.get_logger().warn(
                    f"No subscriber on '{self._seed_pub.topic_name}' yet "
                    "(is MOLA up?) - holding the RTK seed until one appears."
                )
            elif (
                self._subscriber_wait_timeout > 0.0
                and (now_ns - self._waiting_since_ns) * 1e-9 > self._subscriber_wait_timeout
            ):
                self.get_logger().error(
                    f"Gave up seeding relocalization: nothing subscribed to "
                    f"'{self._seed_pub.topic_name}' within "
                    f"{self._subscriber_wait_timeout:.0f} s. LidarOdometry is "
                    "still at its FixedPose start."
                )
                self._stop_retrying()
                return False
            if self._retry_timer is None:
                self._retry_timer = self.create_timer(
                    1.0, lambda: self._try_seed(reason)
                )
            return False

        self._stop_retrying()
        self._seed_pub.publish(seed)
        self._seeded = True
        position = seed.pose.pose.position
        self.get_logger().info(
            f"Seeded MOLA relocalization from {reason}: "
            f"({position.x:.2f}, {position.y:.2f}, {position.z:.2f}) in "
            f"'{seed.header.frame_id}', ICP search radius "
            f"{icp_search_sigma_m(seed):.2f} m"
        )
        return True

    def _stop_retrying(self) -> None:
        if self._retry_timer is not None:
            self._retry_timer.cancel()
            self._retry_timer = None
        self._waiting_since_ns = None

    def _on_reseed(self, request, response):
        if self._latest is None:
            response.success = False
            response.message = "no RTK pose received yet"
            return response
        self._seeded = False
        response.success = self._try_seed(reason="operator reseed request")
        response.message = (
            "re-seeded from the newest RTK pose"
            if response.success
            else "could not seed - see the node log"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = RtkInitialPoseNode()
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
