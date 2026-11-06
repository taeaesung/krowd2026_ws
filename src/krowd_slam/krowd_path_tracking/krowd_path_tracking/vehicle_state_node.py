"""Adapt the localization Odometry into the VehicleState control consumes.

Thin on purpose. VehicleState.msg says "Curvature is estimated upstream, not by
control", and this node is that upstream: it is the one place that decides what
the vehicle's instantaneous curvature is, so the controller cannot quietly
disagree with anything else that asks the same question.

Its input is gnss_imu_pose_node's Odometry on `/gnss_imu/pose`, whose twist
carries the RTK ground speed in linear.x and the yaw rate in angular.z.
"""

import math

import rclpy
from krowd_control_interfaces.msg import VehicleState
from nav_msgs.msg import Odometry
from rclpy.node import Node


class VehicleStateNode(Node):
    """Republish localization odometry as VehicleState with a curvature estimate."""

    def __init__(self):
        super().__init__("vehicle_state_node")

        # `/gnss_imu/pose`, NOT `/state_estimation/pose`. The latter reads like
        # the obvious choice and nothing publishes it: localization.launch.py
        # sets `state_pose_topic` to `/gnss_imu/pose` and overrides
        # localization_monitor_node's identically-misleading default the same
        # way. The name `/state_estimation/*` belongs to a MOLA smoother that is
        # deliberately not the active estimator.
        self.declare_parameter("input_odom_topic", "/gnss_imu/pose")
        self.declare_parameter("vehicle_state_topic", "/localization/vehicle_state")
        # Below this speed w/v is not a curvature, it is a division by noise: a
        # stationary vehicle with 0.02 rad/s of yaw jitter would report a 0.5 m
        # radius. Report straight instead and let the controller's own geometry
        # decide what to do at a standstill.
        self.declare_parameter("min_speed_for_curvature_kmph", 1.8)
        self.declare_parameter("report_period_sec", 10.0)

        # km/h at the parameter boundary, m/s below it: the Odometry consumed
        # and the VehicleState published here are both SI.
        self._min_speed_for_curvature = (
            float(self.get_parameter("min_speed_for_curvature_kmph").value) / 3.6
        )

        self._publisher = self.create_publisher(
            VehicleState, str(self.get_parameter("vehicle_state_topic").value), 10
        )
        input_topic = str(self.get_parameter("input_odom_topic").value)
        self.create_subscription(Odometry, input_topic, self._on_odometry, 10)

        self._received = 0
        self._moved_without_speed = 0
        self._last_position = None
        self.create_timer(
            float(self.get_parameter("report_period_sec").value), self._report
        )

        self.get_logger().info(
            f"vehicle_state_node: '{input_topic}' -> "
            f"'{self.get_parameter('vehicle_state_topic').value}', curvature from "
            f"yaw rate above {self._min_speed_for_curvature * 3.6:.1f} km/h"
        )

    def _on_odometry(self, msg: Odometry) -> None:
        speed = float(msg.twist.twist.linear.x)
        yaw_rate = float(msg.twist.twist.angular.z)

        if not math.isfinite(speed) or not math.isfinite(yaw_rate):
            self.get_logger().warn(
                f"discarding non-finite twist: speed={speed}, yaw_rate={yaw_rate}"
            )
            return

        self._track_speedless_motion(msg, speed)

        state = VehicleState()
        state.odometry = msg
        if abs(speed) >= self._min_speed_for_curvature:
            state.curvature = yaw_rate / speed
        else:
            state.curvature = 0.0
        self._publisher.publish(state)
        self._received += 1

    def _track_speedless_motion(self, msg: Odometry, speed: float) -> None:
        """Count poses that move while the twist claims a standstill.

        The failure this catches is a localization source that fills pose but
        leaves twist at zero. Nothing downstream would fault - the lookahead
        would simply collapse to its minimum and the vehicle would crawl - so it
        has to be reported rather than detected by feel.
        """

        position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self._last_position is not None and speed == 0.0:
            moved = math.hypot(
                position[0] - self._last_position[0],
                position[1] - self._last_position[1],
            )
            if moved > 0.05:
                self._moved_without_speed += 1
        self._last_position = position

    def _report(self) -> None:
        if self._moved_without_speed > 0:
            self.get_logger().warn(
                f"{self._moved_without_speed} of {self._received} poses moved while "
                "reporting exactly zero speed: the localization twist is probably "
                "not being filled. Pure Pursuit scales its lookahead with speed, so "
                "it will run at the minimum lookahead and track poorly."
            )
        self._moved_without_speed = 0


def main(args=None):
    rclpy.init(args=args)
    node = VehicleStateNode()
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
