"""Closed-loop vehicle simulator for the controller's command topics.

Stands in for everything the controller would get from the vehicle and from
the map launch:

* the plant: /simulation/target_angle|target_speed|brake drive a kinematic
  bicycle (measured steering lag and drive speed deficit included) whose pose is
  published on /simulation/odometry and as map -> base_link on /tf;
* the map launch: with `map_dir` set it publishes the mapping run's
  georeference (utm -> map static TF, from <map_dir>/georeference.yaml) and
  announces the run on /localization/map_dir (latched), exactly as
  localization.launch.py does on the vehicle. The route publisher and the
  controller then pick the map up unchanged, in the same `map` frame as on the
  vehicle - so the lane-boundary CSVs (map frame) line up in simulation too.

Do NOT run it together with localization.launch.py or the sensor stack: both
would publish the same TF edges.

The plant turns /target_angle back into a road-wheel angle with the SAME
steering table the controller uses (control.yaml, steering_calibration.py).
"""

import math
import os

import rclpy
import tf2_ros
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from krowd_control_interfaces.msg import Trajectory
from nav_msgs.msg import Odometry, Path
from ros_can_bridge_msgs.msg import EncoderAngleStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker

from .bicycle_sim import (
    BicycleState,
    DelayedCommand,
    achieved_speed,
    step_bicycle,
)
from .config import CONTROL_CONFIG, default_config_dir, load_control_config
from .steering_calibration import SteeringCalibration


def yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
    )


def georeference_transform(map_dir: str) -> tuple[str, str, dict, dict]:
    """(parent, child, translation, rotation) of the run's utm -> map placement."""

    path = os.path.join(os.path.expanduser(map_dir), "georeference.yaml")
    with open(path, "r", encoding="utf-8") as stream:
        georef = yaml.safe_load(stream)
    placement = georef["route_local_frame_to_global"]
    t = placement["source_to_target"]
    return (
        str(placement.get("target_frame", "utm")),
        str(placement.get("source_frame", "map")),
        {k: float(t["translation"][k]) for k in "xyz"},
        {k: float(t["rotation"][k]) for k in "xyzw"},
    )


class VehicleDynamicsSimNode(Node):
    def __init__(self):
        super().__init__("vehicle_dynamics_sim_node")
        self.declare_parameter("control_config", os.path.join(default_config_dir(), CONTROL_CONFIG))
        self.declare_parameter("control_overrides", "")
        # Mapping run directory: publishes its georeference and announces it on
        # /localization/map_dir. "" = neither (something else provides them).
        self.declare_parameter("map_dir", "")
        self.declare_parameter("map_dir_topic", "/localization/map_dir")
        self.declare_parameter("trajectory_topic", "/planning/trajectory")
        self.declare_parameter("angle_topic", "/simulation/target_angle")
        self.declare_parameter("speed_topic", "/simulation/target_speed")
        self.declare_parameter("brake_topic", "/simulation/brake")
        self.declare_parameter("odometry_topic", "/simulation/odometry")
        # The plant's road-wheel angle back as a RAW encoder reading, like the
        # vehicle's CAN bridge (the controller's steer_hold reads it). "" = off.
        self.declare_parameter("encoder_topic", "/encoder")
        self.declare_parameter("vehicle_marker_topic", "/simulation/vehicle_marker")
        self.declare_parameter("driven_path_topic", "/simulation/driven_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("max_accel_mps2", 2.5)
        # Coasting deceleration of the real drive when the target drops (the
        # controller's profile assumes 0.5; the plant does the same by default).
        self.declare_parameter("coast_decel_mps2", 0.5)
        # Brake deceleration when /brake is true (measured envelope ~3.95).
        self.declare_parameter("brake_decel_mps2", 3.95)
        self.declare_parameter("max_steering_rate_deg_s", 40.0)
        self.declare_parameter("simulation_rate_hz", 50.0)
        # The real vehicle's measured imperfections (stanley_tuning_0922_235531),
        # ON by default so the simulation is not kinder than the car. 0 = ideal.
        self.declare_parameter("steering_lag_sec", 0.24)
        self.declare_parameter("speed_deficit_kmph", 1.7)
        self.declare_parameter("initial_waypoint_index", 0)
        self.declare_parameter("initial_lateral_offset_m", 0.0)
        self.declare_parameter("initial_heading_offset_deg", 0.0)

        cfg = load_control_config(
            str(self.get_parameter("control_config").value),
            str(self.get_parameter("control_overrides").value) or None,
        )
        self._wheelbase = float(cfg["vehicle"]["wheelbase_m"])
        self._length = float(cfg["vehicle"]["length_m"])
        self._width = float(cfg["vehicle"]["width_m"])
        self._calib = SteeringCalibration.from_config(cfg["steering"])

        self._max_accel = float(self.get_parameter("max_accel_mps2").value)
        self._coast_decel = float(self.get_parameter("coast_decel_mps2").value)
        self._brake_decel = float(self.get_parameter("brake_decel_mps2").value)
        self._max_steering_rate = math.radians(
            float(self.get_parameter("max_steering_rate_deg_s").value)
        )
        rate = float(self.get_parameter("simulation_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("simulation_rate_hz must be positive")

        self._state = None
        self._z = 0.0
        self._target_speed = 0.0
        self._target_steering = 0.0
        self._steering_delay = DelayedCommand(
            float(self.get_parameter("steering_lag_sec").value)
        )
        self._speed_deficit = float(self.get_parameter("speed_deficit_kmph").value) / 3.6
        self._brake = True
        self._last_time = None
        self._yaw_rate = 0.0

        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # ---- the map launch's part: georeference + map announcement ----------
        self._static_broadcaster = None
        map_dir = str(self.get_parameter("map_dir").value).strip()
        if map_dir:
            parent, child, translation, rotation = georeference_transform(map_dir)
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent
            transform.child_frame_id = child
            transform.transform.translation.x = translation["x"]
            transform.transform.translation.y = translation["y"]
            transform.transform.translation.z = translation["z"]
            transform.transform.rotation.x = rotation["x"]
            transform.transform.rotation.y = rotation["y"]
            transform.transform.rotation.z = rotation["z"]
            transform.transform.rotation.w = rotation["w"]
            self._static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
            self._static_broadcaster.sendTransform(transform)
            announce = self.create_publisher(String, str(self.get_parameter("map_dir_topic").value), latched)
            announce.publish(String(data=os.path.expanduser(map_dir)))
            self._announce = announce
            self.get_logger().info(
                f"map '{map_dir}': published {parent} -> {child} georeference and announced it on "
                f"{self.get_parameter('map_dir_topic').value}"
            )

        self._publisher = self.create_publisher(
            Odometry, str(self.get_parameter("odometry_topic").value), 10
        )
        self._tf_broadcaster = (
            tf2_ros.TransformBroadcaster(self)
            if bool(self.get_parameter("publish_tf").value)
            else None
        )
        self._marker_publisher = self.create_publisher(
            Marker, str(self.get_parameter("vehicle_marker_topic").value), 10
        )
        encoder_topic = str(self.get_parameter("encoder_topic").value)
        self._encoder_publisher = (
            self.create_publisher(EncoderAngleStamped, encoder_topic, 10) if encoder_topic else None
        )
        self._path_publisher = self.create_publisher(
            Path, str(self.get_parameter("driven_path_topic").value), latched
        )
        self._path = Path()
        self._path_last_xy = None
        self.create_subscription(
            Trajectory, str(self.get_parameter("trajectory_topic").value), self._on_trajectory, latched,
        )
        self.create_subscription(Float32, str(self.get_parameter("angle_topic").value), self._on_angle, 10)
        self.create_subscription(Float32, str(self.get_parameter("speed_topic").value), self._on_speed, 10)
        self.create_subscription(Bool, str(self.get_parameter("brake_topic").value), self._on_brake, 10)
        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(0.5, self._publish_path)
        self.get_logger().info(
            "Closed-loop bicycle simulator waiting for /planning/trajectory; "
            f"controller commands feed {self.get_parameter('odometry_topic').value}; "
            f"vehicle footprint on {self.get_parameter('vehicle_marker_topic').value}; "
            + self._calib.describe()
        )

    def _on_trajectory(self, message: Trajectory) -> None:
        if self._state is not None or not message.path.poses:
            return
        frame = str(self.get_parameter("frame_id").value)
        if message.path.header.frame_id and message.path.header.frame_id != frame:
            self.get_logger().error(
                f"trajectory is in '{message.path.header.frame_id}' but the simulator publishes in "
                f"'{frame}'; not starting",
                throttle_duration_sec=10.0,
            )
            return
        index = min(
            max(int(self.get_parameter("initial_waypoint_index").value), 0),
            len(message.path.poses) - 1,
        )
        pose = message.path.poses[index].pose
        route_yaw = yaw_from_quaternion(pose.orientation)
        heading = route_yaw + math.radians(
            float(self.get_parameter("initial_heading_offset_deg").value)
        )
        lateral = float(self.get_parameter("initial_lateral_offset_m").value)
        self._state = BicycleState(
            x=pose.position.x - lateral * math.sin(route_yaw),
            y=pose.position.y + lateral * math.cos(route_yaw),
            yaw=heading,
            speed_mps=0.0,
            steering_rad=0.0,
        )
        self._z = pose.position.z
        self._last_time = self.get_clock().now()
        self.get_logger().info(
            f"Simulator initialized at waypoint {index} of '{message.trajectory_id}': "
            f"({self._state.x:.2f}, {self._state.y:.2f}) in {frame}, "
            f"yaw {math.degrees(self._state.yaw):.1f} deg; steering lag "
            f"{self._steering_delay.delay_s:.2f} s, speed deficit "
            f"{self._speed_deficit * 3.6:.1f} km/h, coasting {self._coast_decel:.2f} m/s^2"
        )
        self._publish()

    def _on_angle(self, message: Float32) -> None:
        command = float(message.data)
        if math.isfinite(command):
            self._target_steering = self._calib.road_wheel_from_encoder(command)
            self._steering_delay.push(
                self.get_clock().now().nanoseconds * 1e-9, self._target_steering
            )

    def _on_speed(self, message: Float32) -> None:
        speed_kmph = float(message.data)
        if math.isfinite(speed_kmph):
            self._target_speed = max(speed_kmph, 0.0) / 3.6

    def _on_brake(self, message: Bool) -> None:
        self._brake = bool(message.data)

    def _tick(self) -> None:
        if self._state is None:
            return
        now = self.get_clock().now()
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        if not 0.0 < dt <= 0.2:
            return
        target_speed = (
            0.0 if self._brake else achieved_speed(self._target_speed, self._speed_deficit)
        )
        decel = self._brake_decel if self._brake else self._coast_decel
        self._state, self._yaw_rate = step_bicycle(
            self._state,
            target_speed,
            self._steering_delay.value(now.nanoseconds * 1e-9),
            dt,
            self._wheelbase,
            self._max_accel,
            decel,
            self._max_steering_rate,
        )
        self._publish()

    def _publish(self) -> None:
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.child_frame_id = str(self.get_parameter("child_frame_id").value)
        message.pose.pose.position.x = self._state.x
        message.pose.pose.position.y = self._state.y
        message.pose.pose.position.z = self._z
        message.pose.pose.orientation.z = math.sin(0.5 * self._state.yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * self._state.yaw)
        message.twist.twist.linear.x = self._state.speed_mps
        message.twist.twist.angular.z = self._yaw_rate
        self._publisher.publish(message)
        if self._encoder_publisher is not None:
            encoder = EncoderAngleStamped()
            encoder.header = message.header
            encoder.angle_rad = float(self._calib.encoder_from_road_wheel(self._state.steering_rad))
            self._encoder_publisher.publish(encoder)
        if self._tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = message.header
            transform.child_frame_id = message.child_frame_id
            transform.transform.translation.x = message.pose.pose.position.x
            transform.transform.translation.y = message.pose.pose.position.y
            transform.transform.translation.z = message.pose.pose.position.z
            transform.transform.rotation = message.pose.pose.orientation
            self._tf_broadcaster.sendTransform(transform)

        xy = (self._state.x, self._state.y)
        if self._path_last_xy is None or math.hypot(xy[0] - self._path_last_xy[0], xy[1] - self._path_last_xy[1]) >= 0.5:
            self._path_last_xy = xy
            pose = PoseStamped()
            pose.header = message.header
            pose.pose = message.pose.pose
            self._path.poses.append(pose)

        marker = Marker()
        marker.header = message.header
        marker.ns = "dynamics_vehicle"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = True
        marker.pose.position.x = message.pose.pose.position.x
        marker.pose.position.y = message.pose.pose.position.y
        marker.pose.position.z = message.pose.pose.position.z + 0.45
        marker.pose.orientation = message.pose.pose.orientation
        marker.scale.x = self._length
        marker.scale.y = self._width
        marker.scale.z = 0.9
        marker.color.r = 0.05
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.8
        self._marker_publisher.publish(marker)

    def _publish_path(self) -> None:
        if not self._path.poses:
            return
        self._path.header.stamp = self.get_clock().now().to_msg()
        self._path.header.frame_id = str(self.get_parameter("frame_id").value)
        self._path_publisher.publish(self._path)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleDynamicsSimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):  # Ctrl+C, or the launch's SIGINT
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
