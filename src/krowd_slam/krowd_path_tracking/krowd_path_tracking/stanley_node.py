"""Stanley path tracking against a live Trajectory.

Same contract as pure_pursuit_node - /planning/trajectory and
/localization/vehicle_state in, /target_angle, /target_speed and /brake out -
so route_publisher_node and vehicle_state_node serve either controller, and
path_tracking.launch.py starts one of them (controller_type). Never both: nothing
else may publish the actuator topics.

Read pure_pursuit_node's docstring before choosing this one. Stanley passes the
heading error to the steering at unity gain, and this vehicle's heading comes
from an RTK velocity course that carries ~9 deg of error near its 0.5 m/s gate.
Two things here exist because of that:

- The route heading comes from the route's GEOMETRY (stanley.path_heading), not
  from the recorded waypoint yaw, so the mapping drive's heading noise cannot
  reach the steering.
- max_heading_error_deg refuses to drive while the vehicle and the route differ
  by more than that. It is what a start that was not yaw-aligned with the route
  looks like, and what a heading glitch mid-route looks like.
"""

import math

import rclpy
from krowd_control_interfaces.msg import Trajectory, VehicleState
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker

from mola_slam_mapping.route_io import project_onto_route

from krowd_path_tracking.pure_pursuit import clamp_steering, rate_limited
from krowd_path_tracking.pure_pursuit_node import (
    TrackedTrajectory,
    end_of_route_reason,
    track_trajectory,
    yaw_from_quaternion,
)
from krowd_path_tracking.heading_creep import HeadingCreep, creep_permitted
from krowd_path_tracking.start_ramp import StartRamp
from krowd_path_tracking.route_geometry import (
    back_at_route_start,
    remaining_arc_length,
    value_at_route_position,
)
from krowd_path_tracking.stanley import (
    front_axle_position,
    heading_not_aligned,
    path_heading,
    steering_angle,
    wrap_angle,
)
from dataclasses import replace

from krowd_path_tracking.vehicle import (
    SteeringActuator,
    Vehicle,
    implausible_steering_limit_reason,
    speed_command_kmh,
    steering_limit_from_turning_radius,
)


class StanleyNode(Node):
    """Follow the published Trajectory with the Stanley law."""

    def __init__(self):
        super().__init__("stanley_node")

        self.declare_parameter("trajectory_topic", "/planning/trajectory")
        self.declare_parameter("vehicle_state_topic", "/localization/vehicle_state")
        self.declare_parameter("angle_topic", "/target_angle")
        self.declare_parameter("speed_topic", "/target_speed")
        self.declare_parameter("brake_topic", "/brake")
        self.declare_parameter("status_topic", "~/status")
        self.declare_parameter("cross_track_error_topic", "~/cross_track_error")
        self.declare_parameter("heading_error_topic", "~/heading_error")
        self.declare_parameter("target_marker_topic", "~/target_marker")

        # Vehicle geometry: the same parameters, from the same config/vehicle.yaml,
        # as pure_pursuit_node.
        self.declare_parameter("wheelbase_m", 1.340)
        self.declare_parameter("track_width_m", 1.080)
        self.declare_parameter("min_turning_radius_m", 3.4)
        self.declare_parameter("max_steering_angle_deg", 0.0)
        self.declare_parameter("max_steering_rate_deg_s", 60.0)
        self.declare_parameter("rear_axle_offset_m", 0.0)
        self.declare_parameter("steering_sign", 1.0)
        # Road-wheel angle -> /target_angle (see vehicle.SteeringActuator):
        # command = sign * gain * road_wheel + offset, clamped to
        # +/- max_steering_command_deg (the steer MCU rejects anything beyond).
        self.declare_parameter("steering_command_gain", 1.0)
        self.declare_parameter("steering_command_offset_rad", 0.0)
        self.declare_parameter("max_steering_command_deg", 142.0)

        self.declare_parameter("cross_track_gain", 1.0)
        self.declare_parameter("cross_track_softening_kmph", 3.6)
        self.declare_parameter("heading_gain", 1.0)
        self.declare_parameter("heading_window_m", 2.0)
        self.declare_parameter("max_heading_error_deg", 60.0)

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("vehicle_state_timeout_sec", 0.5)
        self.declare_parameter("goal_tolerance_m", 1.0)
        # After STOP_AT_END has latched, resume once the vehicle is back within
        # this distance of the route start (see back_at_route_start). It must
        # exceed the ~5 m of driving before the RTK pose reappears on a restarted
        # drive, plus the axle offset. 0 disables.
        self.declare_parameter("restart_near_start_m", 10.0)
        self.declare_parameter("corridor_half_width_m", 3.0)
        self.declare_parameter("stall_warning_period_sec", 5.0)
        # Speed cap for the first metres after tracking starts or resumes; the
        # route's speed profile has no ramp from standstill. See start_ramp.py.
        # 0 disables.
        self.declare_parameter("start_speed_kmph", 5.0)
        self.declare_parameter("start_accel_mps2", 1.0)
        # Before the FIRST vehicle state, with a route loaded: drive straight at
        # this speed so gnss_imu_pose_node can commit its heading (~5 m), for at
        # most this long; then brake. See heading_creep.py. 0 disables (waits).
        self.declare_parameter("heading_creep_speed_kmph", 5.0)
        self.declare_parameter("heading_creep_max_sec", 12.0)
        # ...and only once gnss_imu_pose_node reports NEEDS_MOTION here (no
        # verified ESKF heading, no verified LiDAR alignment). false: creep as
        # soon as a route exists without a vehicle state (the old behaviour).
        self.declare_parameter(
            "heading_acquisition_topic", "/gnss_imu/heading_acquisition"
        )
        self.declare_parameter("heading_creep_requires_request", True)

        self._vehicle = self._build_vehicle()
        self._actuator = self._build_actuator()
        self._steering_sign = float(self.get_parameter("steering_sign").value)
        if self._steering_sign not in (1.0, -1.0):
            # pure_pursuit_node multiplies by whatever is here, so a sign that
            # drifted into a gain there would silently rescale this law.
            raise RuntimeError(
                f"steering_sign must be +1.0 or -1.0, got {self._steering_sign}; "
                "tune cross_track_gain / heading_gain instead"
            )
        for name in (
            "cross_track_gain",
            "heading_gain",
            "heading_window_m",
            "max_heading_error_deg",
        ):
            value = float(self.get_parameter(name).value)
            if not math.isfinite(value) or value < 0.0:
                raise RuntimeError(f"{name} must be a non-negative number: {value}")
        self._cross_track_gain = float(self.get_parameter("cross_track_gain").value)
        # km/h at the parameter boundary, m/s below it - as in route_publisher_node.
        self._softening = (
            float(self.get_parameter("cross_track_softening_kmph").value) / 3.6
        )
        if not math.isfinite(self._softening) or self._softening <= 0.0:
            raise RuntimeError(
                "cross_track_softening_kmph must be positive: the law divides by "
                "(softening + speed) and is undefined at a standstill without it"
            )
        self._heading_gain = float(self.get_parameter("heading_gain").value)
        self._heading_window = float(self.get_parameter("heading_window_m").value)
        self._max_heading_error = math.radians(
            float(self.get_parameter("max_heading_error_deg").value)
        )
        self._state_timeout_ns = int(
            round(float(self.get_parameter("vehicle_state_timeout_sec").value) * 1e9)
        )
        self._goal_tolerance = float(self.get_parameter("goal_tolerance_m").value)
        self._restart_near_start = float(
            self.get_parameter("restart_near_start_m").value
        )
        self._corridor_half_width = float(
            self.get_parameter("corridor_half_width_m").value
        )
        self._start_ramp = StartRamp(
            float(self.get_parameter("start_speed_kmph").value) / 3.6,
            float(self.get_parameter("start_accel_mps2").value),
        )
        self._heading_creep = HeadingCreep(
            float(self.get_parameter("heading_creep_speed_kmph").value) / 3.6,
            float(self.get_parameter("heading_creep_max_sec").value),
        )

        self._angle_pub = self.create_publisher(
            Float32, str(self.get_parameter("angle_topic").value), 10
        )
        self._speed_pub = self.create_publisher(
            Float32, str(self.get_parameter("speed_topic").value), 10
        )
        self._brake_pub = self.create_publisher(
            Bool, str(self.get_parameter("brake_topic").value), 10
        )
        self._status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self._error_pub = self.create_publisher(
            Float32, str(self.get_parameter("cross_track_error_topic").value), 10
        )
        self._heading_error_pub = self.create_publisher(
            Float32, str(self.get_parameter("heading_error_topic").value), 10
        )
        self._marker_pub = self.create_publisher(
            Marker, str(self.get_parameter("target_marker_topic").value), 10
        )

        trajectory_qos = QoSProfile(depth=1)
        trajectory_qos.reliability = ReliabilityPolicy.RELIABLE
        trajectory_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            Trajectory,
            str(self.get_parameter("trajectory_topic").value),
            self._on_trajectory,
            trajectory_qos,
        )
        self.create_subscription(
            VehicleState,
            str(self.get_parameter("vehicle_state_topic").value),
            self._on_vehicle_state,
            10,
        )
        self._heading_state: str | None = None
        self._creep_requires_request = bool(
            self.get_parameter("heading_creep_requires_request").value
        )
        self.create_subscription(
            String,
            str(self.get_parameter("heading_acquisition_topic").value),
            self._on_heading_state,
            trajectory_qos,
        )

        self._trajectory: TrackedTrajectory | None = None
        self._vehicle_state: VehicleState | None = None
        self._vehicle_state_ns: int | None = None
        self._last_steering = 0.0
        self._goal_latched = False
        self._latched_trajectory_id: str | None = None
        self._last_control_ns: int | None = None
        self._last_status = ""

        rate = float(self.get_parameter("control_rate_hz").value)
        if rate <= 0.0:
            raise RuntimeError(f"control_rate_hz must be positive: {rate}")
        self._nominal_dt = 1.0 / rate
        self.create_timer(self._nominal_dt, self._control_once)

        # Wall-clock watchdog on the control loop - see pure_pursuit_node for the
        # silent use_sim_time failure it exists to report.
        self._control_cycles = 0
        self._cycles_at_last_check = -1
        self.create_timer(
            float(self.get_parameter("stall_warning_period_sec").value),
            self._warn_if_control_loop_stalled,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )

        alignment = (
            f"{math.degrees(self._max_heading_error):.0f} deg"
            if self._max_heading_error > 0.0
            else "disabled"
        )
        self.get_logger().info(
            f"stanley_node: wheelbase {self._vehicle.wheelbase_m:.3f} m (cross-track "
            "error at the front axle, "
            f"{self._vehicle.rear_axle_offset_m + self._vehicle.wheelbase_m:.3f} m "
            "ahead of base_link), steering limit "
            f"{math.degrees(self._vehicle.max_steering_angle_rad):.1f} deg (radius "
            f"{self._vehicle.implied_turning_radius_m:.2f} m), sign "
            f"{self._steering_sign:+.0f}, rate limit "
            f"{math.degrees(self._vehicle.max_steering_rate_rad_s):.1f} deg/s, "
            f"k {self._cross_track_gain:.2f} 1/s, softening "
            f"{self._softening * 3.6:.1f} km/h, heading gain "
            f"{self._heading_gain:.2f} over a {self._heading_window:.1f} m window, "
            f"heading alignment limit {alignment}, control at {rate:.1f} Hz"
        )

    def _build_actuator(self) -> SteeringActuator:
        """The road-wheel -> /target_angle mapping, and the limit it imposes.

        The road-wheel limit in use is the smaller of the vehicle's own and
        what the command range can express: with a gain of 8 and the MCU's
        2.5 rad (143 deg), that is 17.8 deg of road wheel however far the
        wheels turn.
        """

        actuator = SteeringActuator(
            gain=float(self.get_parameter("steering_command_gain").value),
            offset_rad=float(self.get_parameter("steering_command_offset_rad").value),
            max_command_rad=math.radians(
                float(self.get_parameter("max_steering_command_deg").value)
            ),
        )
        actuator.validate()
        expressible = actuator.road_wheel_limit_rad
        if expressible < self._vehicle.max_steering_angle_rad:
            self.get_logger().info(
                f"road-wheel steering limit {math.degrees(self._vehicle.max_steering_angle_rad):.1f} deg "
                f"reduced to {math.degrees(expressible):.1f} deg: the command range "
                f"(+/-{math.degrees(actuator.max_command_rad):.0f} deg at gain "
                f"{actuator.gain:.2f}) cannot express more - turning radius "
                f"{self._vehicle.wheelbase_m / math.tan(expressible):.2f} m"
            )
            self._vehicle = replace(self._vehicle, max_steering_angle_rad=expressible)
        return actuator

    def _build_vehicle(self) -> Vehicle:
        wheelbase = float(self.get_parameter("wheelbase_m").value)
        min_radius = float(self.get_parameter("min_turning_radius_m").value)
        override_deg = float(self.get_parameter("max_steering_angle_deg").value)
        if override_deg > 0.0:
            max_steering = math.radians(override_deg)
        else:
            max_steering = steering_limit_from_turning_radius(wheelbase, min_radius)

        vehicle = Vehicle(
            wheelbase_m=wheelbase,
            track_width_m=float(self.get_parameter("track_width_m").value),
            min_turning_radius_m=min_radius,
            max_steering_angle_rad=max_steering,
            max_steering_rate_rad_s=math.radians(
                float(self.get_parameter("max_steering_rate_deg_s").value)
            ),
            rear_axle_offset_m=float(self.get_parameter("rear_axle_offset_m").value),
        )
        if vehicle.max_steering_angle_rad <= 0.0:
            raise RuntimeError(
                "steering limit resolved to zero or less; set max_steering_angle_deg "
                "or min_turning_radius_m"
            )
        reason = implausible_steering_limit_reason(vehicle)
        if reason is not None:
            self.get_logger().warn(f"suspicious steering limit: {reason}")
        return vehicle

    def _on_trajectory(self, msg: Trajectory) -> None:
        poses = msg.path.poses
        count = len(poses)
        if count < 2:
            self.get_logger().warn(
                f"ignoring trajectory '{msg.trajectory_id}': {count} poses"
            )
            return
        if len(msg.target_speeds) != count or len(msg.curvatures) != count:
            self.get_logger().error(
                f"ignoring trajectory '{msg.trajectory_id}': {count} poses but "
                f"{len(msg.curvatures)} curvatures and {len(msg.target_speeds)} "
                "target speeds"
            )
            return

        if msg.trajectory_id != self._latched_trajectory_id:
            # A new plan releases a latched goal; a republish of the same one
            # must not.
            self._goal_latched = False

        self._trajectory = track_trajectory(msg, self._corridor_half_width)

    def _on_vehicle_state(self, msg: VehicleState) -> None:
        self._vehicle_state = msg
        self._vehicle_state_ns = self.get_clock().now().nanoseconds
        self._heading_creep.vehicle_state_seen()

    def _warn_if_control_loop_stalled(self) -> None:
        if self._control_cycles != self._cycles_at_last_check:
            self._cycles_at_last_check = self._control_cycles
            return
        self._cycles_at_last_check = self._control_cycles

        if bool(self.get_parameter("use_sim_time").value):
            cause = (
                "use_sim_time is true and nothing is advancing /clock - start "
                "the bag (ros2 bag play ... --clock), or drop use_sim_time:=true "
                "if you are not replaying one"
            )
        else:
            cause = "the control timer is not firing"
        self.get_logger().error(
            f"control loop has not run for "
            f"{self.get_parameter('stall_warning_period_sec').value} s "
            f"({self._control_cycles} cycles total): {cause}. No command is "
            "being published on /target_angle, /target_speed or /brake."
        )

    def _control_once(self) -> None:
        self._control_cycles += 1
        now_ns = self.get_clock().now().nanoseconds
        dt = self._nominal_dt
        if self._last_control_ns is not None:
            elapsed = (now_ns - self._last_control_ns) * 1e-9
            if 0.0 < elapsed < 10.0 * self._nominal_dt:
                dt = elapsed
        self._last_control_ns = now_ns

        reason = self._blocking_reason(now_ns)
        if reason is not None:
            if reason == "NO_VEHICLE_STATE":
                self._before_first_vehicle_state(now_ns)
            else:
                self._publish_stop(reason)
            return

        trajectory = self._trajectory
        state = self._vehicle_state
        pose = state.odometry.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        front_x, front_y = front_axle_position(
            float(pose.position.x),
            float(pose.position.y),
            yaw,
            self._vehicle.rear_axle_offset_m,
            self._vehicle.wheelbase_m,
        )

        projection = project_onto_route(trajectory.route, front_x, front_y)
        if projection is None:
            self._publish_stop("NO_ROUTE_PROJECTION")
            return

        remaining = remaining_arc_length(
            trajectory.planar, projection.segment_index, projection.fraction
        )
        progress = end_of_route_reason(trajectory, remaining, self._goal_tolerance)
        if progress is not None:
            if trajectory.goal_mode == Trajectory.STOP_AT_END:
                self._goal_latched = True
                self._latched_trajectory_id = trajectory.trajectory_id
            self._publish_stop(progress)
            return

        ahead = trajectory.ahead_of(projection)
        route_yaw = path_heading(
            trajectory.ahead_planar,
            ahead.segment_index,
            ahead.fraction,
            self._heading_window,
        )
        heading_error = wrap_angle(route_yaw - yaw)
        self._error_pub.publish(Float32(data=float(projection.signed_offset)))
        self._heading_error_pub.publish(Float32(data=float(heading_error)))
        self._publish_target_marker(trajectory, projection)

        if heading_not_aligned(heading_error, self._max_heading_error):
            self._publish_stop(
                f"HEADING_NOT_ALIGNED heading_error="
                f"{math.degrees(heading_error):+.1f}deg limit="
                f"{math.degrees(self._max_heading_error):.0f}deg - align the "
                "vehicle with the route direction"
            )
            return

        speed = float(state.odometry.twist.twist.linear.x)
        desired = steering_angle(
            heading_error,
            projection.signed_offset,
            speed,
            self._cross_track_gain,
            self._softening,
            self._heading_gain,
        )
        desired = clamp_steering(desired, self._vehicle.max_steering_angle_rad)
        steering = rate_limited(
            self._last_steering, desired, self._vehicle.max_steering_rate_rad_s, dt
        )
        self._last_steering = steering

        target_speed = value_at_route_position(
            trajectory.target_speeds, projection.segment_index, projection.fraction
        )
        speed_cap = self._start_ramp.cap(speed, front_x, front_y)
        ramping = speed_cap < target_speed
        target_speed = min(target_speed, speed_cap)

        command = self._actuator.command(steering, self._steering_sign)
        self._angle_pub.publish(Float32(data=float(command)))
        self._speed_pub.publish(Float32(data=speed_command_kmh(target_speed)))
        self._brake_pub.publish(Bool(data=False))
        self._publish_status(
            f"TRACKING wp={projection.route_position:.1f} "
            f"cte={projection.signed_offset:+.2f}m "
            f"heading_error={math.degrees(heading_error):+.1f}deg "
            f"steer={math.degrees(steering):+.1f}deg cmd={math.degrees(command):+.0f}deg "
            f"v={speed * 3.6:.1f}->{target_speed * 3.6:.1f}km/h "
            + ("loop" if trajectory.closed_loop else f"remaining={remaining:.1f}m")
            + (" ramp" if ramping else "")
        )

    def _release_goal_latch_at_route_start(self, now_ns: int) -> bool:
        """Resume after a goal stop once the vehicle is back at the route start."""

        state, trajectory = self._vehicle_state, self._trajectory
        if (
            self._restart_near_start <= 0.0
            or state is None
            or self._vehicle_state_ns is None
            or trajectory is None
            or now_ns - self._vehicle_state_ns > self._state_timeout_ns
            or now_ns > trajectory.valid_until_ns
            or state.odometry.header.frame_id != trajectory.frame_id
        ):
            return False
        pose = state.odometry.pose.pose
        front_x, front_y = front_axle_position(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
            self._vehicle.rear_axle_offset_m,
            self._vehicle.wheelbase_m,
        )
        projection = project_onto_route(trajectory.route, front_x, front_y)
        if projection is None or not back_at_route_start(
            trajectory.planar,
            projection.segment_index,
            projection.fraction,
            front_x,
            front_y,
            self._restart_near_start,
            self._corridor_half_width,
            # Well clear of where a goal stop leaves the vehicle.
            2.0 * self._goal_tolerance,
        ):
            return False
        self._goal_latched = False
        self._latched_trajectory_id = None
        self.get_logger().info(
            "goal latch released: vehicle is back at the route start "
            f"(wp {projection.route_position:.1f}, "
            f"{projection.signed_offset:+.2f} m off); tracking again"
        )
        return True

    def _blocking_reason(self, now_ns: int) -> str | None:
        """Why control cannot run this cycle; None when it can."""

        if self._goal_latched and not self._release_goal_latch_at_route_start(now_ns):
            return "GOAL_LATCHED"
        if self._vehicle_state is None or self._vehicle_state_ns is None:
            return "NO_VEHICLE_STATE"
        age_ns = now_ns - self._vehicle_state_ns
        if age_ns > self._state_timeout_ns:
            return f"STALE_VEHICLE_STATE age={age_ns * 1e-9:.2f}s"
        if self._trajectory is None:
            return "NO_TRAJECTORY"
        if now_ns > self._trajectory.valid_until_ns:
            expired = (now_ns - self._trajectory.valid_until_ns) * 1e-9
            return f"TRAJECTORY_EXPIRED by={expired:.2f}s"
        state_frame = self._vehicle_state.odometry.header.frame_id
        if state_frame != self._trajectory.frame_id:
            return (
                f"FRAME_MISMATCH trajectory='{self._trajectory.frame_id}' "
                f"vehicle_state='{state_frame}'"
            )
        return None

    def _publish_stop(self, reason: str) -> None:
        """Command a stop, holding the last steering angle (see pure_pursuit_node)."""

        self._start_ramp.reset()
        self._angle_pub.publish(
            Float32(data=float(self._actuator.command(self._last_steering, self._steering_sign)))
        )
        self._speed_pub.publish(Float32(data=0.0))
        self._brake_pub.publish(Bool(data=True))
        self._publish_status(f"STOP {reason}")

    def _publish_waiting(self, reason: str) -> None:
        """Report that nothing is commanded yet, and command nothing.

        Before the first vehicle state there is nothing to steer against and
        nothing to protect either: the vehicle has not been under this node's
        control. gnss_imu_pose_node publishes its first pose only after ~5 m
        of driving, and that drive is done by hand - a brake held on the CAN
        bus the whole time is exactly what would stop it. ros_can_bridge's
        own speed watchdog keeps the target speed at zero while no command
        arrives. Losing the state later (STALE_VEHICLE_STATE) still brakes.
        """

        self._start_ramp.reset()
        self._publish_status(f"WAITING {reason}")

    def _on_heading_state(self, msg: String) -> None:
        self._heading_state = msg.data

    def _before_first_vehicle_state(self, now_ns: int) -> None:
        """No pose has ever arrived: creep straight to earn one, or just wait.

        Only with a live trajectory - a creep is a real command to the vehicle,
        so the rest of the stack has to be up. The creep ends with the first
        vehicle state (tracking takes over) or times out into a braked stop.
        """

        trajectory = self._trajectory
        if trajectory is None or now_ns > trajectory.valid_until_ns:
            self._publish_waiting("NO_VEHICLE_STATE")
            return
        if not self._heading_creep.started and not creep_permitted(
            self._heading_state, self._creep_requires_request
        ):
            # Localization is still getting the heading without motion (ESKF,
            # LiDAR): stay put. A creep, once started, runs to its end.
            self._publish_waiting(
                f"NO_VEHICLE_STATE heading={self._heading_state or 'UNKNOWN'}"
            )
            return
        verdict = self._heading_creep.command(now_ns)
        if verdict == HeadingCreep.CREEP:
            self._publish_creep(now_ns)
        elif verdict == HeadingCreep.TIMEOUT:
            self._publish_stop(
                "HEADING_NOT_ACQUIRED "
                f"after={self._heading_creep.max_duration_s:.0f}s - no vehicle "
                "state arrived while creeping; restart path tracking to retry"
            )
        else:
            self._publish_waiting("NO_VEHICLE_STATE")

    def _publish_creep(self, now_ns: int) -> None:
        """Straight ahead at the creep speed; the brake is released first so
        ros_can_bridge accepts the speed that follows it."""

        self._start_ramp.reset()
        self._last_steering = 0.0
        self._brake_pub.publish(Bool(data=False))
        self._angle_pub.publish(
            Float32(data=float(self._actuator.command(0.0, self._steering_sign)))
        )
        self._speed_pub.publish(Float32(data=speed_command_kmh(self._heading_creep.speed_mps)))
        self._publish_status(
            "CREEP HEADING_ACQUISITION "
            f"elapsed={self._heading_creep.elapsed_s(now_ns):.1f}s "
            f"max={self._heading_creep.max_duration_s:.0f}s "
            f"v={self._heading_creep.speed_mps * 3.6:.1f}km/h steer=0"
        )

    def _publish_status(self, status: str) -> None:
        self._status_pub.publish(String(data=status))
        # Log on a change of state. For STOP that includes its reason, so a stop
        # that turns from NO_VEHICLE_STATE into HEADING_NOT_ALIGNED is not silent.
        parts = status.split(" ", 2)
        head = " ".join(parts[:2]) if parts[0] == "STOP" else parts[0]
        if head != self._last_status:
            self._last_status = head
            self.get_logger().info(status)

    def _publish_target_marker(self, trajectory: TrackedTrajectory, projection) -> None:
        current = trajectory.route[projection.segment_index]
        following = trajectory.route[projection.segment_index + 1]
        fraction = projection.fraction

        marker = Marker()
        marker.header.frame_id = trajectory.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "stanley_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = current.x + fraction * (following.x - current.x)
        marker.pose.position.y = current.y + fraction * (following.y - current.y)
        marker.pose.position.z = current.z + fraction * (following.z - current.z)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.6
        # Magenta: the front axle's foot point on the route, where the
        # cross-track error is measured. Distinct from pure_pursuit_node's cyan
        # lookahead and localization_monitor_node's red display-only sphere.
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        self._marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = StanleyNode()
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
