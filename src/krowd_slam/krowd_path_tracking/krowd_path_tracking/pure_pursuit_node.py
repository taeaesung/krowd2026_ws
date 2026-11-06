"""Pure Pursuit path tracking against a live Trajectory.

Why Pure Pursuit and not Stanley, since the choice is not obvious and the code
does not otherwise record it: Stanley's first term is the heading error at unity
gain, and this vehicle's heading comes from an RTK velocity course that is
gated at 0.5 m/s and carries ~9 deg of error near that gate. Pure Pursuit takes
the same heading error through the lookahead geometry instead, which attenuates
it by roughly 2 L / ld, and it never reads the route's own yaw - so the recorded
drive's yaw noise cannot reach the steering command at all. It is also defined
at a standstill, where Stanley's atan(k e / v) is not.

The controller commands ros_can_bridge's actuator topics directly
(/target_angle, /target_speed, /brake). Nothing else may publish them at the
same time.
"""

from dataclasses import dataclass, replace
import math

import rclpy
from krowd_control_interfaces.msg import Trajectory, VehicleState
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker

from mola_slam_mapping.route_io import (
    RouteProjection,
    point_ahead_on_route,
    project_onto_route,
)
from mola_slam_mapping.route_visualization import RoutePoint, make_route_point

from krowd_path_tracking.pure_pursuit import (
    clamp_steering,
    curvature_lookahead,
    lookahead_distance,
    rate_limited,
    rear_axle_position,
    steering_angle,
    to_body_frame,
)
from krowd_path_tracking.heading_creep import HeadingCreep, creep_permitted
from krowd_path_tracking.start_ramp import StartRamp
from krowd_path_tracking.route_geometry import (
    back_at_route_start,
    max_abs_value_ahead,
    remaining_arc_length,
    unroll_loop,
    value_at_route_position,
)
from krowd_path_tracking.vehicle import (
    SteeringActuator,
    Vehicle,
    implausible_steering_limit_reason,
    speed_command_kmh,
    steering_limit_from_turning_radius,
)


def yaw_from_quaternion(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


@dataclass(frozen=True)
class TrackedTrajectory:
    """One Trajectory, converted once into the shapes the controller walks.

    `route`/`planar` are the trajectory as published and are what the vehicle
    is projected onto. `ahead_route`/`ahead_planar` are what is walked forwards
    (and, for Stanley's heading window, backwards) from that projection: the
    same route for an open one, three laps of it for a closed loop, entered at
    the middle lap (`lap_offset`) so a walk crosses the seam instead of stopping
    at the last waypoint. See ahead_of.
    """

    trajectory_id: str
    frame_id: str
    goal_mode: int
    valid_until_ns: int
    route: list[RoutePoint]
    planar: list[tuple[float, float]]
    target_speeds: list[float]
    closed_loop: bool
    ahead_route: list[RoutePoint]
    ahead_planar: list[tuple[float, float]]
    lap_offset: int
    # Per waypoint of ahead_route (1/m, signed): the route publisher's smoothed
    # curvature, walked ahead to shorten the lookahead into corners.
    ahead_curvatures: list[float] = ()

    def ahead_of(self, projection: RouteProjection) -> RouteProjection:
        """The projection re-expressed on ahead_route/ahead_planar."""

        if self.lap_offset == 0:
            return projection
        return replace(
            projection, segment_index=projection.segment_index + self.lap_offset
        )


def track_trajectory(msg: Trajectory, corridor_half_width: float) -> TrackedTrajectory:
    """Convert a validated Trajectory for either controller.

    A trajectory is a closed loop when its last pose repeats its first, which is
    how route_publisher_node publishes one. STOP_AT_END never is: a goal at the
    end of a loop that has no end would be driven through, not stopped at.
    """

    route = []
    planar = []
    for pose in msg.path.poses:
        position = pose.pose.position
        route.append(
            make_route_point(
                float(position.x),
                float(position.y),
                float(position.z),
                yaw_from_quaternion(pose.pose.orientation),
                corridor_half_width,
            )
        )
        planar.append((float(position.x), float(position.y)))

    goal_mode = int(msg.goal_mode)
    closed_loop = (
        goal_mode != Trajectory.STOP_AT_END
        and len(planar) >= 4
        and math.hypot(planar[-1][0] - planar[0][0], planar[-1][1] - planar[0][1])
        < 1e-3
    )
    return TrackedTrajectory(
        trajectory_id=msg.trajectory_id,
        frame_id=msg.path.header.frame_id,
        goal_mode=goal_mode,
        valid_until_ns=int(msg.valid_until.sec) * 1_000_000_000
        + int(msg.valid_until.nanosec),
        route=route,
        planar=planar,
        target_speeds=[float(value) for value in msg.target_speeds],
        closed_loop=closed_loop,
        ahead_route=unroll_loop(route, 3) if closed_loop else route,
        ahead_planar=unroll_loop(planar, 3) if closed_loop else planar,
        lap_offset=len(route) - 1 if closed_loop else 0,
        ahead_curvatures=(
            unroll_loop([float(value) for value in msg.curvatures], 3)
            if closed_loop
            else [float(value) for value in msg.curvatures]
        ),
    )


def end_of_route_reason(
    trajectory: TrackedTrajectory, remaining_m: float, goal_tolerance_m: float
) -> str | None:
    """Why tracking stops at the end of the route; None when it does not.

    STOP_AT_END reaches a goal and the caller latches it. Any other goal mode on
    an open route stops at the end too - past the last waypoint there is nothing
    mapped to follow, and the lookahead would clamp onto a point the vehicle has
    already driven over - but does not latch, so tracking resumes as soon as the
    vehicle is somewhere else along the route. A closed loop has no end.
    """

    if trajectory.closed_loop or remaining_m > goal_tolerance_m:
        return None
    if trajectory.goal_mode == Trajectory.STOP_AT_END:
        return f"GOAL_REACHED remaining={remaining_m:.2f}m"
    return f"END_OF_ROUTE remaining={remaining_m:.2f}m"


class PurePursuitNode(Node):
    """Follow the published Trajectory; command steering, speed and brake."""

    def __init__(self):
        super().__init__("pure_pursuit_node")

        self.declare_parameter("trajectory_topic", "/planning/trajectory")
        self.declare_parameter("vehicle_state_topic", "/localization/vehicle_state")
        self.declare_parameter("angle_topic", "/target_angle")
        self.declare_parameter("speed_topic", "/target_speed")
        self.declare_parameter("brake_topic", "/brake")
        self.declare_parameter("status_topic", "~/status")
        self.declare_parameter("cross_track_error_topic", "~/cross_track_error")
        self.declare_parameter("lookahead_distance_topic", "~/lookahead_distance")
        self.declare_parameter("lookahead_marker_topic", "~/lookahead_marker")

        # Vehicle geometry. Defaults mirror config/vehicle.yaml so a bare
        # `ros2 run` is still honest about which car it thinks it is driving.
        self.declare_parameter("wheelbase_m", 1.340)
        self.declare_parameter("track_width_m", 1.080)
        self.declare_parameter("min_turning_radius_m", 3.4)
        # 0 means "derive from min_turning_radius_m". Set it only from a measured
        # road-wheel angle; see vehicle.py on why a steering-wheel figure here is
        # a silent, large error.
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

        self.declare_parameter("lookahead_gain_s", 0.6)
        self.declare_parameter("lookahead_min_m", 3.0)
        self.declare_parameter("lookahead_max_m", 8.0)
        # "speed": ld = clamp(lookahead_gain_s * v, min, max) above.
        # "curvature": ld from the tightest route curvature within
        # lookahead_straight_m ahead, speed ignored - lookahead_straight_m at a
        # radius of lookahead_straight_radius_m or more, lookahead_curve_m at
        # lookahead_curve_radius_m or less, linear in curvature between. See
        # pure_pursuit.curvature_lookahead.
        self.declare_parameter("lookahead_mode", "speed")
        self.declare_parameter("lookahead_straight_m", 4.0)
        self.declare_parameter("lookahead_curve_m", 1.0)
        self.declare_parameter("lookahead_straight_radius_m", 50.0)
        self.declare_parameter("lookahead_curve_radius_m", 15.0)

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
            raise RuntimeError(
                f"steering_sign must be +1.0 or -1.0, got {self._steering_sign}; "
                "the steering ratio belongs in steering_command_gain"
            )
        self._lookahead_gain = float(self.get_parameter("lookahead_gain_s").value)
        self._lookahead_min = float(self.get_parameter("lookahead_min_m").value)
        self._lookahead_max = float(self.get_parameter("lookahead_max_m").value)
        self._lookahead_mode = str(self.get_parameter("lookahead_mode").value)
        self._lookahead_straight = float(
            self.get_parameter("lookahead_straight_m").value
        )
        self._lookahead_curve = float(self.get_parameter("lookahead_curve_m").value)
        self._lookahead_straight_radius = float(
            self.get_parameter("lookahead_straight_radius_m").value
        )
        self._lookahead_curve_radius = float(
            self.get_parameter("lookahead_curve_radius_m").value
        )
        if self._lookahead_mode not in ("speed", "curvature"):
            raise ValueError(
                f"lookahead_mode must be 'speed' or 'curvature', got "
                f"'{self._lookahead_mode}'"
            )
        if self._lookahead_mode == "curvature" and not (
            0.0 < self._lookahead_curve <= self._lookahead_straight
            and 0.0 < self._lookahead_curve_radius < self._lookahead_straight_radius
        ):
            raise ValueError(
                "curvature lookahead needs 0 < lookahead_curve_m <= "
                "lookahead_straight_m and 0 < lookahead_curve_radius_m < "
                "lookahead_straight_radius_m"
            )
        if self._lookahead_min > self._lookahead_max:
            raise RuntimeError(
                f"lookahead_min_m ({self._lookahead_min}) exceeds lookahead_max_m "
                f"({self._lookahead_max})"
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
        # The lookahead actually in use. Published rather than only logged
        # because it is derived (gain * speed, clamped) and therefore not
        # readable from any parameter - and because a second, unrelated
        # fixed-distance lookahead exists on /lookahead_marker.
        self._lookahead_pub = self.create_publisher(
            Float32, str(self.get_parameter("lookahead_distance_topic").value), 10
        )
        self._marker_pub = self.create_publisher(
            Marker, str(self.get_parameter("lookahead_marker_topic").value), 10
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

        # Watchdog on the control loop itself, driven by the WALL clock so it
        # still fires when the node clock is the thing that has stopped.
        #
        # The failure it exists for: use_sim_time with nothing advancing
        # /clock - a bag that is not playing, or has finished. Every ROS timer
        # then silently never fires, so the node is alive, its topics exist and
        # `ros2 topic echo` attaches to them, and not one message is ever
        # published. There is nothing in the logs and nothing in `ros2 node
        # list` to say why.
        self._control_cycles = 0
        self._cycles_at_last_check = -1
        self.create_timer(
            float(self.get_parameter("stall_warning_period_sec").value),
            self._warn_if_control_loop_stalled,
            clock=Clock(clock_type=ClockType.SYSTEM_TIME),
        )

        # The speed below which lookahead_gain_s is inert because the floor
        # wins. Reported because tuning the gain under it does nothing at all,
        # and nothing else would say so.
        if self._lookahead_gain > 0.0:
            gain_binds_above = self._lookahead_min / self._lookahead_gain
            gain_note = f", gain binds above {gain_binds_above * 3.6:.1f} km/h"
        else:
            gain_note = ", gain disabled (fixed lookahead)"
        self.get_logger().info(
            f"pure_pursuit_node: wheelbase {self._vehicle.wheelbase_m:.3f} m, steering "
            f"limit {math.degrees(self._vehicle.max_steering_angle_rad):.1f} deg "
            f"(radius {self._vehicle.implied_turning_radius_m:.2f} m), sign "
            f"{self._steering_sign:+.0f}, rate limit "
            f"{math.degrees(self._vehicle.max_steering_rate_rad_s):.1f} deg/s, "
            f"lookahead {self._lookahead_gain:.2f} s clamped to "
            f"{self._lookahead_min:.1f}..{self._lookahead_max:.1f} m{gain_note}, "
            + (
                f"control at {rate:.1f} Hz"
                if self._lookahead_mode == "speed"
                else f"control at {rate:.1f} Hz; lookahead mode CURVATURE (speed "
                f"ignored): {self._lookahead_straight:.1f} m at R >= "
                f"{self._lookahead_straight_radius:.0f} m -> {self._lookahead_curve:.1f} m "
                f"at R <= {self._lookahead_curve_radius:.0f} m"
            )
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
            # Trajectory.msg promises one value per pose. Interpolating a short
            # array against a long path would silently apply the wrong corner's
            # speed for the rest of the route.
            self.get_logger().error(
                f"ignoring trajectory '{msg.trajectory_id}': {count} poses but "
                f"{len(msg.curvatures)} curvatures and {len(msg.target_speeds)} "
                "target speeds"
            )
            return

        if msg.trajectory_id != self._latched_trajectory_id:
            # A genuinely new plan releases a latched goal; a republish of the
            # same one must not, or the vehicle would pull away from a goal it
            # has already reached and stopped at.
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

        using_sim_time = bool(self.get_parameter("use_sim_time").value)
        if using_sim_time:
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
        axle_x, axle_y = rear_axle_position(
            float(pose.position.x),
            float(pose.position.y),
            yaw,
            self._vehicle.rear_axle_offset_m,
        )

        projection = project_onto_route(trajectory.route, axle_x, axle_y)
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

        speed = float(state.odometry.twist.twist.linear.x)
        if self._lookahead_mode == "curvature" and trajectory.ahead_curvatures:
            # The corner is looked for over the STRAIGHT lookahead, not the
            # current one: over a shortened window the vehicle would see less
            # of the bend, lengthen again, and oscillate.
            ahead = trajectory.ahead_of(projection)
            tightest = max_abs_value_ahead(
                trajectory.ahead_planar,
                trajectory.ahead_curvatures,
                ahead.segment_index,
                ahead.fraction,
                self._lookahead_straight,
            )
            distance = curvature_lookahead(
                tightest,
                self._lookahead_straight,
                self._lookahead_curve,
                self._lookahead_straight_radius,
                self._lookahead_curve_radius,
            )
        else:
            distance = lookahead_distance(
                speed, self._lookahead_gain, self._lookahead_min, self._lookahead_max
            )
        target = point_ahead_on_route(
            trajectory.ahead_route, trajectory.ahead_of(projection), distance
        )
        if target is None:
            self._publish_stop("NO_LOOKAHEAD")
            return

        body_x, body_y = to_body_frame(axle_x, axle_y, yaw, target.x, target.y)
        desired = steering_angle(body_x, body_y, self._vehicle.wheelbase_m)
        desired = clamp_steering(desired, self._vehicle.max_steering_angle_rad)
        steering = rate_limited(
            self._last_steering, desired, self._vehicle.max_steering_rate_rad_s, dt
        )
        self._last_steering = steering

        target_speed = value_at_route_position(
            trajectory.target_speeds, projection.segment_index, projection.fraction
        )
        speed_cap = self._start_ramp.cap(speed, axle_x, axle_y)
        ramping = speed_cap < target_speed
        target_speed = min(target_speed, speed_cap)

        command = self._actuator.command(steering, self._steering_sign)
        self._angle_pub.publish(Float32(data=float(command)))
        self._speed_pub.publish(Float32(data=speed_command_kmh(target_speed)))
        self._brake_pub.publish(Bool(data=False))
        self._error_pub.publish(Float32(data=float(projection.signed_offset)))
        self._lookahead_pub.publish(Float32(data=float(distance)))
        self._publish_lookahead_marker(trajectory.frame_id, target)
        self._publish_status(
            f"TRACKING wp={projection.route_position:.1f} "
            f"cte={projection.signed_offset:+.2f}m ld={distance:.2f}m "
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
        axle_x, axle_y = rear_axle_position(
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
            self._vehicle.rear_axle_offset_m,
        )
        projection = project_onto_route(trajectory.route, axle_x, axle_y)
        if projection is None or not back_at_route_start(
            trajectory.planar,
            projection.segment_index,
            projection.fraction,
            axle_x,
            axle_y,
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
            # The planner is gone or wedged. Following its last plan open-loop is
            # exactly the situation valid_until exists to prevent.
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
        """Command a stop, holding the last steering angle.

        Held rather than centred: these are the cases where the controller does
        not trust its inputs, and straightening the wheels mid-corner is its own
        kind of manoeuvre. Braking on a held angle is the predictable one.
        """

        self._start_ramp.reset()
        self._angle_pub.publish(
            Float32(data=float(self._actuator.command(self._last_steering, self._steering_sign)))
        )
        self._speed_pub.publish(Float32(data=0.0))
        self._brake_pub.publish(Bool(data=True))
        self._clear_lookahead_marker()
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

        self._clear_lookahead_marker()
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
        self._clear_lookahead_marker()
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
        head = status.split(" ", 1)[0]
        if head != self._last_status:
            self._last_status = head
            self.get_logger().info(status)

    def _publish_lookahead_marker(self, frame_id: str, target) -> None:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit_lookahead"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(target.x)
        marker.pose.position.y = float(target.y)
        marker.pose.position.z = float(target.z)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.6
        # Cyan, not red. localization_monitor_node publishes its own red
        # lookahead sphere on /lookahead_marker at a FIXED lookahead_distance_m
        # (5.0 m) for display, and that is the one mapping.rviz shows. Two red
        # spheres a few metres apart, one of which actually steers the vehicle,
        # is a debugging trap: this is the one being steered at.
        marker.color.r = 0.0
        marker.color.g = 0.9
        marker.color.b = 1.0
        marker.color.a = 1.0
        self._marker_pub.publish(marker)

    def _clear_lookahead_marker(self) -> None:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit_lookahead"
        marker.id = 0
        marker.action = Marker.DELETE
        self._marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
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
