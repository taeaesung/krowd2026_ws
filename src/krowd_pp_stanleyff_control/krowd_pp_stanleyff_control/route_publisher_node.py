"""Publish a mapped route as the live Trajectory that control follows.

Carried over from krowd_stanley_control (its speed profile is the one this
controller adopts). This is the Planning side of the krowd_control_interfaces
contract: waypoints.csv is a globally referenced FILE (UTM, x/y/yaw, one
sample per metre, no speed and no curvature) and Trajectory is a live MESSAGE
in the localization frame carrying both. Somebody has to be the bridge, and
it is deliberately not the controller - a controller that reads files decides
for itself where it is going.

The route is republished periodically rather than latched once, so `valid_until`
stays ahead of now. That gives the controller a positive liveness signal: if
this node dies, its trajectory expires a second later and the controller stops
instead of following a plan nobody is maintaining any more.

The trajectory id is `<run_name>:<N>wp` (`:loop` appended for a closed loop):
the run name selects the controller's map profile and N must equal the
profile's expected waypoint count, so the waypoint filters are OFF by default
here (`max_route_reversal_deg: 0`, `min_waypoint_spacing_m: 0`) - a mission
table indexes the CSV rows one to one.
"""

from dataclasses import replace
from pathlib import Path
import math

import rclpy
import tf2_ros
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PoseStamped
from krowd_control_interfaces.msg import Trajectory
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException

from mola_slam_mapping.route_io import load_route_files
from mola_slam_mapping.route_visualization import transform_route

from .route_geometry import (
    forward_progress_indices,
    loop_closure,
    path_curvatures,
    segment_lengths,
    unroll_loop,
)
from .speed_profile import SpeedLimits, curvature_only_profile, speed_profile, step_down_decelerations


GOAL_MODES = {
    "STOP_AT_END": Trajectory.STOP_AT_END,
    "PASS_THROUGH": Trajectory.PASS_THROUGH,
    "FOLLOW_UNTIL_REPLACED": Trajectory.FOLLOW_UNTIL_REPLACED,
}


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation about z."""

    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


class RoutePublisherNode(Node):
    """Loads waypoints.csv, resolves it into the localization frame, and
    publishes it as a Trajectory with curvature and a speed profile."""

    def __init__(self):
        super().__init__("route_publisher_node")

        # Paths stay parameters with no built-in absolute default: this package
        # is meant to move between machines, and a hard-coded /home/... path is
        # the first thing that breaks when it does.
        self.declare_parameter("map_dir", "")
        self.declare_parameter("waypoints_csv", "")
        self.declare_parameter("corridor_csv", "")
        # With no map_dir and no waypoints_csv, follow the run localization
        # loaded: it announces it here, latched (see selected_map_node).
        self.declare_parameter("map_dir_topic", "/localization/map_dir")

        self.declare_parameter("utm_frame", "utm")
        self.declare_parameter("output_frame", "map")
        # Launch-specific escape hatch kept separate from output_frame so a
        # node-specific tuning YAML cannot overwrite it during ROS parameter
        # merging. Used by the closed-loop simulator to keep both route and
        # vehicle in stable UTM coordinates while map<->utm initializes.
        self.declare_parameter("output_frame_override", "")
        self.declare_parameter("trajectory_topic", "/planning/trajectory")
        self.declare_parameter("corridor_half_width_m", 3.0)

        self.declare_parameter("goal_mode", "STOP_AT_END")
        # Outside STOP_AT_END a route that returns to its start is published as
        # a closed loop and driven round indefinitely (see loop_closure). 0
        # disables closing, so every route ends where it was recorded to.
        self.declare_parameter("loop_closure_max_gap_m", 0.0)
        self.declare_parameter("loop_closure_search_m", 30.0)
        self.declare_parameter("loop_closure_max_turn_deg", 60.0)
        # Speeds are configured in km/h, the unit the operator and the vehicle
        # spec both think in. They are converted to SI exactly once, where they
        # are read below; SpeedLimits, the profile and the Trajectory message
        # are all m/s, as nav_msgs and the unit tests require.
        self.declare_parameter("max_speed_kmph", 25.0)
        self.declare_parameter("max_lateral_accel_mps2", 2.0)
        self.declare_parameter("max_accel_mps2", 3.0)
        self.declare_parameter("max_decel_mps2", 0.5)
        self.declare_parameter("min_speed_kmph", 12.0)
        # Optional launch-time ceiling for a cautious real-vehicle trial. This
        # caps the whole configured profile without duplicating its tuning.
        self.declare_parameter("speed_limit_override_kmph", 0.0)
        # Curve cap: at most curve_max_speed_kmph wherever the turning radius is
        # below curve_radius_m. Both 0 disables it.
        self.declare_parameter("curve_max_speed_kmph", 12.0)
        self.declare_parameter("curve_radius_m", 30.0)
        # Graded corner speed: curve_max_speed_kmph at this radius or tighter,
        # rising linearly to max_speed_kmph at curve_radius_m. 0 = flat cap.
        self.declare_parameter("tight_curve_radius_m", 0.0)
        # Name of the speed mode the launch applied (config/speed_modes/), for the log.
        self.declare_parameter("speed_mode", "")
        # Be at a corner's speed this many metres before it (see
        # SpeedLimits.curve_lead_m). 0 disables.
        self.declare_parameter("curve_speed_lead_m", 3.0)
        # Release the last metres of a corner to straight speed (see
        # SpeedLimits.curve_exit_lead_m). 0 disables.
        self.declare_parameter("curve_exit_lead_m", 10.0)
        self.declare_parameter("curvature_smoothing_window", 5)
        # true: each waypoint's speed comes from its own curvature only (no
        # braking zone before a corner, no ramps). false: the three-pass
        # profile with its decelerations turned into steps.
        self.declare_parameter("curvature_only", True)
        # curvature_only: curve runs shorter than this are noise, not corners. 0 = off.
        self.declare_parameter("min_curve_length_m", 0.0)
        self.declare_parameter("corner_merge_gap_m", 0.0)
        # 0 / 0 = no waypoint filtering: the mission tables index CSV rows.
        self.declare_parameter("max_route_reversal_deg", 0.0)
        self.declare_parameter("min_waypoint_spacing_m", 0.0)

        self.declare_parameter("publish_period_sec", 0.5)
        self.declare_parameter("valid_for_sec", 2.0)

        self._utm_frame = str(self.get_parameter("utm_frame").value)
        output_override = str(
            self.get_parameter("output_frame_override").value
        ).strip()
        self._output_frame = output_override or str(
            self.get_parameter("output_frame").value
        )
        self._corridor_half_width = float(
            self.get_parameter("corridor_half_width_m").value
        )
        self._smoothing_window = int(
            self.get_parameter("curvature_smoothing_window").value
        )
        self._curvature_only = bool(self.get_parameter("curvature_only").value)
        self._min_curve_length = float(self.get_parameter("min_curve_length_m").value)
        self._corner_merge_gap = float(self.get_parameter("corner_merge_gap_m").value)
        self._valid_for = float(self.get_parameter("valid_for_sec").value)
        publish_period = float(self.get_parameter("publish_period_sec").value)
        if self._valid_for <= publish_period:
            raise RuntimeError(
                f"valid_for_sec ({self._valid_for}) must exceed publish_period_sec "
                f"({publish_period}), or every trajectory expires before the next "
                "one is published and the controller never leaves fail-safe"
            )

        goal_mode_name = str(self.get_parameter("goal_mode").value).upper()
        if goal_mode_name not in GOAL_MODES:
            raise RuntimeError(
                f"goal_mode '{goal_mode_name}' is not one of {sorted(GOAL_MODES)}"
            )
        self._goal_mode_name = goal_mode_name
        self._goal_mode = GOAL_MODES[goal_mode_name]

        self._limits = SpeedLimits(
            max_speed_mps=float(self.get_parameter("max_speed_kmph").value) / 3.6,
            max_lateral_accel_mps2=float(
                self.get_parameter("max_lateral_accel_mps2").value
            ),
            max_accel_mps2=float(self.get_parameter("max_accel_mps2").value),
            max_decel_mps2=float(self.get_parameter("max_decel_mps2").value),
            min_speed_mps=float(self.get_parameter("min_speed_kmph").value) / 3.6,
            curve_max_speed_mps=float(
                self.get_parameter("curve_max_speed_kmph").value
            ) / 3.6,
            curve_radius_m=float(self.get_parameter("curve_radius_m").value),
            tight_curve_radius_m=float(self.get_parameter("tight_curve_radius_m").value),
            curve_lead_m=float(self.get_parameter("curve_speed_lead_m").value),
            curve_exit_lead_m=float(self.get_parameter("curve_exit_lead_m").value),
        )
        speed_override = (
            float(self.get_parameter("speed_limit_override_kmph").value) / 3.6
        )
        if speed_override < 0.0:
            raise RuntimeError("speed_limit_override_kmph must not be negative")
        if speed_override > 0.0:
            self._limits = replace(
                self._limits,
                max_speed_mps=min(self._limits.max_speed_mps, speed_override),
                min_speed_mps=min(self._limits.min_speed_mps, speed_override),
                curve_max_speed_mps=(
                    min(self._limits.curve_max_speed_mps, speed_override)
                    if self._limits.curve_cap_enabled
                    else 0.0
                ),
            )
        self._limits.validate()

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Transient-local so a controller that starts later still receives the
        # route immediately instead of waiting a whole publish period.
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._trajectory_pub = self.create_publisher(
            Trajectory, str(self.get_parameter("trajectory_topic").value), latched
        )

        self._utm_zone = None
        self._route_utm = None
        self._closed_loop = False
        self._loaded_map_dir = None
        self._trajectory_id = None
        self._published_once = False
        self._warned_missing_transform = False
        self._cached_geometry = None
        self._cached_transform_signature = None

        self.create_timer(publish_period, self._publish_trajectory)

        # An explicitly named route wins. Without one, follow whichever run
        # localization loaded - naming the directory in both launches is how the
        # two end up on different maps.
        if str(self.get_parameter("waypoints_csv").value) or str(
            self.get_parameter("map_dir").value
        ):
            self._load_route(str(self.get_parameter("map_dir").value))
        else:
            topic = str(self.get_parameter("map_dir_topic").value)
            self.create_subscription(String, topic, self._on_map_dir, latched)
            self.get_logger().info(
                "route_publisher_node: no map_dir given; following the "
                f"localization map announced on '{topic}'"
            )

    def _on_map_dir(self, message: String) -> None:
        map_dir = str(message.data).strip()
        if not map_dir:
            self.get_logger().error(
                "ignoring an empty map directory from localization"
            )
            return
        if map_dir == self._loaded_map_dir:
            return
        try:
            self._load_route(map_dir)
        except (OSError, ValueError, KeyError) as error:
            self.get_logger().error(
                f"cannot use the localization map '{map_dir}': {error}"
            )

    def _load_route(self, map_dir: str) -> None:
        """Read one run's route and republish from it; the trajectory id names it."""

        waypoints_path, corridor_path = self._resolve_route_paths(map_dir)
        loaded = load_route_files(
            waypoints_path, corridor_path, self._corridor_half_width
        )
        self._utm_zone = loaded.utm_zone
        self._route_utm, self._closed_loop = self._close_loop(
            self._drop_backtracking(loaded.points)
        )
        self._loaded_map_dir = map_dir
        self._cached_geometry = None
        self._cached_transform_signature = None
        self._published_once = False

        # Resolved, because map_dir is often the `latest` symlink: the id should
        # name the run that was actually loaded, so a controller log can be
        # traced back to one map.
        run_name = waypoints_path.resolve().parent.name
        self._trajectory_id = f"{run_name}:{len(self._route_utm)}wp" + (
            ":loop" if self._closed_loop else ""
        )

        self.get_logger().info(
            f"route_publisher_node: {len(self._route_utm)} waypoints from "
            f"'{waypoints_path}' (UTM zone '{self._utm_zone or 'unset'}', corridor "
            f"{'from file' if loaded.used_saved_corridor else 'synthesized'}), "
            f"resolving '{self._utm_frame}' -> '{self._output_frame}', goal mode "
            f"{self._goal_mode_name}"
            f"{' (closed loop)' if self._closed_loop else ''}, speed mode "
            f"'{self.get_parameter('speed_mode').value or 'none'}', speed ceiling "
            f"{self._limits.max_speed_mps * 3.6:.1f} km/h"
            + self._curve_summary()
            + f", grip {self._limits.max_lateral_accel_mps2:.1f} m/s^2"
            + f", floor {self._limits.min_speed_mps * 3.6:.1f} km/h"
        )

    def _curve_summary(self) -> str:
        limits = self._limits
        if not limits.curve_cap_enabled:
            return ", no curve cap"
        if limits.tight_curve_radius_m > 0.0:
            return (
                f", corners {limits.curve_max_speed_mps * 3.6:.1f} km/h at R <= "
                f"{limits.tight_curve_radius_m:.0f} m rising to "
                f"{limits.max_speed_mps * 3.6:.1f} at R {limits.curve_radius_m:.0f} m"
            )
        return (
            f", curves (R < {limits.curve_radius_m:.0f} m) at most "
            f"{limits.curve_max_speed_mps * 3.6:.1f} km/h"
        )

    def _drop_backtracking(self, route):
        """Remove waypoints that reverse the direction of travel (OFF by default).

        Loudly, when enabled: dropping recorded route data changes the waypoint
        numbering the mission tables rely on. The reported indices are the ones
        to go and look at in waypoints.csv.
        """

        max_reversal = float(self.get_parameter("max_route_reversal_deg").value)
        min_spacing = float(self.get_parameter("min_waypoint_spacing_m").value)
        if max_reversal <= 0.0 and min_spacing <= 0.0:
            return route
        kept = forward_progress_indices(
            [(point.x, point.y) for point in route], max_reversal, min_spacing
        )
        if len(kept) == len(route):
            return route

        dropped = sorted(set(range(len(route))) - set(kept))
        preview = ", ".join(str(index) for index in dropped[:10])
        if len(dropped) > 10:
            preview += ", ..."
        self.get_logger().warn(
            f"dropped {len(dropped)} of {len(route)} waypoints that double back on "
            f"the route (indices {preview}). The trajectory's waypoint count no longer "
            "matches the CSV: a map profile with a mission table will refuse it."
        )
        return [route[index] for index in kept]

    def _close_loop(self, route):
        """The route as a closed loop (last point == first) when it is one.

        Only outside STOP_AT_END: a goal at the end and a loop with no end are
        contradictory. A route that is not a loop is returned unchanged and the
        controller stops at its end.
        """

        if self._goal_mode == Trajectory.STOP_AT_END:
            return route, False
        max_gap = float(self.get_parameter("loop_closure_max_gap_m").value)
        closure = loop_closure(
            [(point.x, point.y) for point in route],
            max_gap,
            float(self.get_parameter("loop_closure_search_m").value),
            float(self.get_parameter("loop_closure_max_turn_deg").value),
        )
        if closure is None:
            if max_gap > 0.0:
                self.get_logger().warn(
                    f"route is not a loop (no join of at most {max_gap:.1f} m between "
                    "its start and its end): tracking will stop at its end"
                )
            return route, False
        head, tail = closure
        join = math.hypot(route[head].x - route[tail].x, route[head].y - route[tail].y)
        self.get_logger().info(
            f"route closed into a loop: waypoints {head}..{tail} of "
            f"{len(route)}, joined back over {join:.2f} m"
        )
        loop = list(route[head : tail + 1])
        loop.append(route[head])
        return loop, True

    def _resolve_route_paths(self, map_dir: str) -> tuple[Path, Path | None]:
        """waypoints_csv/corridor_csv win; otherwise derive both from map_dir."""

        waypoints = str(self.get_parameter("waypoints_csv").value)
        corridor = str(self.get_parameter("corridor_csv").value)

        if waypoints:
            waypoints_path = Path(waypoints).expanduser()
        elif map_dir:
            waypoints_path = Path(map_dir).expanduser() / "waypoints.csv"
        else:
            raise RuntimeError(
                "route_publisher_node needs 'map_dir' (a mapping run directory) "
                "or an explicit 'waypoints_csv'"
            )

        if corridor:
            corridor_path = Path(corridor).expanduser()
        else:
            # Same directory as the waypoints: waypoint_saver_node writes the two
            # together and they are joined by idx, so a corridor from elsewhere
            # would not match.
            candidate = waypoints_path.parent / "corridor.csv"
            corridor_path = candidate if candidate.exists() else None
        return waypoints_path, corridor_path

    def _transform_signature(self, transform) -> tuple:
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            translation.x,
            translation.y,
            translation.z,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

    def _route_in_output_frame(self):
        """The route resolved into the output frame, or None while TF is missing.

        The geometry is recomputed only when the transform actually changes.
        `utm <- map` is a static geo-reference in normal operation, so this is
        once; but gnss_imu_pose_node can correct it during a run, and a corrected
        geo-reference must move the whole route rather than leave the profile
        computed against the old one.
        """

        if self._output_frame == self._utm_frame:
            if self._cached_geometry is None:
                self._cached_geometry = self._build_geometry(self._route_utm)
            return self._cached_geometry

        try:
            transform = self._tf_buffer.lookup_transform(
                self._output_frame,
                self._utm_frame,
                Time(),
                timeout=Duration(seconds=0.0),
            )
        except TransformException as error:
            if not self._warned_missing_transform:
                self._warned_missing_transform = True
                self.get_logger().warn(
                    f"no '{self._output_frame}' <- '{self._utm_frame}' transform yet, "
                    f"so no trajectory is being published ({error}). This is "
                    "gnss_imu_pose_node's static geo-reference; it needs both GNSS "
                    "topics and a heading commit before it appears."
                )
            return None

        signature = self._transform_signature(transform)
        if signature != self._cached_transform_signature:
            self._cached_transform_signature = signature
            self._cached_geometry = self._build_geometry(
                transform_route(self._route_utm, transform)
            )
            if self._warned_missing_transform:
                self.get_logger().info(
                    f"'{self._output_frame}' <- '{self._utm_frame}' resolved; "
                    "publishing the trajectory"
                )
                self._warned_missing_transform = False
        return self._cached_geometry

    def _build_geometry(self, route):
        """Curvature and speed profile for one placement of the route."""

        planar = [(point.x, point.y) for point in route]
        if self._closed_loop:
            # Three laps and keep the middle one, so the curvature smoothing and
            # both speed-profile passes see across the seam.
            count = len(planar)
            laps = unroll_loop(planar, 3)
            curvatures = path_curvatures(laps, self._smoothing_window)
            speeds = self._speeds(segment_lengths(laps), curvatures, stop_at_end=False)
            middle = slice(count - 1, 2 * count - 1)
            return route, curvatures[middle], speeds[middle]

        curvatures = path_curvatures(planar, self._smoothing_window)
        # An open route ends in a stop whatever the goal mode: past its last
        # waypoint there is nothing to follow.
        return route, curvatures, self._speeds(segment_lengths(planar), curvatures, stop_at_end=True)

    def _speeds(self, lengths, curvatures, stop_at_end: bool):
        if self._curvature_only:
            return curvature_only_profile(
                lengths, curvatures, self._limits, stop_at_end, self._min_curve_length, self._corner_merge_gap
            )
        speeds = speed_profile(lengths, curvatures, self._limits, stop_at_end)
        # Commanded profile: each braking ramp becomes one step (speed_profile.py).
        return step_down_decelerations(speeds, self._limits.min_speed_mps)

    def _publish_trajectory(self) -> None:
        if self._route_utm is None:
            self.get_logger().info(
                "no route yet: waiting for the localization map on "
                f"'{self.get_parameter('map_dir_topic').value}'",
                throttle_duration_sec=10.0,
            )
            return
        geometry = self._route_in_output_frame()
        if geometry is None:
            return
        route, curvatures, speeds = geometry

        now = self.get_clock().now()
        stamp = now.to_msg()

        path = PathMsg()
        path.header.stamp = stamp
        path.header.frame_id = self._output_frame
        for point in route:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self._output_frame
            pose.pose.position.x = float(point.x)
            pose.pose.position.y = float(point.y)
            pose.pose.position.z = float(point.z)
            qx, qy, qz, qw = quaternion_from_yaw(point.yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)

        message = Trajectory()
        message.trajectory_id = self._trajectory_id
        message.generation_time = stamp
        message.valid_until = self._valid_until(now)
        message.goal_mode = self._goal_mode
        message.path = path
        message.curvatures = [float(value) for value in curvatures]
        message.target_speeds = [float(value) for value in speeds]
        self._trajectory_pub.publish(message)

        if not self._published_once:
            self._published_once = True
            peak = max((abs(value) for value in curvatures), default=0.0)
            if peak > 1e-6:
                tightest = f"tightest radius {1.0 / peak:.1f} m"
            else:
                tightest = "straight route"
            self.get_logger().info(
                f"publishing trajectory '{self._trajectory_id}': {len(route)} poses "
                f"in '{self._output_frame}', {tightest}, speed "
                f"{min(speeds) * 3.6:.1f}..{max(speeds) * 3.6:.1f} km/h"
            )

    def _valid_until(self, now: Time) -> TimeMsg:
        return (now + Duration(seconds=self._valid_for)).to_msg()


def main(args=None):
    rclpy.init(args=args)
    node = RoutePublisherNode()
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
