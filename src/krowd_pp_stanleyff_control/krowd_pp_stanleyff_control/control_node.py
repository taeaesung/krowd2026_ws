"""KROWD PP + Stanley blended vehicle controller node (with curvature feedforward).

Single owner of /target_angle, /target_speed and /brake (ros_can_bridge).

    Global Path (/planning/trajectory)         --> two monotonic projections:
                                                   rear axle (Pure Pursuit, curvature -> blend weight)
                                                   front axle (Stanley heading / cross-track terms)
    ESKF pose (/ekf/odometry + TF, fallbacks)  --> vehicle state (speed, yaw rate)
    map profile (config/maps/<run>/)           --> missions, segments, lane boundaries; or route_only
    lane boundaries + route corridor           --> virtual-wall supervisor (per-point fallback)
    /glane3d/stopline_markers                  --> live stop-line distance (optional, TTL)
    /traffic_light/detections_json (+Bools)    --> per-direction permit with 0.25 s TTL
    /ekf/heading_verified (+source)            --> start gate / HEADING_ALIGN until it locks
    /integration_control/avoiding              --> 0924 Stanley throughout an avoidance zone;
                                                 --> map route baseline until a cone path is available
                                               --> command mux --> CAN bridge (50 Hz)

Steering:  delta = (1-w) PP + w Stanley + yaw-rate damping, where Stanley carries
           feedforward_gain * atan(L kappa) (the difference from the sibling package), slew-limited at the
           measured road-wheel rate, then supervised (corridor.py).
           In the cone_avoidance_1005 integrated launch only, avoiding=true
           selects the latency-compensated cone_avoidance_0924_3 Stanley law
           inside this same node; the final actuator publishers do not change.
Speed:     route speed profile (route_publisher) -> start ramp -> speed trim,
           each profile drop (corner) commanded as one step early enough to
           COAST down to it from the actual speed (coast_to_corner); the trim
           is off wherever the profile is below its straight speed, and no
           acceleration while the heading is off the route (align_hold),
           capped by the mission / segment / signal caps; stops are owned by
           the braking model (stoppable distance, comfort stop, brake latch,
           brake-release hold).
Stops:     only stop TARGETS (signal missions, route end, the crawl floor) and
           four input failures: NO_POSE (no fresh pose), POSE_FAR (pose more
           than pure_pursuit.pose_far_distance_m off the route - a wrong pose,
           not a tracking error), WRONG_DIRECTION (heading error above
           pure_pursuit.max_heading_error_deg) and the ESKF heading turning
           unverified AFTER it had locked. Everything else keeps driving
           (2026-10-01 decision): an expired trajectory is kept, a pose that
           left the projection window is re-acquired over the whole route, and
           the corridor supervisor never brakes - it commands speed 0 WITHOUT
           the brake while every candidate hits a wall (CORRIDOR_HOLD; the
           crawl instead when standing still) and the crawl while the footprint
           is wholly outside the lane (LANE_OUTSIDE), steering back all the
           while (corridor.py). Which wall segments are
           the vehicle's is fixed once per trajectory (bind_route), so a route
           crossing or running beside itself never lends its other pass's wall.

There is no arm step (removed 2026-10-03): the node publishes to the actuator
topics from its first cycle - a FAILSAFE_STOP (speed 0, brake) with reason
NOT_READY while any BLOCKING configuration value is null (config.py), while
the map profile is unusable, while the heading is unverified, or while the
live inputs are not healthy - and starts tracking by itself, at the current
route projection, the first cycle nothing blocks it. With heading.align_drive
an unverified heading starts instead, into HEADING_ALIGN: road wheel 0 at
heading.align_speed_kmh and no other check until the ESKF heading locks, then
the route is tracked (the course starts on a straight; 2026-10-02 decision).
There is no software E-stop: the physical ASMS / E-stop stops the vehicle.
`~/reload_config` and `~/reset_missions` work at standstill and restart
tracking. Which map is driven follows /localization/map_dir (or the
trajectory id) - see map_profile.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
import tf2_ros
import yaml
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PointStamped
from krowd_control_interfaces.msg import Trajectory
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from ros_can_bridge_msgs.msg import EncoderAngleStamped, WheelSpeedStamped

from . import missions as M
from .blended_steering import BlendParams, compute_steering
from .braking import BrakingModel
from .config import (
    CONTROL_CONFIG,
    PACKAGE_NAME,
    default_config_dir,
    default_maps_dir,
    front_offset_m,
    get,
    is_number,
    load_control_config,
    readiness,
    rear_offset_m,
)
from .corridor import (
    CorridorSupervisor,
    StaticCorridor,
    SupervisorParams,
    load_boundaries,
    route_boundaries,
    validate_boundaries,
)
from .geometry import footprint_corners, to_body_frame, wrap_angle, yaw_from_quaternion
from .map_profile import MapProfile, find_profile, load_profile, route_only_profile, run_name_of
from .pure_pursuit import curvature_lookahead, pure_pursuit_steering, rate_limited, time_floored_lookahead
from .route import RouteModel
from .segments import SegmentParams, SegmentTable, load_segments, resolve_segments
from .signal_permit import SignalPermitTracker
from .speed_planner import (
    align_hold_cap,
    anticipatory_step,
    coast_aim_points,
    coast_step_cap,
    comfort_speed,
    profile_drops,
    step_cap,
)
from .speed_trim import SpeedTrim
from .stanley import front_axle_position, predict_front_axle_pose, steering_angle
from .start_ramp import StartRamp
from .zone_brake import ZoneEntryBrake, load_zone_ranges, stop_needed, zone_at
from .steering_calibration import SteeringCalibration
from .stopline import StoplineEstimator

KMH = 3.6
NODE_NAME = "pp_stanleyff_control_node"
# The Global Path is fixed, so an expired /planning/trajectory is kept; this
# long past its valid_until the node only warns that the publisher is quiet.
TRAJECTORY_STALE_WARN_S = 5.0
# Candidate rollouts drawn in RViz stay this long, so a one-cycle search shows.
CANDIDATE_MARKER_LIFETIME_S = 1.0
# Speed caps on top of which the speed trim may act. Every other cap (start
# ramp, stoppable distance, comfort stop, commit approach, unverified table)
# freezes the trim and is commanded as is.
TRIM_ACTIVE_PREFIXES = ("route_profile", "cruise", "max", "segment:", "mission:", "anticipate_next_zone")


@dataclass
class PoseSample:
    x: float
    y: float
    yaw: float
    speed_mps: float
    yaw_rate: float
    stamp_ns: int
    source: str
    age_s: float


@dataclass
class CycleResult:
    ok_to_drive: bool
    brake: bool
    speed_mps: float
    steering_rw: float
    command_rad: float
    state: str
    reason: str
    info: dict = field(default_factory=dict)


class StoppedTracker:
    def __init__(self, threshold_mps: float, confirm_s: float):
        self.threshold = threshold_mps
        self.confirm_s = confirm_s
        self._since_ns: int | None = None

    def update(self, now_ns: int, speeds: list[float]) -> bool:
        slow = all(abs(v) <= self.threshold for v in speeds)
        if not slow:
            self._since_ns = None
            return False
        if self._since_ns is None:
            self._since_ns = now_ns
        return (now_ns - self._since_ns) * 1e-9 >= self.confirm_s


class ControlNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        cfg_dir = default_config_dir()
        self.declare_parameter("control_config", os.path.join(cfg_dir, CONTROL_CONFIG))
        # Optional second YAML merged over control_config (the simulator uses it).
        self.declare_parameter("control_overrides", "")
        # config/speed_modes/<mode>.yaml chosen by the launch (speed_mode:=low|normal|high);
        # its `control` section is merged between control_config and control_overrides.
        self.declare_parameter("speed_mode_config", "")
        self.declare_parameter("maps_dir", default_maps_dir())
        # Explicit profile directory; "" = pick config/maps/<run> for the run
        # announced on /localization/map_dir (or named by the trajectory id).
        self.declare_parameter("map_profile", "")
        # Explicit localization map directory (waypoints.csv lives there); ""
        # = the /localization/map_dir topic.
        self.declare_parameter("map_dir", "")
        self.declare_parameter("force_route_only", False)
        self.declare_parameter("marker_period_s", 0.2)

        self.paths = {
            "control": str(self.get_parameter("control_config").value),
            "overrides": str(self.get_parameter("control_overrides").value),
            "speed_mode": str(self.get_parameter("speed_mode_config").value).strip(),
            "maps": str(self.get_parameter("maps_dir").value) or default_maps_dir(),
        }
        self.speed_mode = os.path.splitext(os.path.basename(self.paths["speed_mode"]))[0] if self.paths["speed_mode"] else "none"
        self.force_route_only = bool(self.get_parameter("force_route_only").value)
        self.explicit_profile = str(self.get_parameter("map_profile").value).strip()
        self.explicit_map_dir = str(self.get_parameter("map_dir").value).strip()

        # ---- runtime state (before configuration so reload can reuse it) ------
        self.route: RouteModel | None = None
        self.route_problem = "NO_TRAJECTORY"
        self.last_trajectory_msg = None
        self.avoidance_route: RouteModel | None = None
        self.route_s: float | None = None
        self.front_s: float | None = None
        self.avoidance_front_s: float | None = None
        self.steering_source = "global"
        self.avoidance_mode_requested = False
        self.avoidance_mode_rx_ns: int | None = None
        self.avoidance_mode_active = False
        # True from the start of tracking until the route has been acquired
        # once: only then may the vehicle
        # drive TO the route with route_join. Leaving it afterwards is tracked
        # back by the normal controller; only pose_far_distance_m stops.
        self.join_allowed = True
        self.join_missions_pending = False
        self.pose_msgs: dict[str, tuple[Odometry, int]] = {}
        self.wheel: tuple[float, int] | None = None
        self.encoder: tuple[float, int] | None = None
        self.wheel_ratio = deque(maxlen=200)
        # False until the inputs are ready (a NOT_READY stop is published until
        # then); set by _start_tracking, cleared by _stop_tracking.
        self.tracking_started = False
        self.last_delta = 0.0
        self.last_speed_cmd = 0.0
        self.last_cmd_angle = 0.0
        self.last_status_head = ""
        self.last_control_ns: int | None = None
        self.control_cycles = 0
        self.cycles_at_check = -1
        self.cycle_time_ms = 0.0
        self.missions: M.MissionManager | None = None
        self.segments: SegmentTable | None = None
        self.mission_warnings: list[str] = []
        self.resolved_missions: list = []
        self.marker_next_ns = 0
        self._brake_latched_for: str | None = None
        self._braking_last = False
        self._brake_release_until_ns: int | None = None
        self._corridor_crawl = False  # CORRIDOR_HOLD reached a standstill: crawl until a candidate clears
        # Route-profile drops (corners) ahead: (s, speed) of each step down, the
        # profile's straight speed, and the drops whose coast-down step has begun.
        self._route_drops: list[tuple[float, float]] = []
        self._route_vmax = 0.0
        self._coast_latched: set = set()
        self._align_hold: float | None = None  # speed held while the heading is off the route
        self._mismatch_run = 0
        self._agree_run = 0
        self._stopline_mismatch = False
        self.static_corridor: StaticCorridor | None = None
        self.fallback_corridor: StaticCorridor | None = None
        self.supervisor: CorridorSupervisor | None = None
        self.heading_verified: bool | None = None
        self.heading_verified_ns: int | None = None
        self.heading_source: str | None = None
        self._heading_aligning = False  # started before the ESKF heading locked: straight ahead until it does
        self.profile: MapProfile | None = None
        self.mission_cfg: dict = route_only_profile("", None).mission_config()
        self.segments_cfg: dict = {"segments": {"list": []}}

        self._load_control()
        self._select_initial_profile()
        self._configure_models()

        # ---- ROS interfaces (created once; topic names need a restart) -------
        topics = self.cfg["topics"]
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Trajectory, topics["trajectory"], self._on_trajectory, latched)
        self.create_subscription(Trajectory, topics.get("avoidance_trajectory", "/planning/avoidance_trajectory"), self._on_avoidance, 10)
        self.pose_order = [str(t) for t in topics["pose_sources"]]
        for topic in self.pose_order:
            self.create_subscription(Odometry, topic, lambda msg, t=topic: self._on_pose(msg, t), qos_profile_sensor_data)
        self.create_subscription(WheelSpeedStamped, topics["wheel_sensor"], self._on_wheel, qos_profile_sensor_data)
        self.create_subscription(EncoderAngleStamped, topics["encoder"], self._on_encoder, qos_profile_sensor_data)
        self.create_subscription(MarkerArray, topics["stopline_markers"], self._on_stopline, 5)
        self.create_subscription(String, topics["traffic_json"], self._on_traffic_json, 10)
        self.create_subscription(Bool, topics["go_straight_confirmed"], lambda m: self._on_go_bool("straight", m), 10)
        self.create_subscription(Bool, topics["go_left_confirmed"], lambda m: self._on_go_bool("left", m), 10)
        self.create_subscription(Bool, self.avoidance_mode_topic, self._on_avoidance_mode, 10)
        self.create_subscription(
            Bool, self.zone_brake_topic,
            lambda m: (self.zone_brake.observe(bool(m.data), self.get_clock().now().nanoseconds)
                       if self.zone_brake_ranges is None else None), 10,
        )
        heading = self.cfg.get("heading") or {}
        self.create_subscription(Bool, str(heading.get("verified_topic", "/ekf/heading_verified")), self._on_heading_verified, latched)
        self.create_subscription(String, str(heading.get("source_topic", "/ekf/heading_source")), self._on_heading_source, latched)
        if not self.explicit_profile and not self.explicit_map_dir:
            maps = self.cfg.get("maps") or {}
            self.create_subscription(String, str(maps.get("map_dir_topic", "/localization/map_dir")), self._on_map_dir, latched)

        self.pub_angle = self.create_publisher(Float32, topics["target_angle"], 10)
        self.pub_speed = self.create_publisher(Float32, topics["target_speed"], 10)
        self.pub_brake = self.create_publisher(Bool, topics["brake"], 10)
        self.pub_status = self.create_publisher(String, "/control/status", 10)
        self.pub_mission = self.create_publisher(String, "/control/mission_state", 10)
        self.pub_corridor = self.create_publisher(String, "/control/corridor_status", 10)
        self.pub_stopline = self.create_publisher(String, "/control/stopline_state", 10)
        self.pub_permit = self.create_publisher(String, "/control/signal_permit", 10)
        self.pub_steering = self.create_publisher(String, "/control/steering_debug", 10)
        self.pub_cte = self.create_publisher(Float32, "/control/cross_track_error", 10)
        self.pub_heading_err = self.create_publisher(Float32, "/control/heading_error", 10)
        self.pub_lookahead = self.create_publisher(PointStamped, "/control/global_lookahead", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/control/markers", 5)
        self.pub_corridor_markers = self.create_publisher(MarkerArray, "/control/corridor_markers", latched)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        self.create_service(Trigger, "~/reset_missions", self._on_reset_missions)
        self.create_service(Trigger, "~/reload_config", self._on_reload_config)

        self.create_timer(1.0 / self.rate, self._control)
        self.create_timer(5.0, self._stall_check, clock=Clock(clock_type=ClockType.SYSTEM_TIME))
        self._publish_corridor_markers()
        self.get_logger().info(
            f"{NODE_NAME} started (no arm step): stop until ready, then tracking starts by itself; "
            f"map {self._profile_label()}; front offset {self.front_offset:.3f} m "
            f"({self.front_source}), rear {self.rear_offset:.3f} m, control {self.rate:.0f} Hz"
        )

    # ------------------------------------------------------------ configuration
    def _load_control(self) -> None:
        self.cfg = load_control_config(self.paths["control"], self.paths["overrides"], self.paths["speed_mode"])

    def _profile_label(self) -> str:
        return self.profile.label() if self.profile is not None else "unknown (waiting for /localization/map_dir or a trajectory)"

    def _select_initial_profile(self) -> None:
        """Explicit map_profile / map_dir / force_route_only; else wait for the topic."""

        if self.explicit_profile:
            directory = os.path.abspath(os.path.expanduser(self.explicit_profile))
            run_name = os.path.basename(directory.rstrip("/"))
            map_dir = self.explicit_map_dir or None
            try:
                profile = load_profile(directory, run_name, map_dir)
            except (OSError, ValueError, KeyError, TypeError) as error:
                profile = MapProfile(run_name=run_name, display_name="INVALID", directory=directory, map_dir=map_dir, route_only=False)
                profile.problem = f"map profile '{directory}' unusable: {error}"
            self._apply_profile(profile, configure=False)
        elif self.explicit_map_dir:
            self._select_profile_for(run_name_of(self.explicit_map_dir), self.explicit_map_dir, configure=False)
        else:
            self.readiness = readiness(self.cfg, self.mission_cfg, self.segments_cfg, check_files=True)

    def _select_profile_for(self, run_name: str, map_dir: str | None, configure: bool = True) -> None:
        if self.force_route_only:
            profile = route_only_profile(run_name, map_dir)
        else:
            profile = find_profile(self.paths["maps"], run_name, map_dir)
        self._apply_profile(profile, configure=configure)

    def _apply_profile(self, profile: MapProfile, configure: bool = True) -> None:
        self.profile = profile
        if profile.problem is None:
            try:
                self.mission_cfg = profile.mission_config()
            except (OSError, ValueError, KeyError, TypeError) as error:
                profile.problem = f"map profile '{profile.directory}' unusable: {error}"
        if profile.problem is not None:
            # Never fall back to route_only on a BROKEN profile: that would
            # drive the course without its missions.
            self.mission_cfg = route_only_profile(profile.run_name, profile.map_dir).mission_config()
        self.segments_cfg = {"segments": self.mission_cfg.get("segments", {"list": []})}
        self.readiness = readiness(self.cfg, self.mission_cfg, self.segments_cfg, check_files=True)
        if profile.problem is not None:
            self.readiness.blocking.insert(0, profile.problem)
        for w in profile.warnings:
            self.readiness.warnings.append(w)
        self.get_logger().info(f"map profile: {profile.label()}\n" + self.readiness.summary())
        self._load_zone_brake_ranges(profile.map_dir)
        if configure:
            self._configure_models()
            if self.last_trajectory_msg is not None:
                self._on_trajectory(self.last_trajectory_msg)

    def _load_zone_brake_ranges(self, map_dir: str | None) -> None:
        """The map's avoidance zones for the zone brake (zone_brake.py); None =
        no readable file -> brake on /avoidance_space itself, as before."""

        self.zone_brake_ranges = None
        if not map_dir:
            return
        path = os.path.join(os.path.expanduser(map_dir), "avoidance_zones.yaml")
        if not os.path.isfile(path):
            self.get_logger().warn(f"zone brake: no {path}; braking on {self.zone_brake_topic} at the zone itself")
            return
        try:
            self.zone_brake_ranges = load_zone_ranges(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            self.get_logger().error(f"zone brake: {error}; braking on {self.zone_brake_topic} at the zone itself")
            return
        starts = ", ".join(f"#{max(a - self.zone_brake_lead_wp, 0):g} (zone #{a}-{b})" for a, b in self.zone_brake_ranges)
        self.get_logger().info(
            f"zone brake: {path} -> stop target {starts or 'nowhere (no zones)'}, "
            f"{self.zone_brake_lead_wp:g} waypoints before each avoidance zone; stopping-distance check from "
            f"{self.zone_brake_window_wp:g} waypoints before that"
        )

    def _configure_models(self) -> None:
        cfg = self.cfg
        vehicle = cfg["vehicle"]
        self.wheelbase = float(vehicle["wheelbase_m"])
        self.half_width = 0.5 * float(vehicle["width_m"])
        self.front_offset, self.front_source = front_offset_m(vehicle)
        self.rear_offset, _ = rear_offset_m(vehicle, self.front_offset)
        self.rear_axle_offset = float(vehicle.get("rear_axle_offset_m", 0.0))

        self.calib = SteeringCalibration.from_config(cfg["steering"])
        self.get_logger().info(self.calib.describe())
        self.rw_limits = self.calib.road_wheel_limits
        self.max_rw_rate = math.radians(float(cfg["steering"]["max_road_wheel_rate_deg_s"]))

        speed = cfg["speed"]
        self.v_cruise = float(speed["cruise_kmh"]) / KMH
        # null = no approach cap: signal missions and the next-zone anticipation run at cruise.
        self.v_approach = (self.v_cruise if speed.get("signal_approach_kmh") is None
                           else float(speed["signal_approach_kmh"]) / KMH)
        self.v_obstacle = float(speed["obstacle_kmh"]) / KMH
        self.v_turn = None if speed.get("turn_kmh") is None else float(speed["turn_kmh"]) / KMH
        pause = cfg.get("pause") or {}
        self.pause_hold_s = float(pause.get("hold_s", 3.0))
        self.pause_stop_gap = float(pause.get("stop_gap_m", 0.5))
        self.pause_stop_margin = float(pause.get("stop_margin_m", 0.0))
        self.pause_window = float(pause.get("brake_window_m", 15.0))
        self.pause_max_brake_s = float(pause.get("max_brake_s", 5.0))
        self.v_max = float(speed["max_kmh"]) / KMH
        self.v_min_moving = float(speed["min_moving_kmh"]) / KMH
        self.comfort_decel = float(speed.get("comfort_decel_mps2", 0.5))
        self.align_limit = math.radians(float(speed.get("align_heading_deg", 10.0) or 0.0))
        self.align_steer_limit = math.radians(float(speed.get("align_steer_deg", 10.0) or 0.0))
        self.align_encoder_timeout_s = float(speed.get("align_encoder_timeout_s", 0.5))
        coast_command = str(speed.get("coast_command", "ramp")).strip().lower()
        if coast_command not in ("ramp", "step"):
            raise ValueError(f"speed.coast_command must be 'ramp' or 'step', not '{coast_command}'")
        self.coast_aim_radius = float(speed.get("coast_aim_tight_radius_m", 10.0) or 0.0)
        self.coast_aim_before = float(speed.get("coast_aim_before_tight_m", 2.0) or 0.0)
        self.coast_ramp_below = (
            float(speed.get("coast_ramp_below_kmh", 2.0) or 0.0) / KMH if coast_command == "ramp" else None
        )
        self.v_cap_unverified = (
            None if self.calib.verified else float(speed["cap_until_steering_verified_kmh"]) / KMH
        )
        self.brake_release_hold_s = float(speed.get("brake_release_hold_s", 0.8))
        # Short brake on entering an obstacle-avoidance zone (zone_brake.py).
        zb = cfg.get("zone_brake") or {}
        self.zone_brake_enabled = bool(zb.get("enabled", False))
        self.zone_brake_topic = str(zb.get("topic", "/avoidance_space"))
        self.zone_brake_lead_wp = float(zb.get("lead_waypoints", 0.0))
        self.zone_brake_lead_speed = float(zb.get("lead_speed_kmh", 0.0)) / KMH
        self.zone_brake_stop_margin = float(zb.get("stop_margin_m", 0.0))
        self.zone_brake_window_wp = float(zb.get("brake_window_waypoints", 15.0))
        if not hasattr(self, "zone_brake_ranges"):
            self.zone_brake_ranges: list[tuple[int, int]] | None = None
        zb_params = (float(zb.get("stop_hold_s", 1.0)), float(zb.get("rearm_outside_s", 2.0)),
                     float(zb.get("stale_s", 1.0)), float(zb.get("max_brake_s", 5.0)))
        if getattr(self, "_zone_brake_params", None) != zb_params:
            self.zone_brake = ZoneEntryBrake(*zb_params)
            self._zone_brake_params = zb_params
        previous_ramp = getattr(self, "start_ramp", None)
        self.start_ramp = StartRamp(
            float(speed.get("start_ramp_speed_kmh", 5.0)) / KMH, float(speed.get("start_ramp_accel_mps2", 1.0))
        )
        previous_trim = getattr(self, "speed_trim", None)
        self.speed_trim = SpeedTrim(
            float(speed.get("trim_gain_per_s", 0.3)),
            float(speed.get("trim_max_kmh", 5.0)) / KMH,
            float(speed.get("trim_min_speed_kmh", 1.0)) / KMH,
        )
        if previous_trim is not None:
            self.speed_trim.trim_mps = min(previous_trim.trim_mps, self.speed_trim.max_trim_mps)
        del previous_ramp

        blend = cfg.get("blend") or {}
        self.blend = BlendParams(
            wheelbase_m=self.wheelbase,
            straight_radius_m=float(blend.get("straight_radius_m", 50.0)),
            curve_radius_m=float(blend.get("curve_radius_m", 5.0)),
            max_stanley_weight=float(blend.get("max_stanley_weight", 0.5)),
            heading_gain=float(blend.get("heading_gain", 0.6)),
            cross_track_gain=float(blend.get("cross_track_gain", 0.3)),
            softening_mps=float(blend.get("cross_track_softening_kmh", 10.8)) / KMH,
            heading_window_m=float(blend.get("heading_window_m", 4.0)),
            yaw_rate_damping_gain_s=float(blend.get("yaw_rate_damping_gain_s", 0.15)),
            yaw_rate_damping_max_loop_gain=float(blend.get("yaw_rate_damping_max_loop_gain", 0.0)),
            feedforward_gain=float(blend.get("feedforward_gain", 1.0)),
        )
        self.blend.validate()

        # The independent 0924 package used this law for its front-axle path.
        # It is disabled in the normal controller config and enabled only by
        # cone_avoidance_1005's integrated override.
        avoid = cfg.get("avoidance_stanley") or {}
        self.avoidance_stanley_enabled = bool(avoid.get("enabled", False))
        self.avoidance_mode_topic = str(
            avoid.get("mode_topic", "/integration_control/avoiding")
        ).strip()
        self.avoidance_require_signal = bool(avoid.get("require_fresh_signal", True))
        self.avoidance_signal_timeout = float(avoid.get("signal_timeout_s", 0.5))
        self.avoidance_k_heading = float(avoid.get("k_heading", 1.0))
        self.avoidance_k_stanley = float(avoid.get("k_stanley", 1.5))
        self.avoidance_k_soft = float(avoid.get("k_soft_mps", 0.5))
        self.avoidance_latency = float(avoid.get("latency_compensation_s", 0.60))
        self.avoidance_heading_window = float(avoid.get("heading_window_m", 0.60))
        self.avoidance_max_rw = math.radians(float(avoid.get("max_road_wheel_deg", 21.0)))
        self.avoidance_max_rw_rate = math.radians(
            float(avoid.get("max_road_wheel_rate_deg_s", 17.0))
        )

        pp = cfg["pure_pursuit"]
        self.segment_defaults = SegmentParams(
            name="global",
            speed_cap_mps=None,
            lookahead_straight_m=float(pp["lookahead_straight_m"]),
            lookahead_curve_m=float(pp["lookahead_curve_m"]),
            straight_radius_m=float(pp["straight_radius_m"]),
            curve_radius_m=float(pp["curve_radius_m"]),
            max_road_wheel_rate_rad_s=self.max_rw_rate,
        )
        self.lookahead_time = float(pp.get("lookahead_time_s", 0.0))
        self.lookahead_straight_time = float(pp.get("lookahead_straight_time_s", 0.0))
        self.max_lateral = float(pp["max_lateral_offset_m"])
        self.pose_far = float(pp.get("pose_far_distance_m", 10.0))
        self.max_heading_err = math.radians(float(pp["max_heading_error_deg"]))
        self.proj_kwargs = dict(
            back_window_m=float(pp["projection_back_window_m"]),
            forward_window_m=float(pp["projection_forward_window_m"]),
            max_backtrack_m=float(pp["projection_max_backtrack_m"]),
        )
        join = cfg.get("route_join") or {}
        self.join_enabled = bool(join.get("enabled", False))
        self.join_max_distance = float(join.get("max_distance_m", 60.0))
        self.join_speed = float(join.get("speed_kmh", 5.0)) / KMH
        self.join_tie_distance = float(join.get("tie_distance_m", 1.0))
        self.join_pass_tie_distance = float(join.get("pass_tie_distance_m", 8.0))
        self.join_max_advance = float(join.get("max_target_advance_m", 30.0))
        self.join_min_ahead = float(join.get("min_target_ahead_m", 1.5))

        br = cfg["braking"]
        self.braking: BrakingModel | None = None
        if all(is_number(br.get(k)) for k in ("safe_deceleration_mps2", "total_latency_s", "distance_margin_m")):
            self.braking = BrakingModel.from_config(br)

        st = cfg["stopline"]
        band = st.get("lateral_half_band_m")
        self.stopline = StoplineEstimator(
            half_band_m=float(band) if is_number(band) else self.half_width,
            min_span_m=float(st["min_span_m"]),
            max_abs_slope=float(st["max_abs_slope"]),
            max_distance_m=float(st["max_distance_m"]),
        )
        self.live_ttl = float(st["live_ttl_s"]) if is_number(st.get("live_ttl_s")) else None
        self.live_tol = (
            float(st["map_live_association_tolerance_m"])
            if is_number(st.get("map_live_association_tolerance_m"))
            else None
        )
        self.stop_buffer = float(st["stop_buffer_m"]) if is_number(st.get("stop_buffer_m")) else None

        sg = cfg["signal"]
        self.signal_pre_detect_margin = float(sg.get("pre_detect_margin_m", 2.0))
        previous = getattr(self, "permits", None)
        self.permits = SignalPermitTracker(ttl_s=float(sg["ttl_s"]), target_jump_px=float(sg["target_jump_px"]))
        if previous is not None:  # keep the detection history across a reload
            self.permits.dirs = previous.dirs
            self.permits.last_json_stamp_ns = previous.last_json_stamp_ns
            self.permits.last_json_recv_ns = previous.last_json_recv_ns
            self.permits.last_state = previous.last_state
            self.permits.last_selection = previous.last_selection
            self.permits.last_target = previous.last_target
            self.permits.last_fresh = previous.last_fresh
        self.commit_mode = str(sg.get("commit_mode", "last_stoppable_point"))
        self.commit_distance_cfg = sg.get("commit_distance_m")
        self.timeout_departure_s = (
            float(sg["timeout_departure_s"]) if is_number(sg.get("timeout_departure_s")) else None
        )
        self.timeout_requires_alive = bool(sg.get("timeout_requires_detector_alive", True))
        self.detector_alive_timeout = float(sg.get("detector_alive_timeout_s", 1.0))

        heading = cfg.get("heading") or {}
        self.heading_required = bool(heading.get("verified_required", True))
        self.heading_required_source = str(heading.get("required_source", "") or "")
        self.heading_status_timeout = float(heading.get("status_timeout_s", 0.0) or 0.0)
        self.heading_align = bool(heading.get("align_drive", False))
        self.heading_align_speed = float(heading.get("align_speed_kmh", 5.0)) / KMH

        # ---- corridor ------------------------------------------------------
        co = cfg["corridor"]
        self.fallback_half_width = float(co.get("fallback_half_width_m", 3.0))
        self.fallback_extension = float(co.get("fallback_end_extension_m", 10.0))
        self.v_outside_lane = float(co.get("outside_lane_speed_kmh", 5.0)) / KMH
        self.supervisor = None
        self.static_corridor = None
        self.corridor_info = {}
        self.corridor_error = ""
        lane_cfg = self.mission_cfg.get("lane_boundaries")
        try:
            if lane_cfg:
                bounds = load_boundaries(
                    os.path.expanduser(lane_cfg["csv"]),
                    lane_cfg.get("left_namespace_key") or "leftBoundary",
                    lane_cfg.get("right_namespace_key") or "rightBoundary",
                )
                self.corridor_info = validate_boundaries(bounds)
                expected = int(lane_cfg.get("expected_points_per_boundary", 0) or 0)
                if expected and (len(bounds.left) != expected or len(bounds.right) != expected):
                    raise ValueError(
                        f"boundary point count {len(bounds.left)}/{len(bounds.right)} != expected {expected}"
                    )
                self.static_corridor = StaticCorridor(
                    bounds, float(co.get("end_extension_m", 0.0)), int(co.get("window_points", 45))
                )
            margins = [co.get(k) for k in ("localization_error_m", "map_error_m", "tracking_error_m", "physical_margin_m")]
            if all(is_number(v) for v in margins):
                inset = self.half_width + sum(float(v) for v in margins)
                self.supervisor = CorridorSupervisor(
                    self.static_corridor,
                    SupervisorParams(
                        wheelbase_m=self.wheelbase,
                        front_m=self.front_offset,
                        rear_m=self.rear_offset,
                        half_width_m=self.half_width,
                        inset_m=inset,
                        sim_dt_s=float(co["sim_dt_s"]),
                        min_horizon_s=float(co["min_horizon_s"]),
                        max_horizon_s=float(co["max_horizon_s"]),
                        candidate_step_rad=math.radians(float(co["candidate_step_deg"])),
                        candidate_count=int(co["candidate_count"]),
                        candidate_hold_s=float(co.get("candidate_hold_s", 0.5)),
                        candidate_max_lat_accel_mps2=float(co.get("candidate_max_lat_accel_mps2", 0.0)),
                        max_steering_rate_rad_s=self.max_rw_rate,
                        lookahead_straight_m=self.segment_defaults.lookahead_straight_m,
                        lookahead_curve_m=self.segment_defaults.lookahead_curve_m,
                        lookahead_straight_radius_m=self.segment_defaults.straight_radius_m,
                        lookahead_curve_radius_m=self.segment_defaults.curve_radius_m,
                        lookahead_time_s=self.lookahead_time,
                        lookahead_straight_time_s=self.lookahead_straight_time,
                    ),
                    fallback=self.fallback_corridor,
                )
                self.corridor_info["inset_m"] = inset
                if "width_min_m" in self.corridor_info:
                    self.corridor_info["narrowest_free_m"] = self.corridor_info["width_min_m"] - 2.0 * inset
                self.corridor_info["lane_map"] = "yes" if self.static_corridor is not None else "no (route corridor only)"
        except (OSError, ValueError, KeyError) as error:
            self.corridor_error = str(error)
            self.get_logger().error(f"lane boundaries unusable: {error}")
        self.get_logger().info(f"corridor: {self.corridor_info or self.corridor_error}")

        timing = cfg["timing"]
        self.rate = float(timing["control_rate_hz"])
        self.pose_timeout = float(timing["pose_timeout_s"])
        self.stopped_tracker = StoppedTracker(float(timing["stopped_speed_mps"]), float(timing["stopped_confirm_s"]))
        self.wheel_scale = float(timing.get("wheel_speed_scale_to_mps", 1.0))

        if self.route is not None:
            self._bind_corridors(self.route)
            self._build_missions(self.route)

    def _commit_distance(self) -> float:
        """Distance from the stop-control point at which a permitted vehicle commits."""

        if self.braking is None:
            return 0.0
        computed = self.braking.required_distance(self.v_min_moving)
        if is_number(self.commit_distance_cfg):
            configured = float(self.commit_distance_cfg)
            if configured < computed:
                self.get_logger().warn(
                    f"signal.commit_distance_m={configured:.2f} is below the crawl stopping distance "
                    f"{computed:.2f} m; using {computed:.2f}"
                )
            return max(configured, computed)
        return computed

    def _utm_to_route_frame(self, route: RouteModel):
        """Callable UTM (x, y) -> route frame (x, y) through TF, or None when no transform."""

        utm_frame = str(get(self.cfg, "maps.utm_frame", "utm"))
        if route.frame_id == utm_frame:
            return lambda x, y: (x, y)
        try:
            transform = self.tf_buffer.lookup_transform(route.frame_id, utm_frame, Time(), timeout=Duration(seconds=0.0))
        except TransformException:
            return None
        t = transform.transform.translation
        q = transform.transform.rotation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        c, s = math.cos(yaw), math.sin(yaw)
        return lambda x, y: (c * x - s * y + t.x, s * x + c * y + t.y)

    def _build_missions(self, route: RouteModel) -> None:
        previous = self.missions
        try:
            table = M.load_missions(self.mission_cfg.get("missions", {}) or {})
            accept = bool(get(self.mission_cfg, "missions.accept_pass_end_candidates", False))
            self.resolved_missions, self.mission_warnings = M.resolve_missions(
                table, route, accept, pause_window_m=self.pause_window)
            to_route = self._utm_to_route_frame(route) if any(m.utm_points for m in table) else None
            if any(m.utm_points for m in table):
                if to_route is None:
                    self.mission_warnings.append("mission *_utm cross-check skipped: no utm->route TF yet")
                else:
                    tol = self.profile.utm_cross_check_tolerance_m if self.profile else 2.0
                    errors, checked = M.cross_check_missions(table, route, to_route, tol)
                    if errors:
                        raise ValueError("; ".join(errors))
                    self.mission_warnings.append(f"mission *_utm cross-check passed ({checked} points)")
            specs = load_segments(self.segments_cfg.get("segments", {}) or {})
            self.segments = SegmentTable(resolve_segments(specs, route), self.segment_defaults)
        except ValueError as error:
            self.mission_warnings = [f"MISSION_OR_SEGMENT_TABLE_INVALID: {error}"]
            self.route_problem = f"MISSION_OR_SEGMENT_TABLE_INVALID {error}"
            self.get_logger().error(self.route_problem)
            return
        hold = self.braking.required_distance(0.0) if self.braking else 0.0
        self.missions = M.MissionManager(
            self.resolved_missions,
            cruise_mps=self.v_cruise,
            approach_mps=self.v_approach,
            obstacle_mps=self.v_obstacle,
            turn_mps=self.v_turn,
            stop_buffer_m=self.stop_buffer if self.stop_buffer is not None else 0.0,
            hold_distance_m=hold,
            commit_distance_m=self._commit_distance(),
            commit_mode=self.commit_mode,
            timeout_departure_s=self.timeout_departure_s,
            timeout_requires_alive=self.timeout_requires_alive,
            pause_hold_s=self.pause_hold_s,
            pause_stop_gap_m=self.pause_stop_gap,
            pause_max_brake_s=self.pause_max_brake_s,
        )
        self.missions.carry_over(previous)
        for w in self.mission_warnings:
            self.get_logger().warn(f"missions: {w}")
        lines = [
            f"{rm.mission_id} {rm.kind}/{rm.mission.direction} start={rm.start_s:.1f}"
            + (f" decision={rm.decision_s:.1f} stop={rm.stop_anchor_s:.1f}" if rm.kind == M.SIGNAL else "")
            + (f" zone={rm.zone_start_s:.1f} stop={rm.stop_anchor_s:.1f}" if rm.kind == M.PAUSE else "")
            + f" end={rm.end_s:.1f}({rm.end_source})"
            for rm in self.resolved_missions
        ]
        seg_lines = [
            f"'{seg.spec.name}' s={seg.start_s:.0f}..{seg.end_s:.0f}"
            + (f" v<={seg.spec.speed_kmh:.0f}km/h" if seg.spec.speed_kmh else "")
            + (f" ld={seg.spec.lookahead_straight_m}/{seg.spec.lookahead_curve_m}" if seg.spec.lookahead_straight_m or seg.spec.lookahead_curve_m else "")
            for seg in self.segments.resolved
        ]
        self.get_logger().info(
            "missions on '" + route.trajectory_id + "' (commit distance "
            f"{self.missions.commit_distance:.2f} m, hold {hold:.2f} m, timeout "
            f"{'off' if self.timeout_departure_s is None else f'{self.timeout_departure_s:.0f}s'}):\n  "
            + ("\n  ".join(lines) if lines else "(none - route_only)") + "\nsegments:\n  " + ("\n  ".join(seg_lines) if seg_lines else "(none)")
        )

    def _build_fallback_corridor(self, route: RouteModel) -> None:
        try:
            bounds = route_boundaries(route.xy, self.fallback_half_width)
            self.fallback_corridor = StaticCorridor(bounds, self.fallback_extension, 45)
        except ValueError as error:
            self.fallback_corridor = None
            self.get_logger().error(f"route corridor unavailable: {error}")
        if self.supervisor is not None:
            self.supervisor.set_fallback(self.fallback_corridor)

    def _bind_corridors(self, route: RouteModel) -> None:
        """Fix, once per trajectory, which wall segments belong to each route
        point (corridor.py bind_route) - the walls themselves never change."""

        for name, corridor in (("lane walls", self.static_corridor), ("route corridor", self.fallback_corridor)):
            if corridor is None:
                continue
            try:
                stats = corridor.bind_route(route.xy)
            except ValueError as error:
                self.get_logger().error(f"{name}: cannot bind to the route: {error}")
                continue
            if stats["warnings"]:
                self.get_logger().warn(f"{name} bound to the route with warnings: " + " | ".join(stats["warnings"]))
            self.get_logger().info(f"{name} bound to the route: {stats['summary']}")

    # ------------------------------------------------------------------ inputs
    def _on_map_dir(self, msg: String) -> None:
        map_dir = str(msg.data).strip()
        if not map_dir:
            return
        run_name = run_name_of(map_dir)
        if self.profile is not None and self.profile.run_name == run_name and self.profile.problem is None:
            if self.profile.map_dir != map_dir:
                self.profile.map_dir = map_dir
            return
        if self.tracking_started:
            self._stop_tracking(f"localization map changed to '{run_name}'")
        self.get_logger().warn(f"localization map: '{map_dir}' (run '{run_name}')")
        self._select_profile_for(run_name, map_dir)
        if self.missions is not None:
            self.missions.reset_progress()
        self.route_s = None
        self.front_s = None
        self.avoidance_front_s = None

    def _on_trajectory(self, msg: Trajectory) -> None:
        try:
            route = RouteModel.from_trajectory(msg)
        except ValueError as error:
            self.route_problem = f"BAD_TRAJECTORY {error}"
            self.get_logger().error(self.route_problem)
            return
        self.last_trajectory_msg = msg
        if self.profile is None or (not self.profile.run_name and route.run_name):
            # No map announced yet: the trajectory id names the run.
            self._select_profile_for(route.run_name, self.explicit_map_dir or None, configure=True)
            return  # _apply_profile re-enters here with the stored message
        if self.profile.problem is not None:
            self.route_problem = f"MAP_PROFILE_UNUSABLE {self.profile.problem}"
            return
        if not self.profile.route_only:
            rc = self.mission_cfg["route"]
            if self.profile.run_name and route.run_name != self.profile.run_name:
                self.route_problem = f"TRAJECTORY_RUN '{route.run_name}' != profile '{self.profile.run_name}'"
                self.get_logger().error(self.route_problem)
                return
            expected_n = int(rc.get("expected_waypoint_count", 0) or 0)
            if expected_n and route.count != expected_n:
                self.route_problem = f"TRAJECTORY_COUNT {route.count}!={expected_n}"
                self.get_logger().error(self.route_problem + " - mission indices would be misaligned; refusing")
                return
            frame = rc.get("frame_id")
            if frame and route.frame_id != frame:
                self.route_problem = f"TRAJECTORY_FRAME '{route.frame_id}'!='{frame}'"
                self.get_logger().error(self.route_problem)
                return
        first = self.route is None or self.route.trajectory_id != route.trajectory_id or self.route.frame_id != route.frame_id
        self.route = route
        self._route_drops = coast_aim_points(
            profile_drops(route.s, route.speed), route.s, route.curvature, self.coast_aim_radius, self.coast_aim_before
        )
        self._route_vmax = float(np.max(route.speed)) if len(route.speed) else 0.0
        if first:
            self._coast_latched = set()
            self.avoidance_front_s = None
        self.route_problem = ""
        if first or self.fallback_corridor is None:
            self._build_fallback_corridor(route)
            self._publish_corridor_markers()
        if first:
            self._bind_corridors(route)
        if first or self.missions is None:
            self._build_missions(route)

    def _on_avoidance(self, msg: Trajectory) -> None:
        try:
            self.avoidance_route = RouteModel.from_trajectory(msg)
            # This is a sliding local path (roughly s_now-2 .. s_now+20), so
            # its arc-length origin changes on every update.
            self.avoidance_front_s = None
        except ValueError as error:
            self.get_logger().warn(f"ignoring avoidance trajectory: {error}")

    def _on_avoidance_mode(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested != self.avoidance_mode_requested:
            self.avoidance_front_s = None
            if self.avoidance_stanley_enabled:
                self.get_logger().info(
                    "steering law request -> "
                    + ("0924 latency Stanley" if requested else "normal PP+StanleyFF")
                )
        self.avoidance_mode_requested = requested
        self.avoidance_mode_rx_ns = self.get_clock().now().nanoseconds

    def _on_pose(self, msg: Odometry, topic: str) -> None:
        self.pose_msgs[topic] = (msg, self.get_clock().now().nanoseconds)

    def _on_wheel(self, msg: WheelSpeedStamped) -> None:
        self.wheel = (float(msg.speed_mps) * self.wheel_scale, self.get_clock().now().nanoseconds)

    def _on_encoder(self, msg: EncoderAngleStamped) -> None:
        self.encoder = (float(msg.angle_rad), self.get_clock().now().nanoseconds)

    def _on_stopline(self, msg: MarkerArray) -> None:
        stamp_ns = None
        for marker in msg.markers:
            if marker.action == Marker.ADD:
                stamp_ns = Time.from_msg(marker.header.stamp).nanoseconds
                break
        if stamp_ns is None:
            return
        self.stopline.ingest(msg.markers, stamp_ns, self.route_s)

    def _on_traffic_json(self, msg: String) -> None:
        self.permits.ingest_json(msg.data, self.get_clock().now().nanoseconds)

    def _on_go_bool(self, direction: str, msg: Bool) -> None:
        self.permits.ingest_confirmed_bool(direction, bool(msg.data), self.get_clock().now().nanoseconds)

    def _on_heading_verified(self, msg: Bool) -> None:
        self.heading_verified = bool(msg.data)
        self.heading_verified_ns = self.get_clock().now().nanoseconds

    def _on_heading_source(self, msg: String) -> None:
        self.heading_source = str(msg.data).strip()

    # ---------------------------------------------------------------- services
    def _heading_problem(self, now_ns: int) -> str | None:
        if not self.heading_required:
            return None
        if self.heading_verified is None:
            return "HEADING_UNVERIFIED: no /ekf/heading_verified received yet"
        if self.heading_status_timeout > 0.0 and self.heading_verified_ns is not None:
            if (now_ns - self.heading_verified_ns) * 1e-9 > self.heading_status_timeout:
                return "HEADING_STATUS_STALE: /ekf/heading_verified not refreshed"
        if not self.heading_verified:
            return (
                "HEADING_UNVERIFIED (/ekf/heading_verified=false): the ESKF still runs on its saved "
                "heading - drive 10-15 m straight in manual so it locks the RTK course"
            )
        if self.heading_required_source:
            if self.heading_source is None:
                return f"HEADING_SOURCE_UNKNOWN: /ekf/heading_source not received (need '{self.heading_required_source}')"
            if self.heading_source != self.heading_required_source:
                return f"HEADING_SOURCE '{self.heading_source}' != '{self.heading_required_source}'"
        return None

    def _start_blockers(self, heading_problem: str | None) -> list[str]:
        """What still keeps tracking from starting (route and pose are checked before)."""
        reasons = list(self.readiness.blocking)
        if self.profile is None:
            reasons.append("no map selected yet (waiting for /localization/map_dir or /planning/trajectory)")
        if self.route is None:
            reasons.append(self.route_problem or "NO_TRAJECTORY")
        if self.missions is None or self.segments is None:
            reasons.append("missions/segments not resolved: " + (self.route_problem or ""))
        if self.supervisor is None:
            reasons.append("corridor supervisor unavailable: " + (self.corridor_error or "margins unset"))
        elif not self.supervisor.available:
            reasons.append("corridor supervisor has no walls (no lane map and no route corridor yet)")
        if self.braking is None:
            reasons.append("braking model unavailable")
        if self.stop_buffer is None:
            reasons.append("stopline.stop_buffer_m unset")
        # With align_drive an unverified heading starts into HEADING_ALIGN; a
        # status that never arrived still blocks (no ESKF to wait for).
        if heading_problem is not None and not (self.heading_align and self.heading_verified is not None):
            reasons.append(heading_problem)
        return reasons

    def _start_tracking(self, now_ns: int, pose: PoseSample, heading_problem: str | None) -> None:
        """Initialize the route projection and the missions where the vehicle is now."""
        proj = self.route.project(pose.x, pose.y, pose.yaw, None, **self.proj_kwargs)
        skipped = []
        self.join_allowed = True
        if proj is not None and proj.distance <= self.max_lateral:
            self.route_s = proj.s
            skipped = self.missions.initialize_at(proj.s, self.front_offset)
            self.join_missions_pending = False
        else:
            # Off the route: _join_route drives to it, and the missions are
            # initialized where the vehicle actually joins, not here.
            self.route_s = None
            self.join_missions_pending = True
        self.front_s = None
        self.avoidance_front_s = None
        self.tracking_started = True
        self.last_speed_cmd = 0.0
        self._brake_latched_for = None
        self._brake_release_until_ns = None
        self._corridor_crawl = False
        self._coast_latched = set()
        self._align_hold = None
        # Not blocked, so a heading problem here means align_drive is on.
        self._heading_aligning = heading_problem is not None
        self.start_ramp.reset()
        self.speed_trim.reset()
        self.zone_brake.reset()
        if self.static_corridor is not None:
            self.static_corridor.reset()
        if self.fallback_corridor is not None:
            self.fallback_corridor.reset()
        self.get_logger().warn(
            f"TRACKING STARTED on {self._profile_label()} at s={self.route_s if self.route_s is None else round(self.route_s, 1)}"
            + (f", skipped missions already behind: {skipped}" if skipped else "")
            + (f". HEADING_ALIGN: steering 0 at {self.heading_align_speed * KMH:.0f} km/h until the ESKF heading locks"
               if self._heading_aligning else "")
            + ". Warnings: " + " | ".join(self.readiness.warnings + self.mission_warnings)
        )

    def _stop_tracking(self, why: str) -> None:
        """Back to the stop-until-ready state; tracking restarts by itself at the current projection."""
        if self.tracking_started:
            self.get_logger().warn(f"tracking stopped ({why}); holding a stop until the inputs are ready again")
        self.tracking_started = False
        self.route_s = None
        self.front_s = None
        self.avoidance_front_s = None
        self.avoidance_mode_active = False
        # FAILSAFE_STOP holds last_delta: do not carry a lock angle from the
        # previous map/session into the stop.
        self.last_delta = 0.0
        self._corridor_crawl = False
        self._coast_latched = set()
        self._align_hold = None
        self._heading_aligning = False
        self.speed_trim.reset()
        self.start_ramp.reset()

    def _moving(self) -> bool:
        pose = self._resolve_pose(self.get_clock().now().nanoseconds)
        return pose is not None and abs(pose.speed_mps) > self.stopped_tracker.threshold

    def _on_reset_missions(self, request, response):
        if self._moving():
            response.success = False
            response.message = "refused: stop the vehicle before resetting mission progress"
            return response
        if self.missions is not None:
            self.missions.reset_progress()
        self._stop_tracking("mission progress reset")
        response.success = True
        response.message = "mission progress reset; tracking restarts when ready"
        return response

    def _on_reload_config(self, request, response):
        if self._moving():
            response.success = False
            response.message = "refused: stop the vehicle before reloading configuration"
            return response
        self._stop_tracking("configuration reload")
        try:
            self._load_control()
            if self.profile is not None:
                if self.profile.directory and not self.profile.route_only:
                    self._apply_profile(load_profile(self.profile.directory, self.profile.run_name, self.profile.map_dir), configure=False)
                else:
                    self._apply_profile(self.profile, configure=False)
            else:
                self.readiness = readiness(self.cfg, self.mission_cfg, self.segments_cfg, check_files=True)
            self._configure_models()
            if self.last_trajectory_msg is not None:
                self._on_trajectory(self.last_trajectory_msg)
        except Exception as error:  # keep the previous configuration on any error
            response.success = False
            response.message = f"reload FAILED, previous configuration kept: {error}"
            self.get_logger().error(response.message)
            return response
        response.success = True
        response.message = "configuration reloaded: " + self.readiness.summary().replace("\n", " / ")
        self.get_logger().warn("configuration reloaded (topic names need a restart)")
        return response

    # ----------------------------------------------------------------- pose
    def _resolve_pose(self, now_ns: int) -> PoseSample | None:
        if self.route is not None:
            route_frame = self.route.frame_id
        else:
            route_frame = str(get(self.mission_cfg, "route.frame_id", "map") or "map")
        for topic in self.pose_order:
            entry = self.pose_msgs.get(topic)
            if entry is None:
                continue
            msg, recv_ns = entry
            age = (now_ns - recv_ns) * 1e-9
            if age > self.pose_timeout:
                continue
            pose = msg.pose.pose
            frame = msg.header.frame_id.lstrip("/")
            if frame != route_frame:
                transform = None
                for when in (Time.from_msg(msg.header.stamp), Time()):
                    try:
                        transform = self.tf_buffer.lookup_transform(route_frame, frame, when, timeout=Duration(seconds=0.0))
                        break
                    except TransformException:
                        continue
                if transform is None:
                    continue
                pose = do_transform_pose(pose, transform)
            yaw = yaw_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            x = float(pose.position.x) + self.rear_axle_offset * math.cos(yaw)
            y = float(pose.position.y) + self.rear_axle_offset * math.sin(yaw)
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
                continue
            return PoseSample(
                x=x, y=y, yaw=yaw,
                speed_mps=float(msg.twist.twist.linear.x),
                yaw_rate=float(msg.twist.twist.angular.z),
                stamp_ns=Time.from_msg(msg.header.stamp).nanoseconds,
                source=topic, age_s=age,
            )
        return None

    # -------------------------------------------------------------- control
    def _control(self) -> None:
        started = time.perf_counter()
        self.control_cycles += 1
        now_ns = self.get_clock().now().nanoseconds
        dt = 1.0 / self.rate
        if self.last_control_ns is not None:
            elapsed = (now_ns - self.last_control_ns) * 1e-9
            if 0.0 < elapsed < 10.0 * dt:
                dt = elapsed
        self.last_control_ns = now_ns

        pose = self._resolve_pose(now_ns)
        result = self._compute(now_ns, dt, pose)

        # Always on the actuator topics: a stop until ready, then the tracked
        # commands. A FAILSAFE_STOP keeps tracking: when the missing input
        # (pose, trajectory, corridor) recovers, the commands resume.
        self._publish_actuators(result)

        self.cycle_time_ms = (time.perf_counter() - started) * 1e3
        self._publish_status(result, pose, now_ns)

    def _stop_result(self, base: CycleResult, reason: str) -> CycleResult:
        return CycleResult(False, True, 0.0, self.last_delta, self.calib.command(self.last_delta), "FAILSAFE_STOP", reason, base.info)

    def _publish_actuators(self, result: CycleResult) -> None:
        # ros_can_bridge zeroes the speed while brake is true and ignores speed
        # callbacks until it is released, so the brake goes first.
        self.pub_brake.publish(Bool(data=bool(result.brake)))
        speed_kmh = 0.0 if result.brake else max(result.speed_mps, 0.0) * KMH
        speed_kmh = min(speed_kmh, self.v_max * KMH)
        self.pub_speed.publish(Float32(data=float(speed_kmh)))
        self.pub_angle.publish(Float32(data=float(result.command_rad)))
        self.last_cmd_angle = result.command_rad
        self.last_speed_cmd = 0.0 if result.brake else result.speed_mps

    def _upcoming_caps(self, s: float, horizon_m: float) -> list[tuple[float, float]]:
        """Speed caps that start ahead of the vehicle: mission zones + slower segments."""

        out = []
        if self.missions is not None:
            for rm in self.missions.missions:
                if rm.mission_id in self.missions.done or not (s < rm.start_s <= s + horizon_m):
                    continue
                if rm.kind == M.SIGNAL:
                    if self.v_approach < self.v_cruise:
                        out.append((rm.start_s, self.v_approach))
                elif rm.kind == M.OBSTACLE:
                    out.append((rm.start_s, self.v_obstacle))
                elif self.v_turn is not None:
                    out.append((rm.start_s, self.v_turn))
        if self.segments is not None:
            out.extend(self.segments.upcoming_speed_caps(s, horizon_m))
        return out

    def _join_route(self, route, pose: PoseSample, dt: float, info: dict) -> CycleResult:
        """Drive TO the route from off it (route_join in control.yaml).

        Target = the waypoint nearest to the vehicle's LOOKAHEAD position (the
        point one straight lookahead in front of it), lowest index first among
        equally near ones - RouteModel.join_waypoint. A waypoint that is not
        at least min_target_ahead_m in front of the vehicle cannot be steered
        to, so the target moves on along the route until one is.

        Off the route there is no lane corridor and no mission geometry, so
        neither runs here: the speed is the fixed crawl speed (through the
        start ramp) and the ONLY guards are the distance limit and the
        driver's E-stop. Pure Pursuit alone steers here - Stanley's terms need
        a projection the vehicle is not yet on. Allowed from the start of
        tracking until the route has been acquired once; never after that.
        """

        self.route_s = None  # re-acquire over the whole route when it is reached
        self.front_s = None
        self.avoidance_front_s = None
        ld = self.segment_defaults.lookahead_straight_m
        look_x = pose.x + ld * math.cos(pose.yaw)
        look_y = pose.y + ld * math.sin(pose.yaw)
        index, _ = route.join_waypoint(
            look_x, look_y, pose.yaw,
            tie_distance_m=self.join_tie_distance,
            pass_tie_distance_m=self.join_pass_tie_distance,
        )
        gap = float(np.min(np.hypot(route.xy[:, 0] - pose.x, route.xy[:, 1] - pose.y)))

        if gap > self.join_max_distance:
            self.last_speed_cmd = 0.0
            return CycleResult(
                False, True, 0.0, self.last_delta, self.calib.command(self.last_delta), "FAILSAFE_STOP",
                f"OFF_ROUTE distance={gap:.1f}m exceeds route_join.max_distance_m={self.join_max_distance:.0f}m",
                info,
            )

        def unreachable(bx, by):
            return bx < self.join_min_ahead or math.hypot(bx, by) < ld

        join_s = float(route.s[index])
        bx, by = to_body_frame(pose.x, pose.y, pose.yaw, *route.xy[index])
        while (
            unreachable(bx, by)
            and index < route.count - 1
            and float(route.s[index + 1]) - join_s <= self.join_max_advance
        ):
            index += 1
            bx, by = to_body_frame(pose.x, pose.y, pose.yaw, *route.xy[index])
        if bx < self.join_min_ahead:
            self.last_speed_cmd = 0.0
            return CycleResult(
                False, True, 0.0, self.last_delta, self.calib.command(self.last_delta), "FAILSAFE_STOP",
                f"JOIN_NO_TARGET_AHEAD route_gap={gap:.1f}m: the route near wp={index} is not in front "
                "of the vehicle - turn it toward the route",
                info,
            )
        tx, ty = (float(v) for v in route.xy[index])

        desired = self.calib.clamp_road_wheel(pure_pursuit_steering(bx, by, self.wheelbase))
        delta = rate_limited(self.last_delta, desired, self.segment_defaults.max_road_wheel_rate_rad_s, dt)
        self.last_delta = delta
        v_cmd = min(self.join_speed, self.start_ramp.cap(max(pose.speed_mps, 0.0), pose.x, pose.y))
        v_cmd = max(v_cmd, min(self.join_speed, self.v_min_moving))

        info["waypoint"] = float(index)
        info["lookahead"] = (tx, ty, math.hypot(bx, by))
        info["join"] = {"target": index, "gap_m": gap, "lookahead_xy": (look_x, look_y)}
        return CycleResult(
            True, False, v_cmd, delta, self.calib.command(delta), "JOIN_ROUTE",
            f"to wp={index} target={math.hypot(bx, by):.1f}m route_gap={gap:.1f}m (no corridor check off the route)",
            info,
        )

    def _brake_release_hold(self, now_ns: int, brake: bool) -> bool:
        """True while no speed may be commanded after a brake=1 -> 0 transition.

        The brake linear motor takes up to ~1 s to let go, so no speed is
        commanded for brake_release_hold_s after the brake is released. The
        start ramp restarts from that release.
        """

        if brake:
            self._braking_last = True
            self._brake_release_until_ns = None
            return False
        if self._braking_last:
            self._braking_last = False
            self._brake_release_until_ns = now_ns + int(self.brake_release_hold_s * 1e9)
            self.start_ramp.reset()
        if self._brake_release_until_ns is not None:
            if now_ns < self._brake_release_until_ns:
                return True
            self._brake_release_until_ns = None
        return False

    def _align_drive(
        self, now_ns: int, route: RouteModel, pose: PoseSample, dt: float, info: dict, problem: str
    ) -> CycleResult:
        """Straight ahead until the ESKF heading locks (heading.align_drive).

        Started with an unverified heading the yaw cannot be trusted, and with it
        everything built on it: Pure Pursuit, Stanley, the corridor rollouts,
        WRONG_DIRECTION. The course starts on a straight, so the vehicle drives
        straight - road wheel 0 at align_speed_kmh, no brake and, by decision
        (2026-10-02), no other check - and the ESKF locks its RTK course within
        10-15 m. The next cycle tracks the route.

        The speed trim runs so the vehicle really does align_speed_kmh (the
        drive MCU falls short at a crawl), a stop in between (NO_POSE) gets the
        usual brake-release hold, and the start ramp is fed here so tracking
        continues from the speed these metres have ramped to.
        """

        proj = route.project(pose.x, pose.y, None, self.route_s, **self.proj_kwargs)
        if proj is not None:
            if self.route_s is not None:
                self.route_s = proj.s
            info["route_s"] = proj.s
            info["waypoint"] = route.index_at_s(proj.s)
            info["lateral"] = proj.lateral_offset
        releasing = self._brake_release_hold(now_ns, False)
        v = max(pose.speed_mps, 0.0)
        self.start_ramp.cap(v, pose.x, pose.y)
        v_cmd = 0.0 if releasing else self.speed_trim.command(self.heading_align_speed, v, dt)
        info["trim_mps"] = self.speed_trim.trim_mps
        info["trim_active"] = not releasing
        delta = rate_limited(self.last_delta, 0.0, self.segment_defaults.max_road_wheel_rate_rad_s, dt)
        self.last_delta = delta
        return CycleResult(
            True, False, v_cmd, delta, self.calib.command(delta), "HEADING_ALIGN",
            f"straight ahead at {self.heading_align_speed * KMH:.0f} km/h, steering 0, "
            f"until {problem.split(' ')[0].rstrip(':')} clears" + (" | brake release hold" if releasing else ""),
            info,
        )

    def _compute(self, now_ns: int, dt: float, pose: PoseSample | None) -> CycleResult:
        info: dict = {}
        hold_cmd = self.calib.command(self.last_delta)

        def stop(reason, state="FAILSAFE_STOP"):
            self.last_speed_cmd = 0.0
            self._braking_last = True
            return CycleResult(False, True, 0.0, self.last_delta, hold_cmd, state, reason, info)

        route = self.route
        if route is None:
            return stop(self.route_problem or "NO_TRAJECTORY")
        if now_ns > route.valid_until_ns + int(TRAJECTORY_STALE_WARN_S * 1e9):
            # The Global Path is fixed: the last one received stays in use.
            self.get_logger().warn(
                f"/planning/trajectory stale by {(now_ns - route.valid_until_ns) * 1e-9:.0f} s; keeping the last route",
                throttle_duration_sec=10.0,
            )
        if pose is None:
            return stop("NO_POSE (no fresh source transformable into '%s')" % route.frame_id)
        info["pose_source"] = pose.source

        # The mode topic is a heartbeat as well as a selector.  Only the
        # integrated override enables this check, so the ordinary controller
        # launch has exactly its previous behaviour.
        avoidance_signal_age = None
        avoidance_signal_fresh = False
        if self.avoidance_mode_rx_ns is not None:
            avoidance_signal_age = (now_ns - self.avoidance_mode_rx_ns) * 1e-9
            avoidance_signal_fresh = (
                0.0 <= avoidance_signal_age <= self.avoidance_signal_timeout
            )
        avoidance_requested = bool(
            self.avoidance_stanley_enabled
            and avoidance_signal_fresh
            and self.avoidance_mode_requested
        )
        avoidance_path_ready = bool(
            self.avoidance_route is not None
            and now_ns <= self.avoidance_route.valid_until_ns
            and self.avoidance_route.frame_id == route.frame_id
        )
        # While integration requests the initial zone baseline or a live cone
        # path, 0924 Stanley is active. A valid local path replaces the map
        # route only after a cone has actually produced one.
        avoidance_active = avoidance_requested
        if avoidance_active != self.avoidance_mode_active:
            self.avoidance_front_s = None
        self.avoidance_mode_active = avoidance_active
        info["avoidance_mode_requested"] = self.avoidance_mode_requested
        info["avoidance_path_ready"] = avoidance_path_ready
        info["avoidance_signal_fresh"] = avoidance_signal_fresh
        info["avoidance_signal_age_s"] = avoidance_signal_age
        info["steering_mode"] = "stanley_0924" if avoidance_active else "pp_stanleyff"
        if (
            self.avoidance_stanley_enabled
            and self.avoidance_require_signal
            and not avoidance_signal_fresh
        ):
            why = "MISSING" if self.avoidance_mode_rx_ns is None else "STALE"
            return stop(f"AVOIDANCE_MODE_SIGNAL_{why} topic={self.avoidance_mode_topic}")

        heading_problem = self._heading_problem(now_ns)
        if not self.tracking_started:
            blockers = self._start_blockers(heading_problem)
            if blockers:
                return stop("NOT_READY " + " | ".join(blockers))
            self._start_tracking(now_ns, pose, heading_problem)
        if self._heading_aligning and heading_problem is None:
            self._heading_aligning = False
            self.get_logger().warn("HEADING_ALIGN done: the ESKF heading locked, tracking the route")
        if self._heading_aligning:
            return self._align_drive(now_ns, route, pose, dt, info, heading_problem)
        if heading_problem is not None:
            return stop(heading_problem)

        proj = route.project(pose.x, pose.y, pose.yaw, self.route_s, **self.proj_kwargs)
        if proj is None and self.route_s is not None:
            # The pose left the monotonic window (a jump): re-acquire over the
            # whole route rather than stop. Heading disambiguates a near-loop.
            proj = route.project(pose.x, pose.y, pose.yaw, None, **self.proj_kwargs)
            if proj is not None:
                self.get_logger().warn(
                    f"route position re-acquired at s={proj.s:.1f} m (was {self.route_s:.1f} m)",
                    throttle_duration_sec=5.0,
                )
        if proj is None:
            return stop("NO_ROUTE_PROJECTION")
        # proj.distance, not lateral_offset: the offset is measured to the
        # segment's infinite line, so a vehicle 37 m BEHIND the first waypoint
        # but in line with it reads 1.8 m and would pass a lateral-only gate.
        if proj.distance > self.max_lateral and self.join_enabled and self.join_allowed:
            return self._join_route(route, pose, dt, info)
        if proj.distance > self.pose_far:
            # Further than any tracking error can be: the pose is wrong.
            return stop(f"POSE_FAR distance={proj.distance:.1f}m > {self.pose_far:.0f}m off the route")
        if self.join_missions_pending and self.missions is not None:
            self.missions.initialize_at(proj.s, self.front_offset)
            self.join_missions_pending = False
        if proj.distance <= 0.5 * self.max_lateral:
            # Acquired for good. Half the gate, not the gate itself: arriving
            # at an angle the distance hovers around max_lateral for a few
            # cycles, and ending the join there would hand a vehicle still
            # two metres off the route to the normal controller at speed.
            self.join_allowed = False
        if abs(proj.heading_error) > self.max_heading_err:
            return stop(f"WRONG_DIRECTION heading_err={math.degrees(proj.heading_error):+.0f}deg")
        self.route_s = proj.s
        s = proj.s
        v = max(pose.speed_mps, 0.0)
        info["route_s"] = s
        info["waypoint"] = route.index_at_s(s)
        info["lateral"] = proj.lateral_offset
        info["heading_err_deg"] = math.degrees(proj.heading_error)

        speeds = [pose.speed_mps]
        if self.wheel is not None and (now_ns - self.wheel[1]) * 1e-9 < 0.5:
            speeds.append(self.wheel[0])
            if v > 2.0:
                self.wheel_ratio.append(self.wheel[0] / v)
        stopped = self.stopped_tracker.update(now_ns, speeds)
        info["stopped"] = stopped

        if self.missions is None or self.segments is None:
            return stop("MISSIONS_NOT_RESOLVED")
        if self.braking is None:
            return stop("BRAKING_MODEL_UNSET")
        if self.supervisor is None or not self.supervisor.available:
            return stop("CORRIDOR_SUPERVISOR_UNSET")

        seg = self.segments.params_at(s)
        info["segment"] = seg.name

        # ---- missions ------------------------------------------------------
        obs = self.permits.observation(now_ns, self.detector_alive_timeout)
        info["signal_obs"] = obs

        def permit_fn(direction, roi, roi_required):
            return self.permits.decision(direction, now_ns, roi, roi_required)

        # Pause missions stop like the zone brake: standstill NOW, measured envelope.
        still_now = all(abs(x) <= self.stopped_tracker.threshold for x in speeds)
        pause_stop_m = (self.braking.envelope_distance(v) if self.braking is not None else math.inf) \
            + self.pause_stop_margin
        signal_brake_m = (self.braking.required_distance(v) if self.braking is not None else 0.0) + self.signal_pre_detect_margin
        pause_timeouts_before = self.missions.pause_timeouts
        mo = self.missions.step(M.MissionInputs(
            now_ns, s, v, self.front_offset, stopped, permit_fn, self._brake_latched_for, obs.alive, obs.sees_red,
            still_now, pause_stop_m, signal_brake_m,
        ))
        info["mission"] = mo
        if mo.kind == M.PAUSE and (mo.mission_id, mo.state) != getattr(self, "_pause_logged", None):
            self._pause_logged = (mo.mission_id, mo.state)
            if mo.state == M.PAUSE_BRAKE:
                self.get_logger().warn(
                    f"pause {mo.mission_id} brake at s={s:.1f} m, v={v * KMH:.1f} km/h, target "
                    f"{mo.distance_to_stop_control_m:+.1f} m (stopping distance {pause_stop_m:.1f} m): "
                    f"stop, hold {mo.timeout_limit_s:.1f} s")
            elif mo.state == M.PAUSE_GO:
                self.get_logger().warn(f"pause {mo.mission_id} done: {mo.reason}")
        if self.missions.pause_timeouts != pause_timeouts_before:
            self.get_logger().error(
                f"pause brake released after {self.pause_max_brake_s:.1f} s without a standstill "
                f"(v={v * KMH:.1f} km/h) - check the speed inputs / brake")

        # ---- stop targets (enforced with the brake) ------------------------
        stop_targets: list[tuple[float, str]] = []
        if mo.stop_s is not None:
            stop_targets.append((mo.stop_s, f"{mo.mission_id}:{mo.state}"))
        end_control_s = route.length - self.front_offset - (self.stop_buffer or 0.0)
        stop_targets.append((end_control_s, "ROUTE_END"))

        live = self.stopline.estimate(now_ns, self.live_ttl, s)
        info["stopline_live"] = live
        stopline_source = "map"
        if mo.signal_active and mo.stop_control_s is not None and self.live_tol is not None and live.valid:
            d_map = mo.stop_control_s - s
            d_live = live.distance_m - self.front_offset - (self.stop_buffer or 0.0)
            if abs(d_live - d_map) <= self.live_tol:
                self._agree_run += 1
                self._mismatch_run = 0
                if self._agree_run >= 3:
                    self._stopline_mismatch = False
                stopline_source = "live"
                if mo.stop_s is not None:
                    stop_targets.append((s + min(d_live, d_map), "LIVE_STOPLINE"))
            else:
                self._mismatch_run += 1
                self._agree_run = 0
                if self._mismatch_run >= 3:
                    self._stopline_mismatch = True
            if self._stopline_mismatch:
                stop_targets.append((s + min(d_live, d_map), "STOPLINE_MISMATCH"))
                stopline_source = "mismatch"
        elif not live.valid:
            self._mismatch_run = 0
        info["stopline_source"] = stopline_source

        # ---- speed policy ---------------------------------------------------
        d_hold = self.braking.required_distance(self.v_min_moving)  # brake engages here even at crawl
        v_profile = route.speed_at_s(s)
        v_ramp = self.start_ramp.cap(v, pose.x, pose.y)
        caps: list[tuple[float, str]] = [
            (v_profile, "route_profile"),
            (v_ramp, "start_ramp"),
            (self.v_cruise, "cruise"),
            (self.v_max, "max"),
        ]
        if seg.speed_cap_mps is not None:
            caps.append((seg.speed_cap_mps, f"segment:{seg.name}"))
        if mo.speed_cap_mps is not None:
            caps.append((mo.speed_cap_mps, f"mission:{mo.mission_id}"))
        if self.v_cap_unverified is not None:
            caps.append((self.v_cap_unverified, "steering_unverified"))
        # Deceleration is commanded as a STEP (2026-10-01): the comfort ramps
        # only decide where the lower target starts, then it is commanded whole.
        v_other = min(c for c, _ in caps)
        caps.append((anticipatory_step(s, self._upcoming_caps(s, 150.0), self.comfort_decel, v_other), "anticipate_next_zone"))
        # Corners of the route profile: from where coasting from the actual
        # speed at comfort_decel_mps2 reaches the corner speed, the command
        # follows that coast curve down (speed.coast_command ramp) or steps to
        # the corner speed at once (step).
        coast_cap, self._coast_latched = coast_step_cap(
            s, v, self._route_drops, self.comfort_decel, self._coast_latched, ramp_below_mps=self.coast_ramp_below
        )
        caps.append((coast_cap, "coast_to_corner"))
        # The speed trim is off (and reset) from the coast-down step through the
        # corner and its exit ramp: anywhere the profile is below its straight
        # speed. It pushed the command above the corner speed and wound up in
        # the exit ramps (overshooting the next straight, 2026-10-03).
        in_corner = math.isfinite(coast_cap) or v_profile < self._route_vmax - 0.05
        info["in_corner"] = in_corner
        # No acceleration while the body is off the route's direction (a corner
        # run wide): hold the speed it had - the corner speed - until aligned.
        align_err = wrap_angle(route.heading_over_window(s, self.blend.heading_window_m) - pose.yaw)
        # ...AND no acceleration while the wheels are still turned (measured,
        # /encoder): above align_steer_deg, or no fresh reading at all.
        steer_hold = ""
        if self.align_steer_limit > 0.0:
            if self.encoder is None or (now_ns - self.encoder[1]) * 1e-9 > self.align_encoder_timeout_s:
                steer_hold = "ENCODER_STALE"
            else:
                road_wheel = self.calib.road_wheel_from_encoder(self.encoder[0])
                info["steer_road_wheel_deg"] = math.degrees(road_wheel)
                if abs(road_wheel) > self.align_steer_limit:
                    steer_hold = f"STEER {math.degrees(road_wheel):+.1f}deg"
        align_cap, self._align_hold = align_hold_cap(
            align_err, self.align_limit, v, min(c for c, _ in caps), self.v_min_moving, self._align_hold,
            steering_hold=bool(steer_hold),
        )
        heading_off = self.align_limit > 0.0 and abs(align_err) > self.align_limit
        caps.append((align_cap, "align_hold" if heading_off or not steer_hold else f"steer_hold {steer_hold}"))
        info["align_heading_err_deg"] = math.degrees(align_err)
        info["steer_hold"] = steer_hold

        nearest_stop_s, nearest_name = min(stop_targets, key=lambda t: t[0])
        d_avail = nearest_stop_s - s
        v_env = self.braking.speed_limit_for_distance(d_avail)
        caps.append((v_env, f"stoppable:{nearest_name}"))
        v_other = min(c for c, _ in caps)
        comfort = comfort_speed(d_avail - d_hold, self.v_min_moving, self.comfort_decel)
        caps.append((step_cap(comfort, self.v_min_moving, v_other), f"comfort_stop:{nearest_name}"))

        # Permit valid but not committed: stay stoppable for the stop line and
        # drop to the crawl speed in one step where the glide would have begun.
        if mo.approach_stop_s is not None and self.commit_mode == "last_stoppable_point":
            d_appr = mo.approach_stop_s - s
            caps.append((max(self.braking.speed_limit_for_distance(d_appr), self.v_min_moving), "stoppable_approach"))
            comfort = comfort_speed((mo.commit_s or s) - s, self.v_min_moving, self.comfort_decel)
            caps.append((step_cap(comfort, self.v_min_moving, v_other), "comfort_to_commit"))

        # Zone brake region from the map's zone file, on the controller's own
        # route position (waypoint index of the GLOBAL route; on the avoidance
        # splice the index means nothing - count it as inside, no new trigger).
        # Inside [first - lead, last] the brake starts once the zone start is
        # within the measured stopping distance (braking.measured_envelope), so
        # the stop lands at the zone start: not early when slow, inside the
        # zone when fast (accepted).
        on_splice = ":avoid:" in str(route.trajectory_id)
        zone_wp = route.index_at_s(s)
        zone_trigger = not on_splice
        zone_left_m = None
        if self.zone_brake_ranges is not None:
            zone = None if on_splice else zone_at(
                self.zone_brake_ranges, zone_wp, self.zone_brake_lead_wp + self.zone_brake_window_wp)
            self.zone_brake.observe(on_splice or zone is not None, now_ns)
            zone_trigger = False
            if zone is not None:
                # Stop target: lead_waypoints before the zone start (2026-10-10: stopping
                # AT the zone start was too close to the cones).
                target_wp = max(zone[0] - self.zone_brake_lead_wp, 0.0)
                zone_left_m = route.s_at_index(int(min(math.ceil(target_wp), route.count - 1))) - s
                stop_m = (self.braking.envelope_distance(v) if self.braking is not None else math.inf)
                zone_trigger = stop_needed(zone_left_m, stop_m + self.zone_brake_stop_margin)
                info["zone_left_m"] = zone_left_m
                # After the stop, crawl to the zone at lead_speed_kmh (the command
                # guard caps the zone itself).
                to_zone_m = route.s_at_index(min(zone[0], route.count - 1)) - s
                if (not self.zone_brake.armed and to_zone_m > 0.0
                        and self.zone_brake_lead_speed > 0.0):
                    caps.append((self.zone_brake_lead_speed, "zone_lead"))

        v_cap, cap_source = min(caps, key=lambda c: c[0])
        v_cap = max(v_cap, 0.0)
        info["cap_source"] = cap_source
        info["stop_target"] = (nearest_name, d_avail)
        info["v_profile"] = v_profile

        brake = bool(mo.hold_brake)
        stop_reason = "hold" if brake else ""
        # Brake latch: once engaged for a stop target it stays engaged until the
        # target goes away (permit, mismatch cleared). Re-evaluating
        # d <= d_required(v) while decelerating would chatter the brake motor.
        latch_name = nearest_name.split(":")[0]
        if self._brake_latched_for is not None and self._brake_latched_for != latch_name:
            self._brake_latched_for = None
        if d_avail <= self.braking.required_distance(v) or v_env < self.v_min_moving:
            brake = True
            stop_reason = nearest_name
            self._brake_latched_for = latch_name
        elif self._brake_latched_for == latch_name:
            brake = True
            stop_reason = nearest_name + " (latched)"
        # Entering an obstacle-avoidance zone: brake to a standstill, hold it for
        # zone_brake.stop_hold_s (then the usual brake-release hold and start
        # ramp). No slow-down before it. Standstill = every speed source at or
        # below timing.stopped_speed_mps NOW (not the 0.5 s-confirmed `stopped`).
        timeouts_before = self.zone_brake.timed_out
        standing_still = all(abs(x) <= self.stopped_tracker.threshold for x in speeds)
        zone_braking = self.zone_brake_enabled and self.zone_brake.update(
            now_ns, standing_still, can_trigger=zone_trigger
        )
        if self.zone_brake.timed_out != timeouts_before:
            self.get_logger().error(
                f"zone brake released after {self.zone_brake.max_brake_ns * 1e-9:.1f} s without a "
                f"standstill (v={v * KMH:.1f} km/h) - check the speed inputs / brake"
            )
        info["zone_brake"] = zone_braking
        if zone_braking:
            brake = True
            stop_reason = "zone_brake (avoidance zone entry)"
            if self.zone_brake.triggers != getattr(self, "_zone_brake_logged", 0):
                self._zone_brake_logged = self.zone_brake.triggers
                self.get_logger().warn(
                    f"avoidance zone brake at wp {zone_wp:.1f} (s={s:.1f} m, stop target "
                    f"{'n/a' if zone_left_m is None else f'{zone_left_m:+.1f} m'}), v={v * KMH:.1f} km/h: brake to a stop, "
                    f"hold {self.zone_brake.stop_hold_ns * 1e-9:.1f} s"
                )

        releasing = self._brake_release_hold(now_ns, brake)

        trim_active = False
        v_target = 0.0
        if brake or releasing:
            if releasing:
                stop_reason = "brake_release_hold"
        else:
            v_target = v_cap
            if v_target < self.v_min_moving:
                if v_cap >= self.v_min_moving:
                    v_target = self.v_min_moving  # the drive cannot hold a target below the crawl speed
                else:
                    v_target = 0.0
                    brake = True
                    self._braking_last = True
                    stop_reason = f"speed_floor ({cap_source})"
            trim_active = (not brake) and cap_source.startswith(TRIM_ACTIVE_PREFIXES) and not in_corner
        # The speed command is issued after the corridor supervisor below: it
        # may take the speed over (0, or the crawl) and the trim must not
        # integrate against a target that is not being commanded.
        info["ramping"] = cap_source == "start_ramp"

        # ---- steering: PP + Stanley blend on the Global Path ---------------
        steer_route = route
        steer_s = s
        source = "global"
        if avoidance_active and avoidance_path_ready:
            aproj = self.avoidance_route.project(pose.x, pose.y, pose.yaw, None)
            # Integration has already validated and selected this local path.
            # Do not let a second, arbitrary join-distance limit override that
            # decision: a valid avoidance route must be followed even when its
            # initial lateral offset is greater than the normal 2 m route-join
            # limit.  Only a path that cannot be projected at all falls back.
            if aproj is not None:
                steer_route, steer_s = self.avoidance_route, aproj.s
                source = "direct_avoidance_trajectory"
            else:
                # Do not drop back to PP+StanleyFF in a marked avoidance zone
                # merely because a newly received local path cannot be
                # projected yet. Stay on 0924 Stanley and use the map route.
                self.avoidance_front_s = None
                info["avoidance_path_ready"] = False
        elif mo.kind == M.OBSTACLE and self.avoidance_route is not None \
                and now_ns <= self.avoidance_route.valid_until_ns \
                and self.avoidance_route.frame_id == route.frame_id:
            aproj = self.avoidance_route.project(pose.x, pose.y, pose.yaw, None)
            if aproj is not None and abs(aproj.lateral_offset) <= self.max_lateral:
                steer_route, steer_s = self.avoidance_route, aproj.s
                source = "avoidance_trajectory"
        if avoidance_active and source == "global":
            source = "global_avoidance_baseline"
        if source != self.steering_source:
            self.front_s = None
            self.avoidance_front_s = None
            self.steering_source = source
        info["steering_source"] = source

        # Pure Pursuit from the rear axle.
        # The straight value grows with speed (lookahead_straight_time_s), the
        # curvature rule shortens it toward the curve value, the floor holds it.
        straight_ld = time_floored_lookahead(seg.lookahead_straight_m, v, self.lookahead_straight_time)
        kmax = steer_route.max_abs_curvature(steer_s, steer_s + straight_ld)
        ld = time_floored_lookahead(
            curvature_lookahead(kmax, straight_ld, seg.lookahead_curve_m, seg.straight_radius_m, seg.curve_radius_m),
            v,
            self.lookahead_time,
        )
        tx, ty, ts, reached_end = steer_route.point_ahead(steer_s, ld)
        bx, by = to_body_frame(pose.x, pose.y, pose.yaw, tx, ty)
        delta_pp = pure_pursuit_steering(bx, by, self.wheelbase)
        info["lookahead"] = (tx, ty, ld)

        # Stanley terms from the front axle (pose is already the rear axle).
        fx, fy = front_axle_position(pose.x, pose.y, pose.yaw, 0.0, self.wheelbase)
        fproj = steer_route.project(fx, fy, pose.yaw, self.front_s, **self.proj_kwargs)
        if fproj is None:
            fproj = steer_route.project(fx, fy, pose.yaw, None, **self.proj_kwargs)
        if fproj is None and not avoidance_active:
            return stop("NO_FRONT_AXLE_PROJECTION")
        bd = None
        if fproj is not None:
            self.front_s = fproj.s
            route_yaw = steer_route.heading_over_window(fproj.s, self.blend.heading_window_m)
            heading_error = wrap_angle(route_yaw - pose.yaw)
            kappa = steer_route.curvature_at_s(steer_s)
            bd = compute_steering(
                delta_pp, heading_error, fproj.lateral_offset, v, kappa, pose.yaw_rate, self.blend
            )
            info["steering"] = bd

        if avoidance_active:
            # This is the cone_avoidance_0924_3 Stanley calculation.  Its
            # reference is a FRONT-AXLE path, so project the latency-predicted
            # front axle rather than the rear-axle point used by Pure Pursuit.
            predicted = predict_front_axle_pose(
                fx,
                fy,
                pose.yaw,
                v,
                self.last_delta,
                self.wheelbase,
                self.avoidance_latency,
            )
            aproj = steer_route.project(
                predicted.x,
                predicted.y,
                predicted.yaw,
                self.avoidance_front_s,
                **self.proj_kwargs,
            )
            if aproj is None:
                aproj = steer_route.project(
                    predicted.x, predicted.y, predicted.yaw, None, **self.proj_kwargs
                )
            if aproj is None:
                return stop("NO_AVOIDANCE_FRONT_AXLE_PROJECTION")
            self.avoidance_front_s = aproj.s
            avoidance_route_yaw = steer_route.heading_over_window(
                aproj.s, self.avoidance_heading_window
            )
            avoidance_heading_error = wrap_angle(avoidance_route_yaw - predicted.yaw)
            avoidance_heading_term = self.avoidance_k_heading * avoidance_heading_error
            avoidance_cross_track_term = -math.atan2(
                self.avoidance_k_stanley * aproj.lateral_offset,
                self.avoidance_k_soft + abs(v),
            )
            avoidance_raw = steering_angle(
                avoidance_heading_error,
                aproj.lateral_offset,
                v,
                self.avoidance_k_stanley,
                self.avoidance_k_soft,
                self.avoidance_k_heading,
            )
            avoidance_limited = min(
                max(avoidance_raw, -self.avoidance_max_rw), self.avoidance_max_rw
            )
            desired = self.calib.clamp_road_wheel(avoidance_limited)
            steering_rate = min(
                seg.max_road_wheel_rate_rad_s, self.avoidance_max_rw_rate
            )
            info["avoidance_stanley"] = {
                "predicted_x": predicted.x,
                "predicted_y": predicted.y,
                "predicted_yaw": predicted.yaw,
                "projected_s": aproj.s,
                "cross_track_error": aproj.lateral_offset,
                "heading_error": avoidance_heading_error,
                "heading_term": avoidance_heading_term,
                "cross_track_term": avoidance_cross_track_term,
                "delta_raw": avoidance_raw,
                "delta_limited": desired,
            }
            info["active_cross_track"] = aproj.lateral_offset
            info["active_heading_error"] = avoidance_heading_error
        else:
            desired = self.calib.clamp_road_wheel(bd.delta)
            steering_rate = seg.max_road_wheel_rate_rad_s
            info["active_cross_track"] = bd.cross_track_error
            info["active_heading_error"] = bd.heading_error
        delta_nom = rate_limited(self.last_delta, desired, steering_rate, dt)

        # ---- corridor supervisor (uses the same section lookahead) ---------
        sp = self.supervisor.p
        sp.lookahead_straight_m = seg.lookahead_straight_m
        sp.lookahead_curve_m = seg.lookahead_curve_m
        sp.lookahead_straight_radius_m = seg.straight_radius_m
        sp.lookahead_curve_radius_m = seg.curve_radius_m
        sp.max_steering_rate_rad_s = steering_rate
        stopping_time = self.braking.total_latency_s + v / self.braking.safe_decel_mps2
        decision = self.supervisor.evaluate(
            route, pose.x, pose.y, pose.yaw, v, s, delta_nom, self.last_delta, self.rw_limits, stopping_time
        )
        info["corridor"] = decision
        delta = decision.steering_rad
        state = mo.state
        reason = mo.reason
        # The supervisor never brakes (2026-10-01): while every candidate hits
        # a wall it commands speed 0 without the brake and steers the candidate
        # with the most clearance - except from STANDSTILL, where it commands
        # the crawl instead (a vehicle that does not move cannot steer itself
        # back, sim S3); wholly outside the lane it crawls back on the nominal
        # steering. A stop target (the brake) always wins.
        corridor_owner = ""
        if not brake and not releasing and decision.outside:
            v_target = min(v_target, self.v_outside_lane)
            trim_active = False
            state = "LANE_OUTSIDE"
            corridor_owner = (
                f"LANE_OUTSIDE crawl {self.v_outside_lane * KMH:.0f} km/h "
                f"(clearance {decision.current_clearance:+.2f} m)"
            )
        elif not brake and not releasing and not decision.ok:
            # Once it has come to a stop it crawls, and keeps crawling until a
            # candidate clears - not 0 again as soon as it moves.
            if v <= self.stopped_tracker.threshold:
                self._corridor_crawl = True
            v_target = min(v_target, self.v_outside_lane) if self._corridor_crawl else 0.0
            trim_active = False
            state = "CORRIDOR_HOLD"
            corridor_owner = (
                f"CORRIDOR {decision.reason}: "
                + (f"crawl {self.v_outside_lane * KMH:.0f} km/h until a candidate clears" if self._corridor_crawl else "speed 0")
                + f" without brake, steering {math.degrees(decision.candidate_offset_rad):+.0f} deg from nominal"
            )
        if decision.ok:
            self._corridor_crawl = False
        if brake:
            v_cmd = 0.0
            reason = f"stop for {stop_reason} (d={d_avail:.2f}m) | {mo.reason}"
        elif releasing:
            v_cmd = 0.0
            reason = f"brake release hold | {mo.reason}"
        else:
            if in_corner:
                self.speed_trim.reset()
            v_cmd = self.speed_trim.command(v_target, v, dt, active=trim_active)
            if corridor_owner:
                reason = f"{corridor_owner} | {mo.reason}"
        info["trim_mps"] = self.speed_trim.trim_mps
        info["trim_active"] = trim_active
        self.last_delta = delta
        cmd = self.calib.command(delta)
        if brake:
            self.last_speed_cmd = 0.0
        return CycleResult(not brake, brake, v_cmd, delta, cmd, state, reason, info)

    # ------------------------------------------------------------ reporting
    def _publish_status(self, r: CycleResult, pose: PoseSample | None, now_ns: int) -> None:
        mode = "ACTIVE" if self.tracking_started else "WAITING"
        mo = r.info.get("mission")
        wp = r.info.get("waypoint")
        line = f"{mode} {r.state} {r.reason} | map={self._profile_label()} speed_mode={self.speed_mode}"
        if wp is not None:
            line += f" wp={wp:.1f}"
        if pose is not None:
            line += f" v={pose.speed_mps * KMH:.1f}km/h"
        bd = r.info.get("steering")
        avoidance = r.info.get("avoidance_stanley")
        if avoidance is not None:
            line += (
                " | mode=STANLEY_0924"
                f" raw={math.degrees(avoidance['delta_raw']):+.1f}"
                f" (hdg={math.degrees(avoidance['heading_term']):+.1f}"
                f" cte={math.degrees(avoidance['cross_track_term']):+.1f})"
                f" cte={avoidance['cross_track_error']:+.2f}m"
            )
        elif bd is not None:
            line += (
                f" | mode=PP_STANLEYFF w={bd.weight:.2f} pp={math.degrees(bd.delta_pp):+.1f} st={math.degrees(bd.delta_stanley):+.1f}"
                f" (ff={math.degrees(bd.feedforward):+.1f} hdg={math.degrees(bd.heading_term):+.1f} cte={math.degrees(bd.cross_track_term):+.1f})"
                f" yaw={math.degrees(bd.yaw_damping):+.1f} cte={bd.cross_track_error:+.2f}m"
            )
        elif r.info.get("lateral") is not None:
            line += f" cte={r.info['lateral']:+.2f}m"
        line += f" | cmd v={0.0 if r.brake else r.speed_mps * KMH:.1f}km/h steer={math.degrees(r.steering_rw):+.1f}deg " \
                f"angle={r.command_rad:+.2f}rad brake={int(r.brake)}"
        if r.info.get("cap_source"):
            line += f" cap={r.info['cap_source']}"
        if r.info.get("trim_active"):
            line += f" trim=+{r.info.get('trim_mps', 0.0) * KMH:.1f}"
        if r.info.get("ramping"):
            line += " ramp"
        if r.info.get("segment"):
            line += f" seg={r.info['segment']}"
        dec = r.info.get("corridor")
        if dec is not None and dec.fallback_fraction > 0.0:
            line += f" fallback={dec.fallback_fraction * 100:.0f}%"
        if mo is not None and mo.timeout_wait_s is not None:
            line += f" wait={mo.timeout_wait_s:.0f}/{mo.timeout_limit_s:.0f}s"
        line += f" cycle={self.cycle_time_ms:.1f}ms"
        self.pub_status.publish(String(data=line))
        head = f"{mode}:{r.state}:{r.reason.split(' ')[0]}"
        if head != self.last_status_head:
            self.last_status_head = head
            self.get_logger().info(line)

        stamp = self.get_clock().now().to_msg()
        if bd is not None or avoidance is not None:
            lookahead = r.info.get("lookahead")
            steering_debug = {
                "stamp_ns": now_ns,
                "steering_mode": r.info.get("steering_mode", "pp_stanleyff"),
                "delta_out_deg": math.degrees(r.steering_rw), "command_rad": r.command_rad,
                "cross_track_rear_m": r.info.get("lateral"),
                "lookahead_m": None if lookahead is None else lookahead[2],
                "corridor_offset_deg": None if dec is None else math.degrees(dec.candidate_offset_rad),
                "fallback_fraction": None if dec is None else dec.fallback_fraction,
                "speed_mps": None if pose is None else pose.speed_mps, "v_profile_mps": r.info.get("v_profile"),
                "trim_mps": r.info.get("trim_mps"), "cap_source": r.info.get("cap_source"),
                "align_heading_err_deg": r.info.get("align_heading_err_deg"),
                "steering_source": r.info.get("steering_source"),
                "avoidance_requested": r.info.get("avoidance_mode_requested"),
                "avoidance_signal_fresh": r.info.get("avoidance_signal_fresh"),
                "avoidance_signal_age_s": r.info.get("avoidance_signal_age_s"),
            }
            if bd is not None:
                steering_debug.update({
                    "weight": bd.weight, "curvature": bd.curvature,
                    "delta_pp_deg": math.degrees(bd.delta_pp),
                    "delta_stanley_deg": math.degrees(bd.delta_stanley),
                    "feedforward_deg": math.degrees(bd.feedforward),
                    "heading_term_deg": math.degrees(bd.heading_term),
                    "cross_track_term_deg": math.degrees(bd.cross_track_term),
                    "yaw_damping_deg": math.degrees(bd.yaw_damping),
                    "delta_blend_deg": math.degrees(bd.delta),
                    "heading_error_deg": math.degrees(bd.heading_error),
                    "cross_track_front_m": bd.cross_track_error,
                })
            if avoidance is not None:
                steering_debug.update({
                    "avoidance_delta_raw_deg": math.degrees(avoidance["delta_raw"]),
                    "avoidance_delta_limited_deg": math.degrees(avoidance["delta_limited"]),
                    "avoidance_heading_error_deg": math.degrees(avoidance["heading_error"]),
                    "avoidance_heading_term_deg": math.degrees(avoidance["heading_term"]),
                    "avoidance_cross_track_term_deg": math.degrees(avoidance["cross_track_term"]),
                    "avoidance_cross_track_m": avoidance["cross_track_error"],
                    "avoidance_predicted_x": avoidance["predicted_x"],
                    "avoidance_predicted_y": avoidance["predicted_y"],
                    "avoidance_predicted_yaw_deg": math.degrees(avoidance["predicted_yaw"]),
                    "avoidance_projected_s": avoidance["projected_s"],
                })
            self.pub_steering.publish(String(data=json.dumps(steering_debug)))
            active_cte = r.info.get("active_cross_track")
            active_heading = r.info.get("active_heading_error")
            if active_cte is not None:
                self.pub_cte.publish(Float32(data=float(active_cte)))
            if active_heading is not None:
                self.pub_heading_err.publish(Float32(data=float(active_heading)))
        if mo is not None:
            permit = mo.permit
            obs = r.info.get("signal_obs")
            self.pub_mission.publish(String(data=json.dumps({
                "stamp_ns": now_ns, "state": mo.state, "mission": mo.mission_id, "kind": mo.kind,
                "direction": mo.direction, "route_s": r.info.get("route_s"), "waypoint": wp,
                "speed_cap_kmh": None if mo.speed_cap_mps is None else mo.speed_cap_mps * KMH,
                "stop_s": mo.stop_s, "stop_control_s": mo.stop_control_s, "d_stop_control": mo.distance_to_stop_control_m,
                "approach_stop_s": mo.approach_stop_s, "commit_s": mo.commit_s, "permit_source": mo.permit_source,
                "timeout_wait_s": mo.timeout_wait_s, "timeout_limit_s": mo.timeout_limit_s,
                "hold_brake": mo.hold_brake, "reason": mo.reason, "end_s": mo.end_s,
                "stop_target": r.info.get("stop_target"), "stopped": r.info.get("stopped"),
                "segment": r.info.get("segment"), "cap_source": r.info.get("cap_source"),
                "detector_alive": None if obs is None else obs.alive, "sees_red": None if obs is None else obs.sees_red,
                "map": None if self.profile is None else self.profile.label(),
            })))
            if permit is not None:
                self.pub_permit.publish(String(data=json.dumps({
                    "stamp_ns": now_ns, "direction": mo.direction, "valid": permit.valid, "reason": permit.reason,
                    "fresh_age_s": permit.fresh_age_s, "json_age_s": permit.json_age_s, "confirmed": permit.confirmed,
                    "target_x": permit.target_x, "target_y": permit.target_y, "selection": permit.selection,
                    "state": permit.state, "ttl_s": self.permits.ttl_s, "source": mo.permit_source,
                })))
        live = r.info.get("stopline_live")
        if live is not None:
            self.pub_stopline.publish(String(data=json.dumps({
                "stamp_ns": now_ns, "valid": live.valid, "distance_m": live.distance_m, "age_s": live.age_s,
                "reason": live.reason, "source_used": r.info.get("stopline_source"),
                "rejected": self.stopline.rejected_last,
            })))
        if dec is not None:
            self.pub_corridor.publish(String(data=json.dumps({
                "stamp_ns": now_ns, "ok": dec.ok, "reason": dec.reason, "current_clearance_m": dec.current_clearance,
                "predicted_min_clearance_m": dec.predicted_min_clearance, "covered": dec.covered,
                "candidate_offset_deg": math.degrees(dec.candidate_offset_rad),
                "fallback_fraction": dec.fallback_fraction, "candidates_tried": dec.candidates_tried,
                "outside": dec.outside, "lane_map": self.static_corridor is not None,
            })))
        la = r.info.get("lookahead")
        if la is not None and self.route is not None:
            pt = PointStamped()
            pt.header.stamp = stamp
            pt.header.frame_id = self.route.frame_id
            pt.point.x, pt.point.y = float(la[0]), float(la[1])
            self.pub_lookahead.publish(pt)
        if now_ns >= self.marker_next_ns:
            self.marker_next_ns = now_ns + int(float(self.get_parameter("marker_period_s").value) * 1e9)
            self._publish_markers(r, pose, stamp)
        self._publish_diagnostics(r, pose, stamp)

    def _publish_diagnostics(self, r: CycleResult, pose, stamp) -> None:
        status = DiagnosticStatus()
        status.name = PACKAGE_NAME
        status.hardware_id = "vehicle"
        if not self.tracking_started:
            status.level = DiagnosticStatus.WARN
        elif r.brake and r.state == "FAILSAFE_STOP":
            status.level = DiagnosticStatus.ERROR
        elif r.state in ("CORRIDOR_HOLD", "LANE_OUTSIDE", "HEADING_ALIGN"):
            status.level = DiagnosticStatus.WARN
        else:
            status.level = DiagnosticStatus.OK
        status.message = f"{r.state} {r.reason}"
        values = {
            "tracking": str(self.tracking_started), "pose_source": str(r.info.get("pose_source")),
            "map": self._profile_label(), "speed_mode": self.speed_mode,
            "route_s": f"{r.info.get('route_s', float('nan')):.2f}", "cycle_ms": f"{self.cycle_time_ms:.1f}",
            "segment": str(r.info.get("segment")), "cap_source": str(r.info.get("cap_source")),
            "heading_verified": str(self.heading_verified), "heading_source": str(self.heading_source),
            "steering_mode": str(r.info.get("steering_mode", "pp_stanleyff")),
            "avoidance_requested": str(r.info.get("avoidance_mode_requested", False)),
            "avoidance_signal_fresh": str(r.info.get("avoidance_signal_fresh", False)),
            "avoidance_signal_age_s": str(r.info.get("avoidance_signal_age_s")),
            "readiness_blocking": str(len(self.readiness.blocking)),
            "warnings": " | ".join(self.readiness.warnings + self.mission_warnings)[:500],
        }
        if len(self.wheel_ratio) >= 50:
            ratio = float(np.median(self.wheel_ratio))
            values["wheel_over_eskf_ratio"] = f"{ratio:.2f}"
            if 3.0 <= ratio <= 4.2:
                values["wheel_unit_warning"] = "wheel speed looks like km/h (x3.6): set timing.wheel_speed_scale_to_mps=0.2777778"
        if self.encoder is not None:
            values["encoder_rad"] = f"{self.encoder[0]:+.3f}"
            values["encoder_minus_cmd_rad"] = f"{self.encoder[0] - self.last_cmd_angle:+.3f}"
            values["encoder_road_wheel_deg"] = f"{math.degrees(self.calib.road_wheel_from_encoder(self.encoder[0])):+.1f}"
        status.values = [KeyValue(key=k, value=v) for k, v in values.items()]
        arr = DiagnosticArray()
        arr.header.stamp = stamp
        arr.status = [status]
        self.pub_diag.publish(arr)

    def _marker(self, frame, ns, mid, mtype, stamp, rgba, scale=0.1) -> Marker:
        m = Marker()
        m.header.frame_id = frame
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = scale
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish_markers(self, r: CycleResult, pose, stamp) -> None:
        if self.route is None or pose is None:
            return
        frame = self.route.frame_id
        arr = MarkerArray()
        la = r.info.get("lookahead")
        if la is not None:
            m = self._marker(frame, "global_lookahead", 0, Marker.SPHERE, stamp, (0.0, 0.9, 1.0, 1.0), 0.6)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(la[0]), float(la[1]), 0.3
            arr.markers.append(m)
        dec = r.info.get("corridor")
        fp = footprint_corners(pose.x, pose.y, pose.yaw, self.front_offset, self.rear_offset, self.half_width)
        fp_color = (1.0, 0.2, 0.2, 1.0) if dec is not None and dec.outside else (1.0, 1.0, 0.0, 1.0)
        m = self._marker(frame, "footprint", 0, Marker.LINE_STRIP, stamp, fp_color, 0.08)
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.2) for p in list(fp) + [fp[0]]]
        arr.markers.append(m)
        if self.front_s is not None:
            fx, fy = self.route.point_at_s(self.front_s)
            m = self._marker(frame, "front_axle_foot", 0, Marker.SPHERE, stamp, (1.0, 0.2, 1.0, 1.0), 0.35)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(fx), float(fy), 0.3
            arr.markers.append(m)
        if dec is not None and dec.predicted_poses:
            # The chosen rollout, bold, and a ghost of the footprint where it ends.
            color = (0.2, 1.0, 0.2, 0.9) if dec.ok else (1.0, 0.2, 0.2, 0.9)
            m = self._marker(frame, "predicted_path", 0, Marker.LINE_STRIP, stamp, color, 0.08)
            m.points = [Point(x=float(px), y=float(py), z=0.25) for px, py, _ in dec.predicted_poses]
            arr.markers.append(m)
            gx, gy, gyaw = dec.predicted_poses[-1]
            ghost = footprint_corners(gx, gy, gyaw, self.front_offset, self.rear_offset, self.half_width)
            m = self._marker(frame, "predicted_footprint", 0, Marker.LINE_STRIP, stamp, color, 0.06)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.25) for p in list(ghost) + [ghost[0]]]
            arr.markers.append(m)
        if dec is not None and len(dec.rollouts) > 1:
            # Every candidate tried this cycle, faint. They outlive the cycle by
            # CANDIDATE_MARKER_LIFETIME_S so a one-cycle search is not missed.
            lifetime = Duration(seconds=CANDIDATE_MARKER_LIFETIME_S).to_msg()
            for i, ro in enumerate(dec.rollouts):
                if not ro.poses:
                    continue
                faint = (0.4, 1.0, 0.4, 0.35) if ro.ok else (1.0, 0.4, 0.4, 0.35)
                m = self._marker(frame, "candidate_paths", i, Marker.LINE_STRIP, stamp, faint, 0.03)
                m.lifetime = lifetime
                m.points = [Point(x=float(px), y=float(py), z=0.22) for px, py, _ in ro.poses]
                arr.markers.append(m)
        mo = r.info.get("mission")
        if mo is not None and mo.stop_control_s is not None:
            x, y = self.route.point_at_s(mo.stop_control_s)
            m = self._marker(frame, "stop_control_point", 0, Marker.CYLINDER, stamp, (1.0, 0.3, 0.0, 0.9), 0.5)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.25
            arr.markers.append(m)
            ax, ay = self.route.point_at_s(mo.stop_anchor_s)
            m = self._marker(frame, "stop_anchor", 0, Marker.CUBE, stamp, (1.0, 0.0, 0.0, 0.9), 0.4)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(ax), float(ay), 0.2
            arr.markers.append(m)
            if mo.commit_s is not None:
                cx, cy = self.route.point_at_s(mo.commit_s)
                m = self._marker(frame, "commit_point", 0, Marker.SPHERE, stamp, (0.2, 1.0, 0.2, 0.9), 0.45)
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(cx), float(cy), 0.25
                arr.markers.append(m)
        self.pub_markers.publish(arr)

    def _publish_corridor_markers(self) -> None:
        if self.route is None:
            return
        stamp = self.get_clock().now().to_msg()
        frame = self.route.frame_id
        arr = MarkerArray()
        walls = [("route_center", self.route.xy, (1.0, 1.0, 1.0, 0.6), 0.05)]
        if self.static_corridor is not None:
            walls += [("lane_left", self.static_corridor.left, (1.0, 0.85, 0.1, 0.9), 0.12),
                      ("lane_right", self.static_corridor.right, (0.3, 0.7, 1.0, 0.9), 0.12)]
        if self.fallback_corridor is not None:
            walls += [("route_left", self.fallback_corridor.left, (0.6, 0.6, 0.6, 0.5), 0.06),
                      ("route_right", self.fallback_corridor.right, (0.6, 0.6, 0.6, 0.5), 0.06)]
        for mid, (name, poly, rgba, width) in enumerate(walls):
            m = self._marker(frame, "corridor_" + name, mid, Marker.LINE_STRIP, stamp, rgba, width)
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.15) for p in poly]
            arr.markers.append(m)
        self.pub_corridor_markers.publish(arr)

    def _stall_check(self) -> None:
        if self.control_cycles != self.cycles_at_check:
            self.cycles_at_check = self.control_cycles
            return
        self.cycles_at_check = self.control_cycles
        self.get_logger().error(
            "control loop has not run for 5 s: with use_sim_time nothing may be advancing /clock; "
            "no command is being published"
        )

    def destroy_node(self):
        # The TransformListener owns a spinning thread; stop it explicitly so
        # the process can exit on SIGTERM instead of hanging at teardown.
        try:
            self.tf_listener.unregister()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):  # Ctrl+C, or the launch's SIGINT
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            # Never leave a controller process alive after shutdown: a lingering
            # executor thread must not keep publishing to the actuator topics.
            os._exit(0)


if __name__ == "__main__":
    main()
