import math
from pathlib import Path

import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers PoseStamped transform support
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.duration import Duration
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from tf2_ros import TransformException
from visualization_msgs.msg import Marker, MarkerArray

from mola_slam_mapping.route_io import (
    load_route_files,
    off_track_reason,
    point_ahead_on_route,
    nearest_waypoint_id,
    RouteTracker,
    route_to_path,
)
from mola_slam_mapping.route_visualization import (
    build_route_marker_array,
    transform_route,
)
from mola_slam_mapping.traffic_detect_zones import (
    AVOIDANCE,
    SCHOOL_ZONE,
    SLOW_DOWN,
    TRAFFIC_DETECT,
    ZONE_KINDS,
    check_ranges_fit_route,
    parse_index_ranges,
    zone_containing,
)


class LocalizationMonitorNode(Node):
    """Loads a waypoints.csv from a previous mapping run (see waypoint_saver_node)
    and, for each new fused pose, reports the vehicle's signed lateral offset from
    the saved centerline and whether it is still within the saved drivable
    corridor - i.e. "load the map, compare against the current vehicle position"
    for a localization run.

    It also owns the mission zones: waypoint-index ranges of that route, one
    list per kind (traffic_detect_zones.ZONE_KINDS, each from its own
    <map_dir>/<kind>.yaml). Traffic-light detection zones are drawn red in RViz
    and reported on `traffic_detect_space`, slow-down zones orange on
    `slow_down_space`, obstacle-avoidance zones sky blue on `avoidance_space`
    (std_msgs/Bool each, true while the vehicle's projection onto the route
    lies inside a zone of that kind, false otherwise, at pose rate)."""

    def __init__(self):
        super().__init__("localization_monitor_node")

        self.declare_parameter("input_odom_topic", "/state_estimation/pose")
        self.declare_parameter("utm_frame", "utm")
        self.declare_parameter("visualization_frame", "map")
        self.declare_parameter("waypoints_csv", "")
        self.declare_parameter("corridor_csv", "")
        self.declare_parameter("corridor_half_width_m", 3.0)
        self.declare_parameter("lateral_offset_topic", "lateral_offset")
        self.declare_parameter("on_track_topic", "on_track")
        # waypoints.csv idx of the waypoint nearest the vehicle, every pose;
        # -1 while no route is loaded.
        self.declare_parameter("waypoint_id_topic", "waypoint_id")
        self.declare_parameter("markers_topic", "localization_monitor_markers")
        self.declare_parameter("route_markers_topic", "waypoint_markers")
        # The route as data rather than as drawing: position + recorded heading,
        # pose i == waypoints.csv idx i. Latched, so a follower started at any
        # time receives it immediately.
        self.declare_parameter("reference_path_topic", "reference_path")
        # Pure pursuit's `ld`: how far ahead ALONG THE ROUTE the follow target
        # sits. Arc length, not a straight-line radius, so through a corner the
        # target stays on the route instead of cutting it.
        self.declare_parameter("lookahead_distance_m", 5.0)
        # Route position continuity (RouteTracker): after the first pose only
        # the route from back_m behind to forward_m ahead of the last position
        # is searched, so a route that crosses itself or a two-way road whose
        # legs run side by side cannot hand the vehicle (and every zone) to the
        # other pass. Farther than relocate_distance_m from that window, or
        # heading more than 90 deg off it (opposite leg) = the pose jumped: the
        # whole route is searched again.
        self.declare_parameter("route_search_back_m", 8.0)
        self.declare_parameter("route_search_forward_m", 30.0)
        self.declare_parameter("route_relocate_distance_m", 5.0)
        self.declare_parameter("lookahead_topic", "lookahead_point")
        self.declare_parameter("lookahead_marker_topic", "lookahead_marker")
        # Repeat the route/corridor markers this often even when nothing about
        # them changed. Changing RViz's Fixed Frame calls reset() on every topic
        # display and does not resubscribe (rviz_common/ros_topic_display.hpp),
        # so this route - published once, transient-local - would otherwise be
        # wiped for good the moment the operator switches to a base_link view.
        # 0 disables the repeat.
        self.declare_parameter("route_republish_interval_sec", 2.0)
        # Per zone kind, "first-last" waypoint-index ranges, inclusive, e.g.
        # ["170-180"]: traffic_detect_zones, slow_down_zones, avoidance_zones.
        # No default on purpose: rclpy types an empty list as BYTE_ARRAY and
        # would then refuse the yaml's strings; left unset simply means no
        # zones. <topic>_topic renames that kind's Bool, e.g.
        # "traffic_detect_space_topic", "traffic_detect_space".
        for kind in ZONE_KINDS:
            self.declare_parameter(kind.parameter, Parameter.Type.STRING_ARRAY)
            self.declare_parameter(f"{kind.topic}_topic", kind.topic)

        self._utm_frame = self.get_parameter("utm_frame").value
        self._visualization_frame = self.get_parameter("visualization_frame").value
        self._corridor_half_width = float(self.get_parameter("corridor_half_width_m").value)
        self._lookahead_distance = float(
            self.get_parameter("lookahead_distance_m").value
        )
        if not (
            math.isfinite(self._lookahead_distance) and self._lookahead_distance >= 0.0
        ):
            raise RuntimeError(
                "lookahead_distance_m must be finite and non-negative, got "
                f"{self._lookahead_distance}"
            )
        # The follow target itself, kept so it can be read directly and not only
        # off the topic. None until the first pose projects onto the route.
        self.ld = None
        self._lookahead_at_end = False

        csv_path = self.get_parameter("waypoints_csv").value
        if not csv_path:
            raise RuntimeError(
                "localization_monitor_node requires 'waypoints_csv' (the "
                "waypoints.csv saved by a previous mapping run)"
            )
        corridor_path = self.get_parameter("corridor_csv").value
        loaded_route = load_route_files(
            Path(csv_path).expanduser(),
            Path(corridor_path).expanduser() if corridor_path else None,
            self._corridor_half_width,
        )
        self._waypoints = loaded_route.points
        self._route_tracker = RouteTracker(
            self._waypoints,
            back_m=float(self.get_parameter("route_search_back_m").value),
            forward_m=float(self.get_parameter("route_search_forward_m").value),
            relocate_distance_m=float(self.get_parameter("route_relocate_distance_m").value),
        )
        if not loaded_route.used_saved_corridor:
            self.get_logger().warn(
                "corridor.csv is unavailable; rebuilding a constant-width corridor "
                f"({self._corridor_half_width:.2f}m half-width)"
            )

        self._zones = {}
        for kind in ZONE_KINDS:
            try:
                zone_specs = list(self.get_parameter(kind.parameter).value)
            except ParameterUninitializedException:
                zone_specs = []
            try:
                self._zones[kind] = parse_index_ranges(zone_specs, kind)
                check_ranges_fit_route(
                    self._zones[kind], len(self._waypoints), kind
                )
            except ValueError as error:
                # A zone list that does not fit this route is a zone list for a
                # different map: refuse to run rather than act at the wrong place.
                raise RuntimeError(f"{kind.parameter}: {error}") from error
        self._current_zones = {kind: None for kind in ZONE_KINDS}

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._offset_pub = self.create_publisher(
            Float32, self.get_parameter("lateral_offset_topic").value, 10
        )
        self._on_track_pub = self.create_publisher(
            Bool, self.get_parameter("on_track_topic").value, 10
        )
        self._waypoint_id_pub = self.create_publisher(
            Int32, self.get_parameter("waypoint_id_topic").value, 10
        )
        self._route_republish_interval_ns = int(
            round(
                float(self.get_parameter("route_republish_interval_sec").value) * 1e9
            )
        )
        self._zone_topics = {
            kind: self.get_parameter(f"{kind.topic}_topic").value
            for kind in ZONE_KINDS
        }
        self._zone_pubs = {
            kind: self.create_publisher(Bool, self._zone_topics[kind], 10)
            for kind in ZONE_KINDS
        }
        self._marker_pub = self.create_publisher(
            Marker, self.get_parameter("markers_topic").value, 10
        )
        self._lookahead_pub = self.create_publisher(
            PointStamped, self.get_parameter("lookahead_topic").value, 10
        )
        self._lookahead_marker_pub = self.create_publisher(
            Marker, self.get_parameter("lookahead_marker_topic").value, 10
        )
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._route_marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("route_markers_topic").value,
            latched_qos,
        )
        self._reference_path_pub = self.create_publisher(
            PathMsg,
            self.get_parameter("reference_path_topic").value,
            latched_qos,
        )
        self._route_published = False
        self._last_route_transform = None
        self._last_route_publish_ns = None
        self._route_publish_timer = self.create_timer(
            0.5, self._try_publish_reference_route
        )

        topic = self.get_parameter("input_odom_topic").value
        self.create_subscription(Odometry, topic, self._on_odometry, 10)

        self._last_warn_time = 0.0

        zones_text = "; ".join(
            f"{kind.name} "
            f"{', '.join(zone.label for zone in self._zones[kind]) or 'none'}"
            f" -> '{self._zone_topics[kind]}'"
            for kind in ZONE_KINDS
        )
        self.get_logger().info(
            f"localization_monitor_node: publishing the route as "
            f"'{self._reference_path_pub.topic_name}' (nav_msgs/Path, "
            f"{len(self._waypoints)} poses with heading) and as "
            f"'{self._route_marker_pub.topic_name}' (drawing); follow target on "
            f"'{self._lookahead_pub.topic_name}' {self._lookahead_distance:.1f} m ahead"
        )
        self.get_logger().info(
            f"localization_monitor_node: loaded {len(self._waypoints)} waypoints from "
            f"'{csv_path}', comparing against '{topic}' resolved into frame "
            f"'{self._utm_frame}', corridor half-width {self._corridor_half_width:.2f} m, "
            f"zones: {zones_text}"
        )

    def _route_republish_is_due(self) -> bool:
        if self._route_republish_interval_ns <= 0 or self._last_route_publish_ns is None:
            return False
        elapsed = self.get_clock().now().nanoseconds - self._last_route_publish_ns
        return elapsed >= self._route_republish_interval_ns

    def _try_publish_reference_route(self) -> None:
        republish = self._route_republish_is_due()
        if self._visualization_frame == self._utm_frame:
            if self._route_published and not republish:
                return
            visualization_route = self._waypoints
        else:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._visualization_frame,
                    self._utm_frame,
                    Time(),
                    timeout=Duration(seconds=0.0),
                )
            except TransformException:
                return
            signature = self._transform_signature(transform)
            if not self._route_transform_changed(signature) and not republish:
                return
            visualization_route = transform_route(self._waypoints, transform)

        self._route_marker_pub.publish(
            build_route_marker_array(
                visualization_route,
                self._visualization_frame,
                # Stamp 0 = "use the latest transform". The reference route is
                # published once (transient-local) and never changes, so a real
                # stamp would age out of RViz's ~10 s TF buffer and the whole
                # route would vanish in any view whose Fixed Frame is not the
                # route's own - base_link, say. In a map-fixed view no lookup
                # happens at all, so this costs nothing there.
                Time().to_msg(),
                namespace="reference_route",
                highlight_index_ranges=self._zone_index_pairs(TRAFFIC_DETECT),
                slow_down_index_ranges=self._zone_index_pairs(SLOW_DOWN),
                avoidance_index_ranges=self._zone_index_pairs(AVOIDANCE),
                school_zone_index_ranges=self._zone_index_pairs(SCHOOL_ZONE),
            )
        )
        # Same route, same frame, same schedule - once as drawing, once as data.
        self._reference_path_pub.publish(
            route_to_path(
                visualization_route, self._visualization_frame, Time().to_msg()
            )
        )
        self._route_published = True
        self._last_route_publish_ns = self.get_clock().now().nanoseconds
        if self._visualization_frame != self._utm_frame:
            self._last_route_transform = signature
        elif self._route_republish_interval_ns <= 0:
            # Nothing left to do on this timer: the route never moves in UTM and
            # the repeat that would survive a Fixed Frame change is disabled.
            self._route_publish_timer.cancel()

    def _zone_index_pairs(self, kind) -> list[tuple[int, int]]:
        return [(zone.first, zone.last) for zone in self._zones[kind]]

    @staticmethod
    def _transform_signature(transform):
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

    def _route_transform_changed(self, current) -> bool:
        previous = self._last_route_transform
        if previous is None:
            return True
        translation_changed = any(
            abs(current[index] - previous[index]) > 1e-3
            for index in range(3)
        )
        old_quaternion = previous[3:]
        new_quaternion = current[3:]
        old_norm = math.sqrt(sum(value * value for value in old_quaternion))
        new_norm = math.sqrt(sum(value * value for value in new_quaternion))
        if old_norm < 1e-12 or new_norm < 1e-12:
            rotation_changed = old_quaternion != new_quaternion
        else:
            dot = abs(
                sum(a * b for a, b in zip(old_quaternion, new_quaternion))
                / (old_norm * new_norm)
            )
            rotation_changed = dot < math.cos(math.radians(0.01) / 2.0)
        return translation_changed or rotation_changed

    def _on_odometry(self, msg: Odometry) -> None:
        pose_in = PoseStamped()
        pose_in.header = msg.header
        pose_in.pose = msg.pose.pose

        # Zero timeout: only use the transform if already buffered, same rationale
        # as waypoint_saver_node (georeferencing may not have converged yet).
        try:
            pose_utm = self._tf_buffer.transform(
                pose_in, self._utm_frame, timeout=Duration(seconds=0.0)
            )
            if pose_in.header.frame_id == self._visualization_frame:
                pose_visualization = pose_in
            else:
                pose_visualization = self._tf_buffer.transform(
                    pose_in,
                    self._visualization_frame,
                    timeout=Duration(seconds=0.0),
                )
        except TransformException:
            return

        x = pose_utm.pose.position.x
        y = pose_utm.pose.position.y
        q = pose_utm.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        projection = self._route_tracker.project(x, y, yaw)
        if self._route_tracker.relocated and projection is not None:
            self.get_logger().warning(
                f"pose left the route search window (> "
                f"{self._route_tracker.relocate_distance_m:.1f} m away or heading "
                f"> 90 deg off): whole route searched again -> waypoint "
                f"#{nearest_waypoint_id(projection)}",
                throttle_duration_sec=2.0,
            )
        offset = projection.signed_offset if projection is not None else math.nan
        on_track = projection is not None and projection.on_track

        self._offset_pub.publish(Float32(data=offset))
        self._on_track_pub.publish(Bool(data=on_track))
        self._waypoint_id_pub.publish(Int32(data=nearest_waypoint_id(projection)))
        self._publish_zones(projection)
        self._publish_lookahead(projection, msg.header.stamp)
        self._publish_marker(pose_visualization, on_track)

        if not on_track:
            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_warn_time > 2.0:
                self._last_warn_time = now
                self.get_logger().warn(
                    f"Vehicle off track: {off_track_reason(projection)}"
                )

    def _publish_zones(self, projection) -> None:
        """Per zone kind, true while the vehicle's route position lies inside
        one of that kind's zones.

        Deliberately not gated on on_track: a vehicle a few metres outside the
        corridor next to a traffic light still needs the detector running, and
        a needless true is harmless where a missed true is not. Published as
        false everywhere else, at pose rate, so a consumer can tell "not in a
        zone" from "monitor not running".
        """

        for kind in ZONE_KINDS:
            zone = (
                zone_containing(self._zones[kind], projection.route_position)
                if projection is not None
                else None
            )
            self._zone_pubs[kind].publish(Bool(data=zone is not None))
            current = self._current_zones[kind]
            if zone == current:
                continue
            if zone is not None:
                self.get_logger().info(
                    f"Entered {kind.name} zone {zone.label}: "
                    f"'{self._zone_topics[kind]}' = true"
                )
            else:
                self.get_logger().info(
                    f"Left {kind.name} zone {current.label}: "
                    f"'{self._zone_topics[kind]}' = false"
                )
            self._current_zones[kind] = zone

    def _publish_lookahead(self, projection, stamp) -> None:
        """Publish the path-following target: `ld` metres ahead ALONG the route.

        Computed on the UTM route (where the saved waypoints live) and then
        resolved into the visualization frame, so the topic a controller reads
        and the marker an operator sees are the same point in the same frame as
        the rest of the live stack.

        `stamp` is the pose message's own stamp, NOT this node's wall clock. The
        map->base_link edge is broadcast from these very messages, so a viewer
        resolving this marker into a vehicle frame looks the transform up at
        exactly the time it was published. Wall-clock stamps put the marker on a
        different timeline from TF - days away when replaying a recording -
        which costs nothing in a map-fixed view (frame == fixed frame, no lookup)
        and makes the marker vanish in every other one.
        """

        self.ld = point_ahead_on_route(
            self._waypoints, projection, self._lookahead_distance
        )
        if self.ld is None:
            self._delete_lookahead_marker(stamp)
            return


        target = PointStamped()
        target.header.stamp = stamp
        target.header.frame_id = self._utm_frame
        target.point.x, target.point.y, target.point.z = self.ld.x, self.ld.y, self.ld.z
        try:
            local = self._tf_buffer.transform(
                target, self._visualization_frame, timeout=Duration(seconds=0.0)
            )
        except TransformException:
            return

        self._lookahead_pub.publish(local)
        self._publish_lookahead_marker(local)

        if self.ld.reached_route_end and not self._lookahead_at_end:
            self.get_logger().warn(
                "Follow target has reached the end of the saved route "
                f"(waypoint #{self.ld.nearest_waypoint_index}); it is now only "
                f"{self.ld.distance_m:.2f} m ahead, not {self._lookahead_distance:.2f} m."
            )
        self._lookahead_at_end = self.ld.reached_route_end

    def _publish_lookahead_marker(self, local: PointStamped) -> None:
        sphere = Marker()
        sphere.header = local.header
        sphere.ns = "lookahead"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.frame_locked = True
        sphere.pose.position = local.point
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.9
        sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        sphere.lifetime = Duration(seconds=1.0).to_msg()
        self._lookahead_marker_pub.publish(sphere)

        label = Marker()
        label.header = local.header
        label.ns = "lookahead"
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.frame_locked = True
        label.pose.position.x = local.point.x
        label.pose.position.y = local.point.y
        label.pose.position.z = local.point.z + 1.2
        label.pose.orientation.w = 1.0
        label.scale.z = 0.7
        label.color = ColorRGBA(r=1.0, g=0.6, b=0.6, a=1.0)
        # Fractional index, not just the nearest waypoint: it shows the target
        # sliding continuously rather than snapping from one waypoint to the next.
        label.text = f"ld #{self.ld.route_position:.1f}  {self.ld.distance_m:.1f} m"
        label.lifetime = Duration(seconds=1.0).to_msg()
        self._lookahead_marker_pub.publish(label)

    def _delete_lookahead_marker(self, stamp) -> None:
        for marker_id in (0, 1):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self._visualization_frame
            marker.ns = "lookahead"
            marker.id = marker_id
            marker.action = Marker.DELETE
            self._lookahead_marker_pub.publish(marker)

    def _publish_marker(self, pose: PoseStamped, on_track: bool) -> None:
        m = Marker()
        # The pose's own stamp, on the same timeline as the TF broadcast from it
        # - see _publish_lookahead() on why a wall-clock stamp hides a marker in
        # any view whose fixed frame is not the marker's own.
        m.header = pose.header
        m.ns = "localization_monitor"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.frame_locked = True
        m.pose = pose.pose
        m.scale.x = m.scale.y = m.scale.z = 0.6
        m.color = (
            ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            if on_track
            else ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        )
        m.lifetime = Duration(seconds=1.0).to_msg()
        self._marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationMonitorNode()
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
