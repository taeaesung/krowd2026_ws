"""Publish a position/yaw prior from RTK position and the saved route.

At a standstill a single GNSS antenna has no heading.  The selected route does:
its nearest waypoint stores the intended driving direction.  This node uses only
that direction as a *prior* on /initialpose; scan-to-map registration must still
pass its fitness, GNSS residual, uniqueness, and stable-scan gates before the
LiDAR heading is marked verified.
"""

import csv
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus

from mola_slam_mapping.gnss_imu_pose_node import geodetic_to_utm


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class RouteSeed:
    x: float
    y: float
    z: float
    yaw: float
    distance: float
    waypoint_index: int


def load_route(path: str) -> list[tuple[int, float, float, float, float]]:
    points = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            points.append(
                (
                    int(row["idx"]),
                    float(row["utm_x"]),
                    float(row["utm_y"]),
                    float(row["utm_z"]),
                    float(row["yaw_rad"]),
                )
            )
    if not points:
        raise ValueError(f"route is empty: {path}")
    return points


def nearest_route_seed(
    latitude: float,
    longitude: float,
    route: list[tuple[int, float, float, float, float]],
    origin_latitude: float,
    origin_longitude: float,
    origin_altitude: float,
    enu_from_map_x: float,
    enu_from_map_y: float,
    enu_from_map_z: float,
    enu_from_map_yaw: float,
    gps_lever_arm_x: float = 0.0,
    gps_lever_arm_y: float = 0.0,
) -> RouteSeed:
    gps_easting, gps_northing, _, zone, _ = geodetic_to_utm(
        latitude, longitude, 0.0
    )
    # First choose a nearby route direction using the antenna position, then
    # move that measurement back to base_link and choose the final waypoint.
    # This avoids publishing an /initialpose about 2.5 m in front of the car.
    nearest = min(
        route,
        key=lambda point: (point[1] - gps_easting) ** 2 + (point[2] - gps_northing) ** 2,
    )
    preliminary_yaw = nearest[4]
    base_easting = gps_easting - (
        math.cos(preliminary_yaw) * gps_lever_arm_x
        - math.sin(preliminary_yaw) * gps_lever_arm_y
    )
    base_northing = gps_northing - (
        math.sin(preliminary_yaw) * gps_lever_arm_x
        + math.cos(preliminary_yaw) * gps_lever_arm_y
    )
    nearest = min(
        route,
        key=lambda point: (point[1] - base_easting) ** 2 + (point[2] - base_northing) ** 2,
    )
    index, utm_x, utm_y, utm_z, utm_yaw = nearest
    base_easting = gps_easting - (
        math.cos(utm_yaw) * gps_lever_arm_x
        - math.sin(utm_yaw) * gps_lever_arm_y
    )
    base_northing = gps_northing - (
        math.sin(utm_yaw) * gps_lever_arm_x
        + math.cos(utm_yaw) * gps_lever_arm_y
    )
    distance = math.hypot(utm_x - base_easting, utm_y - base_northing)
    origin_easting, origin_northing, _, origin_zone, _ = geodetic_to_utm(
        origin_latitude, origin_longitude, origin_altitude
    )
    if zone != origin_zone:
        raise ValueError(f"GNSS UTM zone {zone} differs from map zone {origin_zone}")
    dx = (base_easting - origin_easting) - enu_from_map_x
    dy = (base_northing - origin_northing) - enu_from_map_y
    cosine = math.cos(enu_from_map_yaw)
    sine = math.sin(enu_from_map_yaw)
    map_x = cosine * dx + sine * dy
    map_y = -sine * dx + cosine * dy
    map_z = (utm_z - origin_altitude) - enu_from_map_z
    return RouteSeed(
        x=map_x,
        y=map_y,
        z=map_z,
        yaw=wrap_angle(utm_yaw - enu_from_map_yaw),
        distance=distance,
        waypoint_index=index,
    )


class RouteInitialPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("route_initial_pose_node")
        self.declare_parameter("gps_topic", "/ublox_gps_node/fix")
        self.declare_parameter("initial_pose_topic", "/initialpose")
        self.declare_parameter("waypoints_csv", "")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("origin_latitude", 0.0)
        self.declare_parameter("origin_longitude", 0.0)
        self.declare_parameter("origin_altitude", 0.0)
        self.declare_parameter("enu_from_map_x", 0.0)
        self.declare_parameter("enu_from_map_y", 0.0)
        self.declare_parameter("enu_from_map_z", 0.0)
        self.declare_parameter("enu_from_map_yaw_deg", 0.0)
        self.declare_parameter("maximum_gps_horizontal_sigma_m", 0.50)
        self.declare_parameter("maximum_route_distance_m", 6.0)
        self.declare_parameter("position_sigma_m", 1.0)
        self.declare_parameter("z_sigma_m", 1.0)
        self.declare_parameter("yaw_sigma_deg", 5.0)
        self.declare_parameter("roll_pitch_sigma_deg", 5.0)
        self.declare_parameter("gps_lever_arm_x_m", 2.5038)
        self.declare_parameter("gps_lever_arm_y_m", 0.0)

        route_path = str(self.get_parameter("waypoints_csv").value)
        self._route = load_route(route_path)
        self._origin = (
            float(self.get_parameter("origin_latitude").value),
            float(self.get_parameter("origin_longitude").value),
            float(self.get_parameter("origin_altitude").value),
        )
        self._transform = (
            float(self.get_parameter("enu_from_map_x").value),
            float(self.get_parameter("enu_from_map_y").value),
            float(self.get_parameter("enu_from_map_z").value),
            math.radians(float(self.get_parameter("enu_from_map_yaw_deg").value)),
        )
        self._maximum_gps_sigma = float(
            self.get_parameter("maximum_gps_horizontal_sigma_m").value
        )
        self._maximum_route_distance = float(
            self.get_parameter("maximum_route_distance_m").value
        )
        self._position_sigma = float(self.get_parameter("position_sigma_m").value)
        self._z_sigma = float(self.get_parameter("z_sigma_m").value)
        self._yaw_sigma = math.radians(float(self.get_parameter("yaw_sigma_deg").value))
        self._roll_pitch_sigma = math.radians(
            float(self.get_parameter("roll_pitch_sigma_deg").value)
        )
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._published = False
        self._publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            str(self.get_parameter("initial_pose_topic").value),
            10,
        )
        self.create_subscription(
            NavSatFix,
            str(self.get_parameter("gps_topic").value),
            self._on_fix,
            10,
        )
        self.get_logger().info(
            f"route_initial_pose_node: {len(self._route)} route points loaded; "
            "waiting for an accurate GNSS fix to publish a stationary yaw prior"
        )

    def _on_fix(self, message: NavSatFix) -> None:
        if self._published or message.status.status < NavSatStatus.STATUS_FIX:
            return
        covariance = message.position_covariance
        sigma = math.sqrt(max(0.0, covariance[0] + covariance[4]))
        if not math.isfinite(sigma) or sigma > self._maximum_gps_sigma:
            return
        try:
            seed = nearest_route_seed(
                message.latitude,
                message.longitude,
                self._route,
                *self._origin,
                *self._transform,
                float(self.get_parameter("gps_lever_arm_x_m").value),
                float(self.get_parameter("gps_lever_arm_y_m").value),
            )
        except ValueError as error:
            self.get_logger().error(f"Cannot make route heading prior: {error}")
            return
        if seed.distance > self._maximum_route_distance:
            self.get_logger().warn(
                f"GNSS fix is {seed.distance:.2f} m from the saved route; "
                f"route yaw prior withheld (limit {self._maximum_route_distance:.2f} m)",
                throttle_duration_sec=5.0,
            )
            return
        if self._publisher.get_subscription_count() == 0:
            return

        prior = PoseWithCovarianceStamped()
        prior.header = message.header
        prior.header.frame_id = self._map_frame
        prior.pose.pose.position.x = seed.x
        prior.pose.pose.position.y = seed.y
        prior.pose.pose.position.z = seed.z
        prior.pose.pose.orientation.z = math.sin(seed.yaw * 0.5)
        prior.pose.pose.orientation.w = math.cos(seed.yaw * 0.5)
        diagonal = (
            self._position_sigma**2,
            self._position_sigma**2,
            self._z_sigma**2,
            self._roll_pitch_sigma**2,
            self._roll_pitch_sigma**2,
            self._yaw_sigma**2,
        )
        for axis, variance in enumerate(diagonal):
            prior.pose.covariance[axis * 7] = variance
        self._publisher.publish(prior)
        self._published = True
        self.get_logger().info(
            f"Published stationary route prior from waypoint {seed.waypoint_index}: "
            f"distance={seed.distance:.2f} m, map pose=({seed.x:.2f}, {seed.y:.2f}), "
            f"yaw={math.degrees(seed.yaw):.1f} deg +/- "
            f"{math.degrees(self._yaw_sigma):.1f} deg; LiDAR must still verify it"
        )


def main(args=None):
    rclpy.init(args=args)
    node = RouteInitialPoseNode()
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
