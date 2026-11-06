"""GNSS/IMU vehicle pose independent of LiDAR odometry and MOLA smoothing."""

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped, TwistWithCovarianceStamped
from mrpt_nav_interfaces.msg import GeoreferencingMetadata
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus
from std_msgs.msg import Bool, String
import tf2_ros

from mola_slam_mapping.georeference_io import load_fixed_georeference


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _stamp_ns(message) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def normalize_quaternion(quaternion):
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1.0e-12:
        raise ValueError("zero-norm quaternion")
    return tuple(value / norm for value in quaternion)


def multiply_quaternions(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def inverse_quaternion(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return (-x, -y, -z, w)


def rotate_vector(quaternion, vector):
    x, y, z, w = normalize_quaternion(quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def quaternion_from_euler(yaw: float, pitch: float, roll: float):
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    cp = math.cos(0.5 * pitch)
    sp = math.sin(0.5 * pitch)
    cr = math.cos(0.5 * roll)
    sr = math.sin(0.5 * roll)
    return normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def euler_from_quaternion(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return yaw, pitch, roll


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def enu_heading_from_map_yaw(yaw_map: float, q_enu_map) -> float:
    """The ENU heading of a body whose yaw in the map frame is `yaw_map`.

    _publish_sample composes q_map_base = inverse(q_enu_map) * q_enu_base, so
    in the plane yaw_map = heading - yaw(q_enu_map). The georeference's pitch
    and roll are ignored here; they are fractions of a degree.
    """

    difference = yaw_map + yaw_from_quaternion(*q_enu_map)
    return math.atan2(math.sin(difference), math.cos(difference))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def course_spread(courses: list[float]) -> tuple[float, float]:
    """Circular mean of the courses and the largest deviation from it."""

    if not courses:
        return 0.0, math.inf
    mean = math.atan2(
        sum(math.sin(course) for course in courses),
        sum(math.cos(course) for course in courses),
    )
    return mean, max(abs(wrap_angle(course - mean)) for course in courses)


def window_start_index(cumulative_travel: list[float], max_travel_m: float) -> int:
    """First index to keep so the window spans at most `max_travel_m` of travel.

    The straight-segment test compares the chord between the first and last
    sample with the distance travelled between them, so a window that keeps
    growing through a curve can never pass it again. Dropping the oldest
    samples lets a straight driven later still qualify.
    """

    start = 0
    while (
        start < len(cumulative_travel) - 1
        and cumulative_travel[-1] - cumulative_travel[start] > max_travel_m
    ):
        start += 1
    return start


def circular_lerp(previous: float, measured: float, alpha: float) -> float:
    difference = (measured - previous + math.pi) % (2.0 * math.pi) - math.pi
    return previous + alpha * difference


def geodetic_to_ecef(latitude_deg: float, longitude_deg: float, altitude: float):
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_latitude**2)
    return (
        (radius + altitude) * cos_latitude * math.cos(longitude),
        (radius + altitude) * cos_latitude * math.sin(longitude),
        (radius * (1.0 - WGS84_E2) + altitude) * sin_latitude,
    )


def geodetic_to_enu(
    latitude_deg: float,
    longitude_deg: float,
    altitude: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude: float,
):
    current = geodetic_to_ecef(latitude_deg, longitude_deg, altitude)
    origin = geodetic_to_ecef(
        origin_latitude_deg,
        origin_longitude_deg,
        origin_altitude,
    )
    delta = tuple(value - reference for value, reference in zip(current, origin))
    latitude = math.radians(origin_latitude_deg)
    longitude = math.radians(origin_longitude_deg)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    sin_longitude = math.sin(longitude)
    cos_longitude = math.cos(longitude)
    return (
        -sin_longitude * delta[0] + cos_longitude * delta[1],
        (
            -sin_latitude * cos_longitude * delta[0]
            - sin_latitude * sin_longitude * delta[1]
            + cos_latitude * delta[2]
        ),
        (
            cos_latitude * cos_longitude * delta[0]
            + cos_latitude * sin_longitude * delta[1]
            + sin_latitude * delta[2]
        ),
    )


@dataclass
class PoseSample:
    stamp_ns: int
    fix: NavSatFix
    velocity: TwistWithCovarianceStamped
    antenna_enu: tuple[float, float, float]
    course: float | None
    speed: float
    pitch: float
    roll: float
    angular_z: float


def geodetic_to_utm(latitude: float, longitude: float, altitude: float):
    """Convert WGS84 geodetic coordinates to UTM easting/northing."""

    if not -80.0 <= latitude <= 84.0:
        raise ValueError("UTM latitude must be in [-80, 84] degrees")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError("longitude must be in [-180, 180] degrees")

    zone = max(1, min(60, int((longitude + 180.0) / 6.0) + 1))
    longitude_origin = math.radians((zone - 1) * 6 - 180 + 3)
    latitude_rad = math.radians(latitude)
    longitude_rad = math.radians(longitude)

    k0 = 0.9996
    eccentric_prime_squared = WGS84_E2 / (1.0 - WGS84_E2)
    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    tan_latitude = math.tan(latitude_rad)
    radius = WGS84_A / math.sqrt(
        1.0 - WGS84_E2 * sin_latitude * sin_latitude
    )
    tangent_squared = tan_latitude * tan_latitude
    eccentricity_term = eccentric_prime_squared * cos_latitude * cos_latitude
    longitude_term = cos_latitude * (longitude_rad - longitude_origin)

    meridian = WGS84_A * (
        (1.0 - WGS84_E2 / 4.0 - 3.0 * WGS84_E2**2 / 64.0
         - 5.0 * WGS84_E2**3 / 256.0) * latitude_rad
        - (3.0 * WGS84_E2 / 8.0 + 3.0 * WGS84_E2**2 / 32.0
           + 45.0 * WGS84_E2**3 / 1024.0) * math.sin(2.0 * latitude_rad)
        + (15.0 * WGS84_E2**2 / 256.0
           + 45.0 * WGS84_E2**3 / 1024.0) * math.sin(4.0 * latitude_rad)
        - (35.0 * WGS84_E2**3 / 3072.0) * math.sin(6.0 * latitude_rad)
    )

    easting = 500000.0 + k0 * radius * (
        longitude_term
        + (1.0 - tangent_squared + eccentricity_term) * longitude_term**3 / 6.0
        + (
            5.0 - 18.0 * tangent_squared + tangent_squared**2
            + 72.0 * eccentricity_term - 58.0 * eccentric_prime_squared
        ) * longitude_term**5 / 120.0
    )
    northing = k0 * (
        meridian
        + radius * tan_latitude * (
            longitude_term**2 / 2.0
            + (
                5.0 - tangent_squared + 9.0 * eccentricity_term
                + 4.0 * eccentricity_term**2
            ) * longitude_term**4 / 24.0
            + (
                61.0 - 58.0 * tangent_squared + tangent_squared**2
                + 600.0 * eccentricity_term
                - 330.0 * eccentric_prime_squared
            ) * longitude_term**6 / 720.0
        )
    )
    if latitude < 0.0:
        northing += 10000000.0
    band = "CDEFGHJKLMNPQRSTUVWXX"[int((latitude + 80.0) // 8.0)]
    return easting, northing, altitude, zone, band


# /gnss_imu/heading_acquisition values. Path tracking creeps straight (to earn
# an RTK heading) only on NEEDS_MOTION; everything else means "wait".
HEADING_WAITING_GNSS = "WAITING_GNSS"
HEADING_WAITING_ESKF = "WAITING_ESKF"
HEADING_WAITING_LIDAR = "WAITING_LIDAR"
HEADING_NEEDS_MOTION = "NEEDS_MOTION"
HEADING_ACQUIRED = "ACQUIRED"


def eskf_heading_status(message, status_name: str) -> tuple[bool, str] | None:
    """(heading_verified, heading_source) of gps_imu_eskf in a DiagnosticArray.

    None when the array carries no status of that name - /diagnostics is shared
    by every node that reports health.
    """

    for status in message.status:
        if status.name != status_name:
            continue
        values = {entry.key: entry.value for entry in status.values}
        return (
            values.get("heading_verified", "False") == "True",
            values.get("heading_source", ""),
        )
    return None


def heading_acquisition_state(
    aligned: bool,
    seed_source: str | None,
    seeding_enabled: bool,
    eskf_pending: bool,
    lidar_pending: bool,
    gnss_ok: bool = True,
) -> str:
    """Where the initial heading stands, in the order it is looked for.

    ESKF (a heading the sensor stack has already verified) first, then the
    verified LiDAR map alignment, and only when neither is coming does the
    vehicle have to move for the RTK straight segment. Without a GNSS fix
    nothing can seed and no straight segment can be measured, so NEEDS_MOTION
    is never reported then: a creep would be blind for nothing.
    """

    if aligned:
        return f"{HEADING_ACQUIRED} {seed_source or 'rtk_straight'}"
    if not gnss_ok:
        return HEADING_WAITING_GNSS
    if not seeding_enabled:
        return HEADING_NEEDS_MOTION
    if eskf_pending:
        return HEADING_WAITING_ESKF
    if lidar_pending:
        return HEADING_WAITING_LIDAR
    return HEADING_NEEDS_MOTION


class GnssImuPoseNode(Node):
    """Publish map->base_link from GNSS position/course and IMU roll/pitch.

    The u-blox fix_velocity vector in the vehicle bag is ENU even though its
    header says gps_link. Its direction agrees with the differentiated GNSS
    track; it must therefore be used as a global course, not as body vx/vy.

    An initial five-metre straight segment is buffered before heading is
    committed. Once ready, the buffered beginning is published too. During
    mapping this first globally oriented base pose defines map origin. During
    localization a saved ENU<-map georeference places the pose in the exact
    frame of the loaded map.
    """

    def __init__(self):
        super().__init__("gnss_imu_pose_node")

        self.declare_parameter("fix_topic", "/ublox_gps_node/fix")
        self.declare_parameter(
            "velocity_topic", "/ublox_gps_node/fix_velocity"
        )
        self.declare_parameter("imu_topic", "/imu/data_ekf")
        self.declare_parameter("output_topic", "/gnss_imu/pose")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("enu_frame", "enu")
        self.declare_parameter("utm_frame", "utm")
        self.declare_parameter("georeference_topic", "/state_estimator/geo_ref_metadata")
        self.declare_parameter("map_dir", "")
        self.declare_parameter("georeference_map_filename", "global_map.mm")
        self.declare_parameter("initial_alignment_min_distance_m", 5.0)
        self.declare_parameter("initial_alignment_min_straightness_ratio", 0.95)
        self.declare_parameter("heading_min_speed_mps", 0.5)
        self.declare_parameter("heading_filter_alpha", 0.65)
        self.declare_parameter("sync_tolerance_sec", 0.06)
        # Localization only (a saved georeference): commit the heading from the
        # verified LiDAR map pose as soon as it and a GNSS fix exist, so the
        # pose is published at a standstill instead of after ~5 m of straight
        # driving. The RTK course filters the heading as usual once the vehicle
        # exceeds heading_min_speed_mps, and the ordinary straight-segment
        # alignment (initial_alignment_min_distance_m of straight driving)
        # keeps running underneath: when it passes, its chord heading REPLACES
        # the seeded one and the two are compared - a disagreement over
        # lidar_heading_warn_deg is what a wrong LiDAR alignment looks like.
        self.declare_parameter("seed_heading_from_lidar", True)
        self.declare_parameter("lidar_pose_topic", "/mola/map_localization/odometry")
        self.declare_parameter(
            "lidar_heading_verified_topic", "/mola/map_localization/heading_verified"
        )
        self.declare_parameter("lidar_heading_warn_deg", 20.0)
        # The straight segment that replaces a seeded heading must also be
        # straight by its course samples: all within this of their mean, and
        # the chord within this of it. The chord/travel ratio alone passes a
        # gentle 60 deg sweep, whose chord heading is 20-30 deg off.
        self.declare_parameter("seed_confirm_max_course_spread_deg", 10.0)
        # Before LiDAR: a heading the sensor stack's ESKF (gps_imu_eskf) has
        # already verified - it verifies from the RTK course while the vehicle
        # is driven to the start line, then holds it at a standstill. Accepted
        # only from these heading_source values; "saved_prior" (last run's
        # heading) is unverified and never used. The ESKF yaw is ENU, the same
        # convention as the RTK course (0913_2 bag: median 0.0 deg apart).
        self.declare_parameter("seed_heading_from_eskf", True)
        self.declare_parameter("eskf_odometry_topic", "/ekf/global_odometry")
        self.declare_parameter("eskf_diagnostics_topic", "/diagnostics")
        self.declare_parameter("eskf_diagnostic_name", "gps_imu_eskf")
        self.declare_parameter("eskf_heading_sources", ["rtk_course", "lidar_map"])
        self.declare_parameter("eskf_max_odometry_age_sec", 0.5)
        # How long after the first GNSS fix to wait for the ESKF's status before
        # moving on to LiDAR, and then for a verified LiDAR alignment before
        # reporting NEEDS_MOTION (path tracking then creeps straight).
        self.declare_parameter("eskf_wait_sec", 3.0)
        self.declare_parameter("lidar_seed_wait_sec", 10.0)
        self.declare_parameter(
            "heading_acquisition_topic", "/gnss_imu/heading_acquisition"
        )
        self.declare_parameter("gps_lever_x_m", 2.5038)
        self.declare_parameter("gps_lever_y_m", 0.0)
        self.declare_parameter("gps_lever_z_m", 1.021)
        self.declare_parameter(
            "base_to_imu_qx", -0.00927259539161
        )
        self.declare_parameter(
            "base_to_imu_qy", 0.030217093795
        )
        self.declare_parameter(
            "base_to_imu_qz", 0.00028033096354
        )
        self.declare_parameter(
            "base_to_imu_qw", 0.99950030897
        )

        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._enu_frame = str(self.get_parameter("enu_frame").value)
        self._utm_frame = str(self.get_parameter("utm_frame").value)
        self._min_distance = float(
            self.get_parameter("initial_alignment_min_distance_m").value
        )
        self._min_straightness = float(
            self.get_parameter(
                "initial_alignment_min_straightness_ratio"
            ).value
        )
        self._min_heading_speed = float(
            self.get_parameter("heading_min_speed_mps").value
        )
        self._heading_alpha = float(
            self.get_parameter("heading_filter_alpha").value
        )
        self._sync_tolerance_ns = int(
            float(self.get_parameter("sync_tolerance_sec").value) * 1.0e9
        )
        self._gps_lever = (
            float(self.get_parameter("gps_lever_x_m").value),
            float(self.get_parameter("gps_lever_y_m").value),
            float(self.get_parameter("gps_lever_z_m").value),
        )
        self._q_base_imu = normalize_quaternion(
            tuple(
                float(self.get_parameter(name).value)
                for name in (
                    "base_to_imu_qx",
                    "base_to_imu_qy",
                    "base_to_imu_qz",
                    "base_to_imu_qw",
                )
            )
        )

        if self._min_distance <= 0.0:
            raise ValueError("initial_alignment_min_distance_m must be positive")
        if not 0.0 < self._min_straightness <= 1.0:
            raise ValueError(
                "initial_alignment_min_straightness_ratio must be in (0, 1]"
            )
        if not 0.0 < self._heading_alpha <= 1.0:
            raise ValueError("heading_filter_alpha must be in (0, 1]")

        self._origin = None
        self._t_enu_map = None
        self._georeference_published = False
        self._q_enu_map = None
        self._using_saved_georeference = False
        self._load_saved_georeference()

        self._fixes = deque(maxlen=200)
        self._velocities = deque(maxlen=200)
        self._processed_fix_stamps = set()
        self._alignment_samples: list[PoseSample] = []
        self._alignment_cumulative: list[float] = []
        self._alignment_travel = 0.0
        self._aligned = False
        self._heading = None
        self._latest_pitch = 0.0
        self._latest_roll = 0.0
        self._latest_angular_z = 0.0
        self._last_fix_stamp = None
        self._seed_from_lidar = (
            bool(self.get_parameter("seed_heading_from_lidar").value)
            and self._using_saved_georeference
        )
        self._lidar_warn_rad = math.radians(
            float(self.get_parameter("lidar_heading_warn_deg").value)
        )
        self._seed_confirm_spread_rad = math.radians(
            float(self.get_parameter("seed_confirm_max_course_spread_deg").value)
        )
        self._lidar_yaw_map: float | None = None
        self._lidar_verified = False
        # The seeded heading the pose started on (ESKF or LiDAR), kept until
        # the first RTK straight segment has been compared against it.
        self._seed_heading: float | None = None
        self._seed_source: str | None = None
        self._seed_from_eskf = (
            bool(self.get_parameter("seed_heading_from_eskf").value)
            and self._using_saved_georeference
        )
        self._eskf_sources = {
            str(source) for source in self.get_parameter("eskf_heading_sources").value
        }
        self._eskf_diagnostic_name = str(
            self.get_parameter("eskf_diagnostic_name").value
        )
        self._eskf_max_age_ns = int(
            float(self.get_parameter("eskf_max_odometry_age_sec").value) * 1.0e9
        )
        self._eskf_wait_ns = int(float(self.get_parameter("eskf_wait_sec").value) * 1.0e9)
        self._lidar_wait_ns = int(
            float(self.get_parameter("lidar_seed_wait_sec").value) * 1.0e9
        )
        self._eskf_status: tuple[bool, str] | None = None
        self._eskf_yaw: float | None = None
        self._eskf_yaw_received_ns: int | None = None
        # Receive time of the first GNSS fix: the ESKF/LiDAR waits count from
        # there, not from launch, so a bag that starts late or a slow receiver
        # does not use them up before any data exists.
        self._first_fix_received_ns: int | None = None
        # Receive time of the last GNSS fix/velocity pair with a valid fix.
        self._last_good_pair_ns: int | None = None
        self._gnss_ok_max_age_ns = 2_000_000_000
        self._eskf_decided_ns: int | None = None
        self._heading_state: str | None = None
        self._last_velocity_stamp = None
        self._last_imu_stamp = None
        # Input accounting for _report_input_health(). Both GNSS streams are
        # required: _match_pending() returns immediately unless BOTH deques are
        # non-empty, so one missing stream silently produces no pose at all, no
        # map/utm frames, and therefore no route.
        self._fix_count = 0
        self._velocity_count = 0
        self._imu_count = 0
        self._published_count = 0

        self._publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("output_topic").value),
            20,
        )
        # There is deliberately no second "sensor time" publisher: /gnss_imu/pose
        # already carries the GNSS measurement stamp (see _publish_sample), which is
        # the same clock domain BridgeROS2 uses for /lidar_odometry/pose, so
        # map_odom_tf_node can interpolate the two directly.
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._georeference_publisher = self.create_publisher(
            GeoreferencingMetadata,
            str(self.get_parameter("georeference_topic").value),
            latched_qos,
        )
        self._static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        if self._using_saved_georeference:
            self._publish_georeference()

        self.create_subscription(
            NavSatFix,
            str(self.get_parameter("fix_topic").value),
            self._on_fix,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TwistWithCovarianceStamped,
            str(self.get_parameter("velocity_topic").value),
            self._on_velocity,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("imu_topic").value),
            self._on_imu,
            qos_profile_sensor_data,
        )
        if self._seed_from_lidar:
            lidar_pose_topic = str(self.get_parameter("lidar_pose_topic").value)
            verified_topic = str(
                self.get_parameter("lidar_heading_verified_topic").value
            )
            self.create_subscription(
                Odometry, lidar_pose_topic, self._on_lidar_pose, 10
            )
            self.create_subscription(
                Bool, verified_topic, self._on_lidar_verified, 10
            )
            self.get_logger().info(
                "heading is also seeded from the LiDAR map pose on "
                f"'{lidar_pose_topic}' once '{verified_topic}' is true, so the "
                "pose can start at a standstill"
            )
        elif bool(self.get_parameter("seed_heading_from_lidar").value):
            self.get_logger().info(
                "seed_heading_from_lidar ignored: no saved georeference (mapping), "
                "the heading comes from the first straight segment"
            )
        if self._seed_from_eskf:
            self.create_subscription(
                Odometry,
                str(self.get_parameter("eskf_odometry_topic").value),
                self._on_eskf_odometry,
                qos_profile_sensor_data,
            )
            self.create_subscription(
                DiagnosticArray,
                str(self.get_parameter("eskf_diagnostics_topic").value),
                self._on_eskf_diagnostics,
                10,
            )
            self.get_logger().info(
                "heading is taken from the sensor-stack ESKF first when it reports "
                f"heading_verified from {sorted(self._eskf_sources)}; LiDAR next, "
                "then the RTK straight segment (NEEDS_MOTION)"
            )
        self._heading_state_publisher = self.create_publisher(
            String,
            str(self.get_parameter("heading_acquisition_topic").value),
            latched_qos,
        )
        self.create_timer(0.2, self._publish_heading_state)
        self.create_timer(5.0, self._report_input_health)

        mode = (
            "saved map georeference"
            if self._using_saved_georeference
            else "first straight segment"
        )
        self.get_logger().info(
            "gnss_imu_pose_node: GNSS position + ENU velocity course + IMU "
            "roll/pitch -> '%s' in '%s'; alignment source=%s"
            % (
                self.get_parameter("output_topic").value,
                self._map_frame,
                mode,
            )
        )

    def _load_saved_georeference(self) -> None:
        map_dir = str(self.get_parameter("map_dir").value).strip()
        if not map_dir:
            return
        map_filename = str(
            self.get_parameter("georeference_map_filename").value
        )
        fixed = load_fixed_georeference(Path(map_dir).expanduser(), map_filename)
        self._origin = (
            fixed.latitude,
            fixed.longitude,
            fixed.altitude,
        )
        self._t_enu_map = (fixed.x, fixed.y, fixed.z)
        self._q_enu_map = quaternion_from_euler(
            math.radians(fixed.yaw_deg),
            math.radians(fixed.pitch_deg),
            math.radians(fixed.roll_deg),
        )
        self._using_saved_georeference = True

    def _reset_replay_state(self) -> None:
        self._fixes.clear()
        self._velocities.clear()
        self._processed_fix_stamps.clear()
        self._alignment_samples.clear()
        self._alignment_cumulative.clear()
        self._alignment_travel = 0.0
        self._aligned = False
        self._heading = None
        self._latest_pitch = 0.0
        self._latest_roll = 0.0
        self._latest_angular_z = 0.0
        self._lidar_verified = False
        self._seed_heading = None
        self._seed_source = None
        self._first_fix_received_ns = None
        self._last_good_pair_ns = None
        self._eskf_decided_ns = None
        if not self._using_saved_georeference:
            self._origin = None
            self._t_enu_map = None
            self._q_enu_map = None
            self._georeference_published = False
        self.get_logger().info("Detected backward bag time; reset GNSS/IMU pose state")

    def _check_backward(self, stamp_ns: int, attribute: str) -> None:
        previous = getattr(self, attribute)
        if previous is not None and stamp_ns + 500_000_000 < previous:
            self._reset_replay_state()
            self._last_fix_stamp = None
            self._last_velocity_stamp = None
            self._last_imu_stamp = None
        setattr(self, attribute, stamp_ns)

    def _report_input_health(self) -> None:
        """Name the missing input instead of silently producing no pose.

        Every downstream artifact depends on this node: no pose means no map->odom
        TF, no "map"/"utm" frames, and an empty waypoints.csv. The most common cause
        is a bag replayed with an incomplete --topics list, which is otherwise
        indistinguishable from a hardware fault.
        """

        if self._published_count:
            return

        missing = []
        if not self._fix_count:
            missing.append(f"'{self.get_parameter('fix_topic').value}' (GNSS position)")
        if not self._velocity_count:
            missing.append(
                f"'{self.get_parameter('velocity_topic').value}' (GNSS ENU course)"
            )
        if not self._imu_count:
            missing.append(f"'{self.get_parameter('imu_topic').value}' (IMU tilt)")

        if missing:
            self.get_logger().warning(
                "No pose published: no message yet on " + ", ".join(missing)
                + ". BOTH GNSS topics are required. When replaying a bag, include "
                "every one of them in --topics.",
                throttle_duration_sec=10.0,
            )
            return

        if not self._aligned:
            self.get_logger().warning(
                "All inputs arriving (%d fix / %d velocity / %d imu) but heading is "
                "not yet committed: needs a >=%.1f m segment with straightness >=%.2f "
                "and >=5 samples above %.2f m/s. Travelled %.1f m so far."
                % (
                    self._fix_count,
                    self._velocity_count,
                    self._imu_count,
                    self._min_distance,
                    self._min_straightness,
                    self._min_heading_speed,
                    self._alignment_travel,
                ),
                throttle_duration_sec=10.0,
            )
            return

        self.get_logger().warning(
            "Inputs arriving and heading committed, but no pose published - GNSS fix "
            "and velocity stamps may not pair within %.3f s of each other."
            % (self._sync_tolerance_ns * 1.0e-9),
            throttle_duration_sec=10.0,
        )

    def _on_fix(self, message: NavSatFix) -> None:
        stamp_ns = _stamp_ns(message)
        self._check_backward(stamp_ns, "_last_fix_stamp")
        self._fix_count += 1
        if self._first_fix_received_ns is None:
            self._first_fix_received_ns = self.get_clock().now().nanoseconds
        self._fixes.append(message)
        self._match_pending()

    def _on_velocity(self, message: TwistWithCovarianceStamped) -> None:
        stamp_ns = _stamp_ns(message)
        self._check_backward(stamp_ns, "_last_velocity_stamp")
        self._velocity_count += 1
        self._velocities.append(message)
        self._match_pending()

    def _on_imu(self, message: Imu) -> None:
        stamp_ns = _stamp_ns(message)
        self._check_backward(stamp_ns, "_last_imu_stamp")
        self._imu_count += 1
        orientation = message.orientation
        try:
            q_enu_imu = normalize_quaternion(
                (
                    orientation.x,
                    orientation.y,
                    orientation.z,
                    orientation.w,
                )
            )
        except ValueError:
            return
        q_enu_base_measured = multiply_quaternions(
            q_enu_imu,
            inverse_quaternion(self._q_base_imu),
        )
        _, pitch, roll = euler_from_quaternion(q_enu_base_measured)
        self._latest_pitch = pitch
        self._latest_roll = roll
        self._latest_angular_z = float(message.angular_velocity.z)

    def _match_pending(self) -> None:
        if not self._fixes or not self._velocities:
            return
        for fix in tuple(self._fixes):
            fix_stamp = _stamp_ns(fix)
            if fix_stamp in self._processed_fix_stamps:
                continue
            velocity = min(
                self._velocities,
                key=lambda candidate: abs(_stamp_ns(candidate) - fix_stamp),
            )
            if abs(_stamp_ns(velocity) - fix_stamp) > self._sync_tolerance_ns:
                continue
            self._processed_fix_stamps.add(fix_stamp)
            self._process_pair(fix, velocity)

    def _process_pair(
        self,
        fix: NavSatFix,
        velocity: TwistWithCovarianceStamped,
    ) -> None:
        values = (fix.latitude, fix.longitude, fix.altitude)
        if (
            fix.status.status < NavSatStatus.STATUS_FIX
            or not all(math.isfinite(value) for value in values)
        ):
            return

        self._last_good_pair_ns = self.get_clock().now().nanoseconds
        if self._origin is None:
            self._origin = values
        antenna_enu = geodetic_to_enu(
            fix.latitude,
            fix.longitude,
            fix.altitude,
            *self._origin,
        )
        vector = velocity.twist.twist.linear
        speed = math.hypot(vector.x, vector.y)
        course = (
            math.atan2(vector.y, vector.x)
            if speed >= self._min_heading_speed
            else None
        )
        sample = PoseSample(
            stamp_ns=_stamp_ns(fix),
            fix=fix,
            velocity=velocity,
            antenna_enu=antenna_enu,
            course=course,
            speed=speed,
            pitch=self._latest_pitch,
            roll=self._latest_roll,
            angular_z=self._latest_angular_z,
        )

        if not self._aligned:
            if self._try_seed_heading():
                self._publish_sample(sample)
                return
            self._collect_alignment_sample(sample)
            self._try_finish_initial_alignment()
            return

        if self._seed_heading is not None:
            # Driving on a seeded heading: the straight-segment alignment keeps
            # running and, when it passes, its heading replaces the seed.
            self._collect_alignment_sample(sample)
            self._try_confirm_seed_with_rtk()
        self._publish_sample(sample)

    def _collect_alignment_sample(self, sample: PoseSample) -> None:
        if self._alignment_samples:
            previous = self._alignment_samples[-1].antenna_enu
            self._alignment_travel += math.hypot(
                sample.antenna_enu[0] - previous[0],
                sample.antenna_enu[1] - previous[1],
            )
        self._alignment_samples.append(sample)
        self._alignment_cumulative.append(self._alignment_travel)

    def _alignment_chord(self, start: int = 0):
        """(chord_m, straightness, heading_count, yaw) of the buffered samples
        from index `start` on, or None while they do not yet make a qualifying
        straight segment."""

        if len(self._alignment_samples) - start < 2:
            return None
        first = self._alignment_samples[start].antenna_enu
        last = self._alignment_samples[-1].antenna_enu
        delta_east = last[0] - first[0]
        delta_north = last[1] - first[1]
        chord = math.hypot(delta_east, delta_north)
        travel = self._alignment_cumulative[-1] - self._alignment_cumulative[start]
        straightness = chord / travel if travel > 1.0e-6 else 0.0
        heading_count = sum(
            sample.course is not None for sample in self._alignment_samples[start:]
        )
        if (
            chord < self._min_distance
            or straightness < self._min_straightness
            or heading_count < 5
        ):
            return None
        return chord, straightness, heading_count, math.atan2(delta_north, delta_east)

    def _try_confirm_seed_with_rtk(self) -> None:
        """Replace a seeded heading with the RTK straight-segment heading."""

        # Unlike the pre-alignment buffer this one is driven through curves
        # too, so keep only the most recent stretch; a straight driven after
        # a corner must still be able to qualify.
        start = window_start_index(
            self._alignment_cumulative, 2.0 * self._min_distance
        )
        if start:
            del self._alignment_samples[:start]
            del self._alignment_cumulative[:start]
        chord = self._alignment_chord()
        if chord is None:
            return
        chord_m, straightness, heading_count, rtk_heading = chord
        courses = [
            sample.course
            for sample in self._alignment_samples
            if sample.course is not None
        ]
        mean_course, spread = course_spread(courses)
        chord_vs_mean = abs(wrap_angle(rtk_heading - mean_course))
        if spread > self._seed_confirm_spread_rad or chord_vs_mean > self._seed_confirm_spread_rad:
            return
        difference = rtk_heading - self._seed_heading
        difference = math.atan2(math.sin(difference), math.cos(difference))
        seed = self._seed_heading
        source = self._seed_source
        self._seed_heading = None
        self._heading = rtk_heading
        self._alignment_samples.clear()
        self._alignment_cumulative.clear()
        message = (
            f"RTK straight-segment heading {math.degrees(rtk_heading):.1f} deg "
            f"(chord {chord_m:.1f} m, straightness {straightness:.3f}, "
            f"{heading_count} course samples) replaces the {source} seed "
            f"{math.degrees(seed):.1f} deg: {math.degrees(difference):+.1f} deg apart"
        )
        if abs(difference) > self._lidar_warn_rad:
            self.get_logger().warning(
                message + f" (over {math.degrees(self._lidar_warn_rad):.0f} deg): "
                f"the {source} heading was probably wrong"
            )
        else:
            self.get_logger().info(message)

    def _on_lidar_pose(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        self._lidar_yaw_map = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    def _on_lidar_verified(self, message: Bool) -> None:
        self._lidar_verified = bool(message.data)

    def _on_eskf_odometry(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        self._eskf_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self._eskf_yaw_received_ns = self.get_clock().now().nanoseconds

    def _on_eskf_diagnostics(self, message: DiagnosticArray) -> None:
        status = eskf_heading_status(message, self._eskf_diagnostic_name)
        if status is None:
            return
        self._eskf_status = status
        if self._eskf_decided_ns is None and not self._eskf_usable_status():
            # The ESKF has answered and its heading is not verified (yet): LiDAR
            # may seed from now on. A later ESKF verification still seeds if
            # nothing has by then - during a creep, say.
            self._eskf_decided_ns = self.get_clock().now().nanoseconds

    def _eskf_usable_status(self) -> bool:
        return (
            self._eskf_status is not None
            and self._eskf_status[0]
            and self._eskf_status[1] in self._eskf_sources
        )

    def _usable_eskf_yaw(self) -> float | None:
        if not self._seed_from_eskf or not self._eskf_usable_status():
            return None
        if self._eskf_yaw is None or self._eskf_yaw_received_ns is None:
            return None
        age = self.get_clock().now().nanoseconds - self._eskf_yaw_received_ns
        return self._eskf_yaw if age <= self._eskf_max_age_ns else None

    def _eskf_pending(self, now_ns: int) -> bool:
        """Still waiting on the ESKF: LiDAR must not seed ahead of it."""

        if not self._seed_from_eskf:
            return False
        if self._usable_eskf_yaw() is not None:
            return True
        if self._eskf_decided_ns is not None:
            return False
        if self._first_fix_received_ns is None:
            return True
        if now_ns - self._first_fix_received_ns < self._eskf_wait_ns:
            return True
        # No ESKF status within the wait (sensor stack not running, or a bag
        # without /diagnostics): decided, move on to LiDAR.
        self._eskf_decided_ns = now_ns
        return False

    def _lidar_pending(self, now_ns: int) -> bool:
        if not self._seed_from_lidar:
            return False
        # The LiDAR window opens when the ESKF has been decided AND GNSS has
        # arrived: the LiDAR localizer needs the RTK position to seed its
        # global search, so a window that ran before any fix would be spent
        # on nothing. An early "not verified" from the ESKF cannot open it.
        if self._first_fix_received_ns is None:
            return True
        decided = self._first_fix_received_ns
        if self._eskf_decided_ns is not None:
            decided = max(decided, self._eskf_decided_ns)
        return now_ns - decided < self._lidar_wait_ns

    def _gnss_ok(self, now_ns: int) -> bool:
        return (
            self._last_good_pair_ns is not None
            and now_ns - self._last_good_pair_ns <= self._gnss_ok_max_age_ns
        )

    def _publish_heading_state(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        state = heading_acquisition_state(
            aligned=self._aligned,
            seed_source=self._seed_source,
            seeding_enabled=self._seed_from_eskf or self._seed_from_lidar,
            eskf_pending=self._eskf_pending(now_ns),
            lidar_pending=self._lidar_pending(now_ns),
            gnss_ok=self._gnss_ok(now_ns),
        )
        if state == self._heading_state:
            return
        self._heading_state = state
        self._heading_state_publisher.publish(String(data=state))
        if state == HEADING_NEEDS_MOTION:
            self.get_logger().warning(
                "heading acquisition: NEEDS_MOTION - no verified ESKF heading and no "
                "verified LiDAR alignment; drive (or let path tracking creep) "
                f"{self._min_distance:.0f} m straight for the RTK heading"
            )
        else:
            self.get_logger().info(f"heading acquisition: {state}")

    def _try_seed_heading(self) -> bool:
        """Commit a seeded heading: the ESKF's first, else the LiDAR map's."""

        if self._aligned or self._q_enu_map is None:
            return False
        now_ns = self.get_clock().now().nanoseconds
        eskf_yaw = self._usable_eskf_yaw()
        if eskf_yaw is not None:
            source = f"ESKF ({self._eskf_status[1]})"
            detail = f"ESKF ENU yaw {math.degrees(eskf_yaw):.2f} deg"
            heading = eskf_yaw
        elif (
            self._seed_from_lidar
            and not self._eskf_pending(now_ns)
            and self._lidar_verified
            and self._lidar_yaw_map is not None
        ):
            heading = enu_heading_from_map_yaw(self._lidar_yaw_map, self._q_enu_map)
            source = "LiDAR"
            detail = (
                f"verified LiDAR map yaw {math.degrees(self._lidar_yaw_map):.2f} deg "
                f"-> ENU heading {math.degrees(heading):.2f} deg"
            )
        else:
            return False
        self._aligned = True
        self._heading = heading
        self._seed_heading = heading
        self._seed_source = source
        self._publish_georeference()
        self.get_logger().info(
            f"GNSS/IMU heading seeded from the {source}: {detail} after "
            f"{self._alignment_travel:.2f} m; publishing "
            f"{len(self._alignment_samples)} buffered poses. The RTK course filters "
            f"it above {self._min_heading_speed:.2f} m/s and the first "
            f"{self._min_distance:.0f} m straight segment replaces it"
        )
        for buffered in self._alignment_samples:
            self._publish_sample(buffered)
        self._alignment_samples.clear()
        self._alignment_cumulative.clear()
        self._alignment_travel = 0.0
        return True

    def _try_finish_initial_alignment(self) -> None:
        # Judge straightness over the most recent stretch only, not everything
        # since launch. The whole-buffer chord could never pass once the run had
        # a standstill wobble, a reversing manoeuvre or a turn in it: on
        # 2026-09-17 the vehicle drove 82 m by hand and the heading never
        # committed. The buffered poses from the run start are still all
        # published below - only the test window is narrowed.
        start = window_start_index(
            self._alignment_cumulative, 2.0 * self._min_distance
        )
        chord_result = self._alignment_chord(start)
        if chord_result is None:
            return
        chord, straightness, _, initial_yaw = chord_result
        if not self._using_saved_georeference:
            first_sample = self._alignment_samples[0]
            initial_orientation = quaternion_from_euler(
                initial_yaw,
                first_sample.pitch,
                first_sample.roll,
            )
            rotated_lever = rotate_vector(initial_orientation, self._gps_lever)
            self._t_enu_map = tuple(
                first_sample.antenna_enu[index] - rotated_lever[index]
                for index in range(3)
            )
            self._q_enu_map = initial_orientation

        self._aligned = True
        self._heading = initial_yaw
        self._publish_georeference()
        self.get_logger().info(
            "GNSS/IMU heading ready after %.2f m: chord=%.2f m, "
            "straightness=%.3f, initial ENU yaw=%.2f deg; publishing %d "
            "buffered poses"
            % (
                self._alignment_travel,
                chord,
                straightness,
                math.degrees(initial_yaw),
                len(self._alignment_samples),
            )
        )
        for sample in self._alignment_samples:
            self._publish_sample(sample)
        self._alignment_samples.clear()
        self._alignment_cumulative.clear()

    def _publish_georeference(self) -> None:
        if (
            self._georeference_published
            or self._origin is None
            or self._t_enu_map is None
            or self._q_enu_map is None
        ):
            return

        latitude, longitude, altitude = self._origin
        easting, northing, height, zone, band = geodetic_to_utm(
            latitude, longitude, altitude
        )
        stamp = self.get_clock().now().to_msg()

        enu_to_map = TransformStamped()
        enu_to_map.header.stamp = stamp
        enu_to_map.header.frame_id = self._enu_frame
        enu_to_map.child_frame_id = self._map_frame
        (
            enu_to_map.transform.translation.x,
            enu_to_map.transform.translation.y,
            enu_to_map.transform.translation.z,
        ) = self._t_enu_map
        (
            enu_to_map.transform.rotation.x,
            enu_to_map.transform.rotation.y,
            enu_to_map.transform.rotation.z,
            enu_to_map.transform.rotation.w,
        ) = self._q_enu_map

        # This matches BridgeROS2/MRPT convention: the UTM frame origin,
        # expressed in ENU, is the negative UTM coordinate of the ENU origin.
        enu_to_utm = TransformStamped()
        enu_to_utm.header.stamp = stamp
        enu_to_utm.header.frame_id = self._enu_frame
        enu_to_utm.child_frame_id = self._utm_frame
        enu_to_utm.transform.translation.x = -easting
        enu_to_utm.transform.translation.y = -northing
        enu_to_utm.transform.translation.z = -height
        enu_to_utm.transform.rotation.w = 1.0

        # ONE call with both edges. rclpy's StaticTransformBroadcaster keeps no
        # accumulator (unlike the C++ class): each sendTransform publishes only what
        # it is given, onto a depth-1 TRANSIENT_LOCAL /tf_static publisher. Two calls
        # therefore evict the first from the durability cache, and any subscriber that
        # matches later - an RViz or tf2_echo started mid-run - receives enu->utm only
        # and can never connect "map" to "utm". Every static edge this node owns must
        # go in this single list.
        self._static_tf_broadcaster.sendTransform([enu_to_map, enu_to_utm])

        metadata = GeoreferencingMetadata()
        metadata.valid = True
        metadata.latitude = latitude
        metadata.longitude = longitude
        metadata.height = altitude
        metadata.utm_zone = zone
        metadata.utm_band = band
        (
            metadata.t_enu_to_map.pose.position.x,
            metadata.t_enu_to_map.pose.position.y,
            metadata.t_enu_to_map.pose.position.z,
        ) = self._t_enu_map
        (
            metadata.t_enu_to_map.pose.orientation.x,
            metadata.t_enu_to_map.pose.orientation.y,
            metadata.t_enu_to_map.pose.orientation.z,
            metadata.t_enu_to_map.pose.orientation.w,
        ) = self._q_enu_map
        for index in (0, 7, 14, 21, 28, 35):
            metadata.t_enu_to_map.covariance[index] = 1.0e-6
        metadata.t_enu_to_utm.position.x = -easting
        metadata.t_enu_to_utm.position.y = -northing
        metadata.t_enu_to_utm.position.z = -height
        metadata.t_enu_to_utm.orientation.w = 1.0
        self._georeference_publisher.publish(metadata)
        self._georeference_published = True
        self.get_logger().info(
            "Published static TF enu->map and enu->utm "
            f"(UTM zone {zone}{band}, origin={easting:.3f}, "
            f"{northing:.3f}, {height:.3f})"
        )


    def _publish_sample(self, sample: PoseSample) -> None:
        if self._t_enu_map is None or self._q_enu_map is None:
            return
        if sample.course is not None:
            self._heading = (
                sample.course
                if self._heading is None
                else circular_lerp(
                    self._heading,
                    sample.course,
                    self._heading_alpha,
                )
            )
        if self._heading is None:
            return

        q_enu_base = quaternion_from_euler(
            self._heading,
            sample.pitch,
            sample.roll,
        )
        rotated_lever = rotate_vector(q_enu_base, self._gps_lever)
        base_enu = tuple(
            sample.antenna_enu[index] - rotated_lever[index]
            for index in range(3)
        )
        q_map_enu = inverse_quaternion(self._q_enu_map)
        position_map = rotate_vector(
            q_map_enu,
            tuple(
                base_enu[index] - self._t_enu_map[index]
                for index in range(3)
            ),
        )
        q_map_base = multiply_quaternions(q_map_enu, q_enu_base)

        output = Odometry()
        # Preserve measurement time. BridgeROS2 publishes LiDAR odometry in the
        # same sensor-time domain, so map->odom can interpolate matching poses.
        output.header.stamp.sec = sample.stamp_ns // 1_000_000_000
        output.header.stamp.nanosec = sample.stamp_ns % 1_000_000_000
        output.header.frame_id = self._map_frame
        output.child_frame_id = self._base_frame
        output.pose.pose.position.x = position_map[0]
        output.pose.pose.position.y = position_map[1]
        output.pose.pose.position.z = position_map[2]
        (
            output.pose.pose.orientation.x,
            output.pose.pose.orientation.y,
            output.pose.pose.orientation.z,
            output.pose.pose.orientation.w,
        ) = q_map_base

        covariance = list(sample.fix.position_covariance)
        output.pose.covariance[0] = max(covariance[0], 0.05**2)
        output.pose.covariance[7] = max(covariance[4], 0.05**2)
        output.pose.covariance[14] = max(covariance[8], 1.0**2)
        output.pose.covariance[21] = math.radians(2.0) ** 2
        output.pose.covariance[28] = math.radians(2.0) ** 2
        output.pose.covariance[35] = math.radians(2.0) ** 2

        output.twist.twist.linear.x = sample.speed
        output.twist.twist.angular.z = sample.angular_z
        velocity_covariance = sample.velocity.twist.covariance
        output.twist.covariance[0] = max(
            velocity_covariance[0],
            velocity_covariance[7],
            0.05**2,
        )
        output.twist.covariance[7] = 0.5**2
        output.twist.covariance[14] = 1.0
        output.twist.covariance[21] = 0.2**2
        output.twist.covariance[28] = 0.2**2
        output.twist.covariance[35] = 0.1**2
        self._publisher.publish(output)
        if not self._published_count:
            self.get_logger().info(
                "First /gnss_imu/pose published; map->odom and the UTM route "
                "projection can now proceed."
            )
        self._published_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = GnssImuPoseNode()
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
