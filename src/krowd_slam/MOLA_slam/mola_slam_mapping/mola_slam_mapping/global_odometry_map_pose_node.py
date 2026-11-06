"""Project ESKF global odometry into the selected saved-map frame."""

from copy import deepcopy
import math

from mrpt_nav_interfaces.msg import GeoreferencingMetadata
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)


WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def normalize_quaternion(quaternion):
    values = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm < 1.0e-12:
        raise ValueError("invalid quaternion")
    return values / norm


def quaternion_from_euler(yaw: float, pitch: float, roll: float):
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    return normalize_quaternion(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def matrix_from_quaternion(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quaternion_from_matrix(matrix):
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("invalid rotation matrix")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            ) * 2.0
            quaternion = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            ) * 2.0
            quaternion = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(
                max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            ) * 2.0
            quaternion = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    return normalize_quaternion(quaternion)


def geodetic_to_ecef(origin):
    latitude, longitude, altitude = (float(value) for value in origin)
    latitude = math.radians(latitude)
    longitude = math.radians(longitude)
    normal = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(latitude) ** 2)
    return np.array(
        [
            (normal + altitude) * math.cos(latitude) * math.cos(longitude),
            (normal + altitude) * math.cos(latitude) * math.sin(longitude),
            (normal * (1.0 - WGS84_E2) + altitude) * math.sin(latitude),
        ]
    )


def enu_from_ecef_rotation(origin):
    latitude = math.radians(float(origin[0]))
    longitude = math.radians(float(origin[1]))
    return np.array(
        [
            [-math.sin(longitude), math.cos(longitude), 0.0],
            [
                -math.sin(latitude) * math.cos(longitude),
                -math.sin(latitude) * math.sin(longitude),
                math.cos(latitude),
            ],
            [
                math.cos(latitude) * math.cos(longitude),
                math.cos(latitude) * math.sin(longitude),
                math.sin(latitude),
            ],
        ]
    )


def map_from_ekf_enu_transform(
    ekf_origin,
    map_origin,
    translation_enu_from_map,
    quaternion_enu_from_map,
):
    """Return map<-ESKF_ENU while allowing the two ENU origins to differ."""

    map_enu_from_ecef = enu_from_ecef_rotation(map_origin)
    ekf_enu_from_ecef = enu_from_ecef_rotation(ekf_origin)
    map_enu_from_ekf_enu = map_enu_from_ecef @ ekf_enu_from_ecef.T
    map_enu_from_ekf_origin = map_enu_from_ecef @ (
        geodetic_to_ecef(ekf_origin) - geodetic_to_ecef(map_origin)
    )

    map_enu_from_map = matrix_from_quaternion(quaternion_enu_from_map)
    map_from_map_enu = map_enu_from_map.T
    rotation = map_from_map_enu @ map_enu_from_ekf_enu
    translation = map_from_map_enu @ (
        map_enu_from_ekf_origin - np.asarray(translation_enu_from_map, dtype=float)
    )
    return rotation, translation


def transform_global_odometry(
    message: Odometry,
    rotation_map_from_enu,
    translation_map_from_enu,
    expected_frame: str = "enu",
    map_frame: str = "map",
) -> Odometry:
    """Rigidly transform an ENU<-base Odometry pose into map<-base."""

    if message.header.frame_id.strip() != expected_frame:
        raise ValueError(
            f"expected odometry frame '{expected_frame}', got "
            f"'{message.header.frame_id}'"
        )
    pose = message.pose.pose
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    orientation = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    if not np.all(np.isfinite(position)):
        raise ValueError("odometry contains a non-finite position")
    rotation = np.asarray(rotation_map_from_enu, dtype=float)
    translation = np.asarray(translation_map_from_enu, dtype=float)
    map_position = rotation @ position + translation
    map_orientation = quaternion_from_matrix(
        rotation @ matrix_from_quaternion(orientation)
    )

    covariance = np.asarray(message.pose.covariance, dtype=float).reshape(6, 6)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("odometry contains a non-finite pose covariance")
    covariance_rotation = np.zeros((6, 6))
    covariance_rotation[:3, :3] = rotation
    covariance_rotation[3:, 3:] = rotation
    map_covariance = covariance_rotation @ covariance @ covariance_rotation.T

    output = Odometry()
    output.header = deepcopy(message.header)
    output.header.frame_id = map_frame
    output.child_frame_id = message.child_frame_id
    (
        output.pose.pose.position.x,
        output.pose.pose.position.y,
        output.pose.pose.position.z,
    ) = (float(value) for value in map_position)
    (
        output.pose.pose.orientation.x,
        output.pose.pose.orientation.y,
        output.pose.pose.orientation.z,
        output.pose.pose.orientation.w,
    ) = (float(value) for value in map_orientation)
    output.pose.covariance = map_covariance.ravel().tolist()
    # nav_msgs/Odometry defines twist in child_frame_id, so changing only the
    # parent/world frame must not rotate it.
    output.twist = deepcopy(message.twist)
    return output


class GlobalOdometryMapPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("global_odometry_map_pose_node")
        self.declare_parameter("input_topic", "/ekf/global_odometry")
        self.declare_parameter("origin_topic", "/ekf/georeference")
        self.declare_parameter("output_topic", "/localization/reference_pose")
        self.declare_parameter("input_frame", "enu")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("map_origin_latitude", 0.0)
        self.declare_parameter("map_origin_longitude", 0.0)
        self.declare_parameter("map_origin_altitude", 0.0)
        self.declare_parameter("enu_from_map_x", 0.0)
        self.declare_parameter("enu_from_map_y", 0.0)
        self.declare_parameter("enu_from_map_z", 0.0)
        self.declare_parameter("enu_from_map_yaw_deg", 0.0)
        self.declare_parameter("enu_from_map_pitch_deg", 0.0)
        self.declare_parameter("enu_from_map_roll_deg", 0.0)

        self._input_frame = str(self.get_parameter("input_frame").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._map_origin = tuple(
            float(self.get_parameter(name).value)
            for name in (
                "map_origin_latitude",
                "map_origin_longitude",
                "map_origin_altitude",
            )
        )
        self._enu_from_map_translation = np.array(
            [
                float(self.get_parameter("enu_from_map_x").value),
                float(self.get_parameter("enu_from_map_y").value),
                float(self.get_parameter("enu_from_map_z").value),
            ]
        )
        self._enu_from_map_quaternion = quaternion_from_euler(
            math.radians(float(self.get_parameter("enu_from_map_yaw_deg").value)),
            math.radians(float(self.get_parameter("enu_from_map_pitch_deg").value)),
            math.radians(float(self.get_parameter("enu_from_map_roll_deg").value)),
        )
        if not np.all(np.isfinite(self._map_origin)):
            raise ValueError("selected map has a non-finite geodetic origin")

        self._transform = None
        self._publisher = self.create_publisher(
            Odometry, str(self.get_parameter("output_topic").value), 20
        )
        latched_qos = QoSProfile(depth=1)
        latched_qos.reliability = ReliabilityPolicy.RELIABLE
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            GeoreferencingMetadata,
            str(self.get_parameter("origin_topic").value),
            self._on_origin,
            latched_qos,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("input_topic").value),
            self._on_odometry,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "Waiting for the ESKF origin, then publishing LiDAR-independent "
            f"map pose on '{self.get_parameter('output_topic').value}'"
        )

    def _on_origin(self, message: GeoreferencingMetadata) -> None:
        origin = (message.latitude, message.longitude, message.height)
        if not message.valid or not np.all(np.isfinite(origin)):
            self.get_logger().warning(
                "Ignoring invalid ESKF geodetic-origin metadata",
                throttle_duration_sec=5.0,
            )
            return
        self._transform = map_from_ekf_enu_transform(
            origin,
            self._map_origin,
            self._enu_from_map_translation,
            self._enu_from_map_quaternion,
        )
        self.get_logger().info(
            "ESKF global odometry aligned to selected map georeference "
            f"(ESKF origin={origin[0]:.8f}, {origin[1]:.8f})"
        )

    def _on_odometry(self, message: Odometry) -> None:
        if self._transform is None:
            self.get_logger().warning(
                "Map pose waiting for /ekf/georeference",
                throttle_duration_sec=5.0,
            )
            return
        try:
            output = transform_global_odometry(
                message,
                *self._transform,
                expected_frame=self._input_frame,
                map_frame=self._map_frame,
            )
        except ValueError as error:
            self.get_logger().warning(
                f"Rejecting ESKF global pose: {error}",
                throttle_duration_sec=5.0,
            )
            return
        self._publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalOdometryMapPoseNode()
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
