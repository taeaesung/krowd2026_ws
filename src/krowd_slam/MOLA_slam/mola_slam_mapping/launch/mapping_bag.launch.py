from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _static_tf_node(name, parent, child, xyz, quaternion):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        arguments=[
            "--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
            "--qx", str(quaternion[0]), "--qy", str(quaternion[1]),
            "--qz", str(quaternion[2]), "--qw", str(quaternion[3]),
            "--frame-id", parent, "--child-frame-id", child,
        ],
    )


def generate_launch_description():
    """Start mapping with fixed calibration; optionally play sensor topics only."""

    package_share = Path(get_package_share_directory("mola_slam_mapping"))
    mapping_launch = package_share / "launch" / "mapping.launch.py"
    default_lidar_pipeline = str(
        Path(get_package_share_directory("mola_lidar_odometry"))
        / "pipelines"
        / "lidar3d-dlio-like.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path", default_value=str(Path.home() / "0824_5" / "0824_5")
        ),
        # A full 1x replay can outrun this CPU once GICP, map publication, and
        # RViz run together.  That makes scan selection timing-dependent and
        # was measured to produce 11-15% dropped LiDAR frames plus inconsistent
        # vertical trajectories on 0824_5.  Half speed keeps the sensor data and
        # timestamps unchanged while giving the online pipeline enough wall time.
        DeclareLaunchArgument("bag_rate", default_value="0.5"),
        DeclareLaunchArgument("bag_start_delay", default_value="8.0"),
        DeclareLaunchArgument("auto_play", default_value="false"),
        DeclareLaunchArgument("lidar_topic", default_value="/ouster/points"),
        DeclareLaunchArgument("imu_topic", default_value="/imu/data_ekf"),
        DeclareLaunchArgument("gnss_topic", default_value="/ublox_gps_node/fix"),
        DeclareLaunchArgument("gnss_velocity_topic", default_value="/ublox_gps_node/fix_velocity"),
        DeclareLaunchArgument("lidar_pipeline", default_value=default_lidar_pipeline),
        DeclareLaunchArgument(
            "deskew_method", default_value="MotionCompensationMethod::Linear"
        ),
        DeclareLaunchArgument("publish_in_sim_time", default_value="true"),
        DeclareLaunchArgument("lo_pose_step_base_m", default_value="0.20"),
        DeclareLaunchArgument("lo_max_linear_speed_mps", default_value="12.0"),
        DeclareLaunchArgument("lo_rotation_step_base_deg", default_value="3.0"),
        DeclareLaunchArgument("lo_max_angular_speed_deg_s", default_value="60.0"),
        DeclareLaunchArgument("wheel_odom_source", default_value="none"),
        # The bag carries the vehicle ESKF's own nav_msgs/Odometry
        # ("odom"->"base_link", body-frame twist), which BridgeROS2 subscribes to
        # directly as WHEEL_ODOM_TOPIC - no converter node, so it pairs with
        # wheel_odom_source:="none". Forwarded here because a bag replay is
        # exactly where LiDAR odometry has no other linear-velocity prior.
        DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel_odom"),
        DeclareLaunchArgument("viz", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        # Recorded /tf is excluded below, so this wrapper safely owns the sole
        # dynamic odom->base_link edge and closes the complete visualization tree.
        DeclareLaunchArgument("base_link_tf_source", default_value="lidar"),
        # These re-declare mapping.launch.py's arguments and OVERRIDE its
        # defaults when forwarded below, so they must match it: the saved
        # rolling_local_map.mm only covers this radius around where mapping ends
        # (see mapping.launch.py). A first fix that changed only mapping.launch.py
        # left every bag replay still truncating at 75 m.
        DeclareLaunchArgument("incremental_map_half_size_m", default_value="200.0"),
        DeclareLaunchArgument("keyframe_history_radius_m", default_value="250.0"),
        DeclareLaunchArgument("map_removal_check_every_n", default_value="20"),
        DeclareLaunchArgument("simplemap_revisit_window_s", default_value="30.0"),
        DeclareLaunchArgument("rviz_map_voxel_size_m", default_value="0.25"),
        DeclareLaunchArgument("fusion_imu_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("gnss_min_xy_sigma_m", default_value="0.05"),
        DeclareLaunchArgument("gnss_min_z_sigma_m", default_value="1.0"),
        DeclareLaunchArgument(
            "initial_alignment_min_distance_m", default_value="5.0"
        ),
        DeclareLaunchArgument(
            "initial_alignment_min_straightness_ratio",
            default_value="0.95",
        ),
        DeclareLaunchArgument(
            "initial_alignment_yaw_window_sec", default_value="2.0"
        ),
        DeclareLaunchArgument(
            "initial_alignment_min_yaw_samples", default_value="5"
        ),
        DeclareLaunchArgument(
            "initial_alignment_max_yaw_spread_deg",
            default_value="3.0",
        ),
        # Forwarded so a bag replay can be repeated with each estimator and the
        # results compared. See mapping.launch.py for the values.
        DeclareLaunchArgument(
            "state_estimator",
            default_value="mola::state_estimation_simple::StateEstimationSimple",
        ),
        DeclareLaunchArgument(
            "correction_filter_weight", default_value="0.3"
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(mapping_launch)),
            launch_arguments={
                "state_estimator": LaunchConfiguration("state_estimator"),
                "correction_filter_weight": LaunchConfiguration(
                    "correction_filter_weight"
                ),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "gnss_topic": LaunchConfiguration("gnss_topic"),
                "gnss_velocity_topic": LaunchConfiguration("gnss_velocity_topic"),
                "lidar_pipeline": LaunchConfiguration("lidar_pipeline"),
                "deskew_method": LaunchConfiguration("deskew_method"),
                "publish_in_sim_time": LaunchConfiguration("publish_in_sim_time"),
                "lo_pose_step_base_m": LaunchConfiguration(
                    "lo_pose_step_base_m"
                ),
                "lo_max_linear_speed_mps": LaunchConfiguration(
                    "lo_max_linear_speed_mps"
                ),
                "lo_rotation_step_base_deg": LaunchConfiguration(
                    "lo_rotation_step_base_deg"
                ),
                "lo_max_angular_speed_deg_s": LaunchConfiguration(
                    "lo_max_angular_speed_deg_s"
                ),
                "wheel_odom_source": LaunchConfiguration("wheel_odom_source"),
                "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                "viz": LaunchConfiguration("viz"),
                "rviz": LaunchConfiguration("rviz"),
                "base_link_tf_source": LaunchConfiguration("base_link_tf_source"),
                "incremental_map_half_size_m": LaunchConfiguration(
                    "incremental_map_half_size_m"
                ),
                "keyframe_history_radius_m": LaunchConfiguration(
                    "keyframe_history_radius_m"
                ),
                "map_removal_check_every_n": LaunchConfiguration(
                    "map_removal_check_every_n"
                ),
                "simplemap_revisit_window_s": LaunchConfiguration(
                    "simplemap_revisit_window_s"
                ),
                "rviz_map_voxel_size_m": LaunchConfiguration(
                    "rviz_map_voxel_size_m"
                ),
                "fusion_imu_rate_hz": LaunchConfiguration("fusion_imu_rate_hz"),
                "gnss_min_xy_sigma_m": LaunchConfiguration(
                    "gnss_min_xy_sigma_m"
                ),
                "gnss_min_z_sigma_m": LaunchConfiguration(
                    "gnss_min_z_sigma_m"
                ),
                "initial_alignment_min_distance_m": LaunchConfiguration(
                    "initial_alignment_min_distance_m"
                ),
                "initial_alignment_min_straightness_ratio": LaunchConfiguration(
                    "initial_alignment_min_straightness_ratio"
                ),
                "initial_alignment_yaw_window_sec": LaunchConfiguration(
                    "initial_alignment_yaw_window_sec"
                ),
                "initial_alignment_min_yaw_samples": LaunchConfiguration(
                    "initial_alignment_min_yaw_samples"
                ),
                "initial_alignment_max_yaw_spread_deg": LaunchConfiguration(
                    "initial_alignment_max_yaw_spread_deg"
                ),
            }.items(),
        ),

        # Replay the bag's own calibration before any recorded sensor sample. In the
        # MCAP, the first GNSS/LiDAR samples precede some /tf_static records, which made
        # BridgeROS2 reject the first samples and could expose startup timing races.
        # These are exact copies of 0824_5's transforms. Recorded /tf and
        # /tf_static are deliberately not replayed: /tf contains the old ESKF's
        # enu->base_link trajectory and would give base_link a competing parent.
        _static_tf_node(
            "bag_tf_base_to_imu",
            "base_link", "imu_link",
            (0.3938, 0.0, 0.876),
            (-0.00927259539161, 0.030217093795, 0.00028033096354, 0.99950030897),
        ),
        _static_tf_node(
            "bag_tf_base_to_zed",
            "base_link", "zed_camera_link",
            (0.461, 0.0, 0.8576),
            (0.0, 0.0, 0.0, 1.0),
        ),
        _static_tf_node(
            "bag_tf_zed_to_ouster",
            "zed_camera_link", "os_sensor",
            (-0.0414949508, 0.0504791616, -0.0901744425),
            (0.00486081091, -0.00173969287, -0.0150764324, 0.999873016),
        ),
        _static_tf_node(
            "bag_tf_ouster_to_lidar",
            "os_sensor", "os_lidar",
            (0.0, 0.0, 0.03618),
            (0.0, 0.0, 1.0, 0.0),
        ),
        _static_tf_node(
            "bag_tf_base_to_gps",
            "base_link", "gps_link",
            (2.5038, 0.0, 1.021),
            (0.0, 0.0, 0.0, 1.0),
        ),

        LogInfo(
            condition=UnlessCondition(LaunchConfiguration("auto_play")),
            msg=[
                "Bag auto-play is OFF. In terminal 2 run: ros2 bag play ",
                LaunchConfiguration("bag_path"),
                " --rate ", LaunchConfiguration("bag_rate"),
                " --disable-keyboard-controls --topics ",
                LaunchConfiguration("lidar_topic"),
                " ", LaunchConfiguration("imu_topic"),
                " ", LaunchConfiguration("gnss_topic"),
                " ", LaunchConfiguration("gnss_velocity_topic"),
            ],
        ),

        # Let mola-cli finish module and ROS subscription setup before replay starts.
        TimerAction(
            period=LaunchConfiguration("bag_start_delay"),
            condition=IfCondition(LaunchConfiguration("auto_play")),
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "bag", "play", LaunchConfiguration("bag_path"),
                        "--rate", LaunchConfiguration("bag_rate"),
                        "--disable-keyboard-controls",
                        "--topics",
                        LaunchConfiguration("lidar_topic"),
                        LaunchConfiguration("imu_topic"),
                        LaunchConfiguration("gnss_topic"),
                        LaunchConfiguration("gnss_velocity_topic"),
                    ],
                    output="screen",
                ),
            ],
        ),
    ])
