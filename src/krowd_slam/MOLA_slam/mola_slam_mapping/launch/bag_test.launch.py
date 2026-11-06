from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """TEST ONLY. Bundles the 3 pieces previously run by hand in separate
    terminals for replaying the URBAN_C0 bag: mapping.launch.py (in
    wheel_odom_source:=bag mode), the static TF publishers this bag needs
    (it has no /tf of its own), and `ros2 bag play` itself.

    URBAN_C0-specific, not for the real vehicle - see AI_CONTEXT.md's
    GNSS-clock-skew note for why this bag eventually crashes mola-cli a few
    seconds in regardless of TF/extrinsics accuracy.
    """
    package_share = Path(get_package_share_directory("mola_slam_mapping"))
    mapping_launch = package_share / "launch" / "mapping.launch.py"

    return LaunchDescription([
        DeclareLaunchArgument(
            "bag_path", default_value=str(Path.home() / "URBAN_C0_ros2_fixed")
        ),
        DeclareLaunchArgument("bag_rate", default_value="1.0"),
        # Gives mola-cli time to fully come up (module init, ROS2 subscriptions)
        # before the bag starts publishing - matches the old play_bag.sh's
        # `sleep 6`.
        DeclareLaunchArgument("bag_start_delay", default_value="6.0"),
        DeclareLaunchArgument("viz", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(mapping_launch)),
            launch_arguments={
                "wheel_odom_source": "bag",
                "viz": LaunchConfiguration("viz"),
                "rviz": LaunchConfiguration("rviz"),
            }.items(),
        ),

        # URBAN_C0 has no recorded /tf: base_link<->os_sensor and base_link<->gps
        # are identity (no calibration available for either), base_link<->imu_link
        # is the real extrinsic computed from the dataset's own
        # CALIBRATION_CAMERA_IMU.txt + CALIBRATION_CAMERA_LIDAR.txt (Kalibr
        # T_ic/T_ci chained through the camera frame). See AI_CONTEXT.md.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_base_link_to_os_sensor",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                "--frame-id", "base_link", "--child-frame-id", "os_sensor",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_base_link_to_imu_link",
            arguments=[
                "--x", "0.815256", "--y", "-0.062407", "--z", "-0.317835",
                "--qx", "0.004332", "--qy", "0.007699",
                "--qz", "0.002003", "--qw", "0.999959",
                "--frame-id", "base_link", "--child-frame-id", "imu_link",
            ],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_base_link_to_gps",
            arguments=[
                "--x", "0", "--y", "0", "--z", "0",
                "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
                "--frame-id", "base_link", "--child-frame-id", "gps",
            ],
        ),

        TimerAction(
            period=LaunchConfiguration("bag_start_delay"),
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "bag", "play", LaunchConfiguration("bag_path"),
                        "--rate", LaunchConfiguration("bag_rate"),
                    ],
                    output="screen",
                ),
            ],
        ),
    ])
