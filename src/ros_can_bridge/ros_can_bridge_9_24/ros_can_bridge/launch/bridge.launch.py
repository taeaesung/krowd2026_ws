from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    can_interface_arg = DeclareLaunchArgument(
        "can_interface",
        default_value="can0",
    )
    can_interface = LaunchConfiguration("can_interface")

    return LaunchDescription([
        can_interface_arg,
        Node(
            package="ros_can_bridge",
            executable="ros_to_can_node",
            name="ros_to_can_node",
            output="screen",
            parameters=[{"can_interface": can_interface}],
        ),
        Node(
            package="ros_can_bridge",
            executable="can_to_ros_node",
            name="can_to_ros_node",
            output="screen",
            parameters=[{"can_interface": can_interface}],
        ),
    ])
