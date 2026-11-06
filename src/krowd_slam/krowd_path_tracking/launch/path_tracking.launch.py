"""Bring up route publication, the vehicle-state adapter and an optional controller.

Controllers are disabled by default so mission-specific controllers can own the
actuator topics safely. Set controller:=true for normal route driving, choosing
Pure Pursuit (default) or Stanley with controller_type. Both publish the same
actuator topics, so at most one is ever started.

Runs alongside a localization stack that is already producing a pose - it does
not start one. On the vehicle that is mola_slam_mapping's localization.launch.py.

Every path is a launch argument and every default that names a location is
relative to $HOME, so moving this package to another machine needs no edit
inside it.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare  # noqa: F401


def _source_config(package: str, name: str) -> Path:
    """config/<name> from the package's SOURCE tree when it can be found.

    colcon copies config/*.yaml into install/ at build time, so a launch that
    read the installed copy would ignore every edit until the next
    `colcon build`. Reading the source file instead makes an edited YAML take
    effect on the next launch. The workspace is found from the install prefix
    (install/<pkg>/share/<pkg> -> workspace), and the installed copy is the
    fallback when no source checkout exists (a deployed install).
    """

    share = Path(get_package_share_directory(package))
    installed = share / "config" / name
    workspace = share.parents[3]
    for depth in range(4):
        pattern = "src/" + "*/" * depth + package
        for candidate in sorted(workspace.glob(pattern)):
            source = candidate / "config" / name
            if (candidate / "package.xml").is_file() and source.is_file():
                return source
    return installed


def _controller_selected(launch_controller, controller_type, name):
    """Start `name` only when controllers are enabled and it is the one selected."""

    return IfCondition(
        PythonExpression(
            [
                "'",
                launch_controller,
                "'.lower() in ('true', '1') and '",
                controller_type,
                "' == '",
                name,
                "'",
            ]
        )
    )


def generate_launch_description():
    # Source-tree YAML (see _source_config): edit config/*.yaml, relaunch.
    vehicle_params = str(_source_config("krowd_path_tracking", "vehicle.yaml"))
    tracking_params = str(_source_config("krowd_path_tracking", "path_tracking.yaml"))

    map_dir = LaunchConfiguration("map_dir")
    waypoints_csv = LaunchConfiguration("waypoints_csv")
    corridor_csv = LaunchConfiguration("corridor_csv")
    vehicle_config = LaunchConfiguration("vehicle_config")
    tracking_config = LaunchConfiguration("tracking_config")
    launch_controller = LaunchConfiguration("controller")
    controller_type = LaunchConfiguration("controller_type")
    use_sim_time = LaunchConfiguration("use_sim_time")

    arguments = [
        DeclareLaunchArgument(
            "map_dir",
            default_value="",
            description=(
                "Mapping run directory holding waypoints.csv and corridor.csv. "
                "Empty (the default) follows whichever run localization loaded, "
                "which it announces on /localization/map_dir - so the map is "
                "named once, in localization.launch.py, and the two cannot end "
                "up on different runs."
            ),
        ),
        DeclareLaunchArgument(
            "waypoints_csv",
            default_value="",
            description="Overrides <map_dir>/waypoints.csv when set.",
        ),
        DeclareLaunchArgument(
            "corridor_csv",
            default_value="",
            description="Overrides <map_dir>/corridor.csv when set.",
        ),
        DeclareLaunchArgument(
            "vehicle_config",
            default_value=vehicle_params,
            description="Vehicle geometry YAML.",
        ),
        DeclareLaunchArgument(
            "tracking_config",
            default_value=tracking_params,
            description="Path tracking tuning YAML.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description=(
                "true when replaying a bag with --clock. These nodes compare "
                "timestamps they produce against timestamps they consume - the "
                "trajectory's valid_until, the vehicle state's age - so running "
                "them on a different clock from the localization stack makes "
                "everything look either fresh forever or permanently expired."
            ),
        ),
        DeclareLaunchArgument(
            "controller",
            default_value="false",
            description=(
                "Start the controller chosen by controller_type. Defaults to false "
                "so mission controllers such as stop_mission can own /target_angle, "
                "/target_speed and /brake without competing publishers. Set true "
                "explicitly for normal route driving."
            ),
        ),
        DeclareLaunchArgument(
            "controller_type",
            default_value="pure_pursuit",
            choices=["pure_pursuit", "stanley"],
            description=(
                "pure_pursuit_node or stanley_node. Same inputs and actuator "
                "topics; see the README before choosing Stanley."
            ),
        ),
    ]

    return LaunchDescription(
        arguments
        + [
            LogInfo(
                msg=[
                    "path tracking parameters: ",
                    vehicle_config,
                    " + ",
                    tracking_config,
                ]
            ),
            Node(
                package="krowd_path_tracking",
                executable="route_publisher_node",
                name="route_publisher_node",
                output="screen",
                parameters=[
                    vehicle_config,
                    tracking_config,
                    {
                        "map_dir": map_dir,
                        "waypoints_csv": waypoints_csv,
                        "corridor_csv": corridor_csv,
                        "use_sim_time": use_sim_time,
                    },
                ],
            ),
            Node(
                package="krowd_path_tracking",
                executable="vehicle_state_node",
                name="vehicle_state_node",
                output="screen",
                parameters=[
                    vehicle_config,
                    tracking_config,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="krowd_path_tracking",
                executable="pure_pursuit_node",
                name="pure_pursuit_node",
                output="screen",
                condition=_controller_selected(
                    launch_controller, controller_type, "pure_pursuit"
                ),
                parameters=[
                    vehicle_config,
                    tracking_config,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="krowd_path_tracking",
                executable="stanley_node",
                name="stanley_node",
                output="screen",
                condition=_controller_selected(
                    launch_controller, controller_type, "stanley"
                ),
                parameters=[
                    vehicle_config,
                    tracking_config,
                    {"use_sim_time": use_sim_time},
                ],
            ),
        ]
    )
