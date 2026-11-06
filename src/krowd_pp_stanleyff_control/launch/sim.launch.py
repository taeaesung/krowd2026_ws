"""Simulation side: vehicle model + map stand-in + RViz. The controller is NOT here.

    # terminal 1 - the simulated vehicle and map (instead of the sensor stack + localization.launch.py)
    ros2 launch krowd_pp_stanleyff_control sim.launch.py map_dir:=~/krowd_maps/20260913_103732
    # terminal 2 - the controller, exactly as on the vehicle, pointed at the simulator
    ros2 launch krowd_pp_stanleyff_control control.launch.py sim:=true

This launch plays the parts the vehicle and the map launch play for real:
it publishes the run's utm -> map georeference, announces the run on
/localization/map_dir (so the route publisher and the controller pick the map
up unchanged), and drives a bicycle model from /simulation/target_* whose pose
comes back on /simulation/odometry and as map -> base_link. RViz starts here
(rviz/sim.rviz, Fixed Frame map, camera following the vehicle); the control
launch never starts RViz. No CAN topic is used.

Run it WITHOUT localization.launch.py and the sensor stack: they publish the
same TF edges. Restart this launch to put the vehicle back at the start.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE = "krowd_pp_stanleyff_control"


def _source_file(package: str, relative: str) -> str:
    """<relative> from the package's SOURCE tree when present, else the installed copy."""

    share = Path(get_package_share_directory(package))
    installed = share / relative
    workspace = share.parents[3]
    for depth in range(4):
        pattern = "src/" + "*/" * depth + package
        for candidate in sorted(workspace.glob(pattern)):
            source = candidate / relative
            if (candidate / "package.xml").is_file() and source.exists():
                return str(source)
    return str(installed)


def generate_launch_description():
    control_cfg = _source_file(PACKAGE, "config/control.yaml")
    sim_cfg = _source_file(PACKAGE, "config/sim_overrides.yaml")
    rviz_cfg = _source_file(PACKAGE, "rviz/sim.rviz")

    def number(name, kind=float):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    return LaunchDescription([
        DeclareLaunchArgument("map_dir", description="Mapping run directory (waypoints.csv, georeference.yaml)."),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("initial_waypoint_index", default_value="0"),
        DeclareLaunchArgument("initial_lateral_offset_m", default_value="0.0"),
        DeclareLaunchArgument("initial_heading_offset_deg", default_value="0.0"),
        DeclareLaunchArgument("steering_lag_sec", default_value="0.24",
                              description="Steering dead time (measured 2026-09-22). 0 = ideal."),
        DeclareLaunchArgument("speed_deficit_kmph", default_value="1.7",
                              description="How far the drive MCU runs under /target_speed (measured 2026-09-22). 0 = ideal."),
        DeclareLaunchArgument("coast_decel_mps2", default_value="0.5",
                              description="Plant coasting deceleration; set the measured value to test the profile."),
        LogInfo(msg=["simulation: vehicle model + map ", LaunchConfiguration("map_dir"),
                     " - start the controller with: ros2 launch ", PACKAGE, " control.launch.py sim:=true"]),
        Node(
            package=PACKAGE,
            executable="vehicle_dynamics_sim_node",
            name="vehicle_dynamics_sim_node",
            output="screen",
            parameters=[{
                "control_config": control_cfg,
                "control_overrides": sim_cfg,
                "map_dir": LaunchConfiguration("map_dir"),
                "frame_id": "map",
                "vehicle_marker_topic": "/simulation/vehicle_marker",
                "steering_lag_sec": number("steering_lag_sec"),
                "speed_deficit_kmph": number("speed_deficit_kmph"),
                "coast_decel_mps2": number("coast_decel_mps2"),
                "initial_waypoint_index": number("initial_waypoint_index", int),
                "initial_lateral_offset_m": number("initial_lateral_offset_m"),
                "initial_heading_offset_deg": number("initial_heading_offset_deg"),
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_sim",
            arguments=["-d", rviz_cfg],
            output="log",
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
