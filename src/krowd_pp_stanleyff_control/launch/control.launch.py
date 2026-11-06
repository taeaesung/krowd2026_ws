"""Route publisher + PP/Stanley controller. The same launch on the vehicle and in simulation.

Never starts RViz: on the vehicle the map launch (localization.launch.py)
opens it, in simulation sim.launch.py does.

    # vehicle - on top of the sensor stack and localization.launch.py
    ros2 launch krowd_pp_stanleyff_control control.launch.py
    #   follows whichever map localization announced on /localization/map_dir;
    #   config/maps/<run>/profile.yaml selects missions/segments/lane walls,
    #   a run without one drives route_only.

    # simulation - on top of sim.launch.py
    ros2 launch krowd_pp_stanleyff_control control.launch.py sim:=true

sim:=true only changes where the controller connects: control_overrides
defaults to config/sim_overrides.yaml (pose from /simulation/odometry,
commands on /simulation/target_*, no ESKF heading gate).

There is no arm step: the controller publishes to the actuator topics from its
first cycle - a stop (NOT_READY) until the inputs are ready, then it tracks by
itself. The physical ASMS / E-stop is the only gate.

speed_mode:=low|normal|high (default normal) picks config/speed_modes/<mode>.yaml:
its `route_publisher` values go over speed_profile.yaml, its `control` values
over control.yaml (under control_overrides). Restart the launch to change it.
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PACKAGE = "krowd_pp_stanleyff_control"
NODE_NAME = "pp_stanleyff_control_node"
TRUE = ("true", "1", "yes", "on")


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


def _speed_mode(name: str) -> tuple[str, dict]:
    """(path, route_publisher parameters) of config/speed_modes/<name>.yaml."""

    path = Path(_source_file(PACKAGE, f"config/speed_modes/{name}.yaml"))
    if not path.is_file():
        available = sorted(p.stem for p in path.parent.glob("*.yaml")) if path.parent.is_dir() else []
        raise RuntimeError(f"speed_mode '{name}': no {path} (available: {', '.join(available) or 'none'})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = dict(data.get("route_publisher") or {})
    # ROS rejects an integer for a parameter declared as a double: 40 -> 40.0.
    for key, value in params.items():
        if isinstance(value, int) and not isinstance(value, bool):
            params[key] = float(value)
    return str(path), params


def _nodes(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context).strip()

    sim = arg("sim").lower() in TRUE
    overrides = arg("control_overrides") or (_source_file(PACKAGE, "config/sim_overrides.yaml") if sim else "")
    use_sim_time = arg("use_sim_time").lower() in TRUE
    map_dir = arg("map_dir")
    mode = arg("speed_mode").lower() or "normal"
    mode_path, mode_profile = _speed_mode(mode)
    return [
        LogInfo(msg=f"{PACKAGE}: {'SIMULATION (/simulation/* topics)' if sim else 'VEHICLE (/target_angle /target_speed /brake)'}"
                    f", speed_mode={mode} {mode_profile}, overrides={overrides or 'none'}"
                    f", map={map_dir or '/localization/map_dir'}"),
        Node(
            package=PACKAGE,
            executable="route_publisher_node",
            name="route_publisher_node",
            output="screen",
            parameters=[
                arg("speed_profile_config"),
                mode_profile,
                {
                    "map_dir": map_dir,
                    "speed_limit_override_kmph": float(arg("speed_limit_kmph") or 0.0),
                    "speed_mode": mode,
                    "use_sim_time": use_sim_time,
                },
            ],
        ) if arg("start_route_publisher").lower() in TRUE else LogInfo(msg="route publisher not started"),
        Node(
            package=PACKAGE,
            executable="control_node",
            name=NODE_NAME,
            output="screen",
            parameters=[{
                "control_config": arg("control_config"),
                "control_overrides": overrides,
                "speed_mode_config": mode_path,
                "maps_dir": arg("maps_dir"),
                "map_profile": arg("map_profile"),
                "map_dir": map_dir,
                "force_route_only": arg("force_route_only").lower() in TRUE,
                "use_sim_time": use_sim_time,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="false",
                              description="true = connect to sim.launch.py (/simulation/* topics)."),
        DeclareLaunchArgument("map_dir", default_value="",
                              description="Localization map directory (waypoints.csv). Empty = follow /localization/map_dir."),
        DeclareLaunchArgument("control_config", default_value=_source_file(PACKAGE, "config/control.yaml")),
        DeclareLaunchArgument("control_overrides", default_value="",
                              description="YAML merged over control_config. Empty = sim_overrides.yaml with sim:=true, none otherwise."),
        DeclareLaunchArgument("speed_profile_config", default_value=_source_file(PACKAGE, "config/speed_profile.yaml")),
        DeclareLaunchArgument("maps_dir", default_value=_source_file(PACKAGE, "config/maps")),
        DeclareLaunchArgument("map_profile", default_value="",
                              description="Explicit profile directory; empty = config/maps/<run> for the announced map."),
        DeclareLaunchArgument("force_route_only", default_value="false"),
        DeclareLaunchArgument("speed_mode", default_value="normal",
                              description="low (10/15 km/h) | normal (10/25) | high (10/40): config/speed_modes/<mode>.yaml"),
        DeclareLaunchArgument("speed_limit_kmph", default_value="0.0",
                              description="Whole-profile ceiling for a cautious trial; 0 = the profile as configured."),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("start_route_publisher", default_value="true"),
        OpaqueFunction(function=_nodes),
    ])
