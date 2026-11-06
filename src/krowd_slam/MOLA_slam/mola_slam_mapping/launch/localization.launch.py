from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    SetEnvironmentVariable,
    SetLaunchConfiguration,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from mola_slam_mapping.georeference_io import load_fixed_georeference
from mola_slam_mapping.local_mola_runtime import local_mola_runtime
from mola_slam_mapping.traffic_detect_zones import ZONE_KINDS


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


def _sensor_static_tf_node(name, parent, child, xyz, quaternion):
    return Node(
        # On for a bag replay (bag:=...) whatever publish_sensor_static_tf says:
        # a replay has no sensor stack to own these edges.
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration("publish_sensor_static_tf"), "'.lower() in "
            "('true', '1') or '", LaunchConfiguration("replay"), "'.lower() in "
            "('true', '1') or '", LaunchConfiguration("bag"), "' != ''",
        ])),
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


def _georeference_filename(map_filename: str) -> str:
    """Which sidecar describes the frame a given saved map is stored in.

    rolling_local_map.mm has no sidecar of its own - it is the same MOLA map
    frame as global_map.mm, written by the same run, so it borrows that one.
    Every other map carries its own, and the distinction matters: a
    re-trajectoried global_map_corrected.mm from scripts/retraject_map.sh is
    stored in `map`, whereas global_map.mm is stored in `odom`, and on
    testweek_0905_3 those two frames differ by 36.5 deg and 75 m. Reading the
    wrong sidecar therefore does not merely shift the map, it silently gives
    LidarOdometry and gnss_imu_pose_node two different frames both called
    "map".
    """

    return "global_map.mm" if map_filename == "rolling_local_map.mm" else map_filename


def _configure_saved_map_georeference(context):
    map_directory = Path(
        LaunchConfiguration("map_dir").perform(context)
    ).expanduser()
    map_filename = LaunchConfiguration("map_filename").perform(context)
    map_path = map_directory / map_filename
    if not map_path.is_file():
        raise RuntimeError(f"Saved metric map does not exist: '{map_path}'")

    # Both mapping artifacts are written in the same MOLA map frame. The
    # DLIO-like localization pipeline must load the rolling map because its
    # IncrementalPointCloud representation matches the live scan matcher.
    # Verified 2026-09-03: map_filename:=global_map.mm loads, then every scan
    # dies in KeyframePointCloudMap::nn_search_cov2cov with "Implementation
    # expects local map to be also of type KeyframePointCloudMap" and no pose is
    # ever published. The rolling map, in turn, only covers
    # incremental_map_half_size_m (mapping.launch.py) around where mapping
    # ended - see that argument's comment.
    georeference_filename = _georeference_filename(map_filename)
    try:
        georeference = load_fixed_georeference(
            map_directory, georeference_filename
        )
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise RuntimeError(
            f"Cannot localize against '{map_directory}': valid saved-map "
            f"georeference is required ({error})"
        ) from error

    environment = {
        "MOLA_ESTIMATE_GEO_REFERENCE": "false",
        "MOLA_LINK_FIRST_POSE_SIGMA": "",
        "MOLA_LO_PUBLISH_REF_FRAME": "map",
        # Reuse the expanded path that was validated above. Passing the raw
        # LaunchConfiguration here would leave a leading '~' for the C++ map
        # loader, which does not perform shell-style home expansion.
        "MOLA_LOAD_MM": str(map_path),
        **georeference.as_environment(),
    }
    actions = [
        SetEnvironmentVariable(name, value) for name, value in environment.items()
    ]
    # Tell path tracking which run this is, so its route comes from the same one
    # without the directory being typed into both launches.
    actions.append(
        Node(
            package="mola_slam_mapping",
            executable="selected_map_node",
            name="selected_map_node",
            output="screen",
            parameters=[{
                "map_dir": str(map_directory),
                "map_dir_topic": LaunchConfiguration("map_dir_topic"),
            }],
        )
    )
    waypoints_csv = map_directory / "waypoints.csv"
    if (
        LaunchConfiguration("seed_initial_pose_from_route").perform(context).lower()
        in ("true", "1")
        and waypoints_csv.is_file()
    ):
        actions.append(
            Node(
                package="mola_slam_mapping",
                executable="route_initial_pose_node",
                name="route_initial_pose_node",
                output="screen",
                parameters=[{
                    "gps_topic": LaunchConfiguration("gnss_topic"),
                    "waypoints_csv": str(waypoints_csv),
                    "origin_latitude": georeference.latitude,
                    "origin_longitude": georeference.longitude,
                    "origin_altitude": georeference.altitude,
                    "enu_from_map_x": georeference.x,
                    "enu_from_map_y": georeference.y,
                    "enu_from_map_z": georeference.z,
                    "enu_from_map_yaw_deg": georeference.yaw_deg,
                    "maximum_route_distance_m": ParameterValue(
                        LaunchConfiguration("route_seed_max_distance_m"), value_type=float
                    ),
                    "yaw_sigma_deg": ParameterValue(
                        LaunchConfiguration("route_seed_yaw_sigma_deg"), value_type=float
                    ),
                }],
            )
        )
    # ESKF global odometry projected into the selected map, on
    # reference_pose_topic. This node accounts for both the ESKF ENU origin and
    # the selected map's saved ENU transform. It is not the vehicle pose drawn
    # by default (see vehicle_pose_topic): it needs the live ESKF's latched
    # /ekf/georeference, which a recorded bag never contains.
    actions.append(
        Node(
            package="mola_slam_mapping",
            executable="global_odometry_map_pose_node",
            name="global_odometry_map_pose_node",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("ekf_global_pose_topic"),
                "origin_topic": LaunchConfiguration("ekf_origin_topic"),
                "output_topic": LaunchConfiguration("reference_pose_topic"),
                "map_origin_latitude": georeference.latitude,
                "map_origin_longitude": georeference.longitude,
                "map_origin_altitude": georeference.altitude,
                "enu_from_map_x": georeference.x,
                "enu_from_map_y": georeference.y,
                "enu_from_map_z": georeference.z,
                "enu_from_map_yaw_deg": georeference.yaw_deg,
                "enu_from_map_pitch_deg": georeference.pitch_deg,
                "enu_from_map_roll_deg": georeference.roll_deg,
            }],
        )
    )
    # gnss_imu_pose_node must anchor its map frame to the SAME sidecar the
    # loaded map is stored in, or the two disagree while both publishing "map".
    # Derived here rather than hardcoded so it cannot drift from the map
    # actually being loaded; an explicit launch argument still wins.
    if not LaunchConfiguration("georeference_map_filename").perform(context):
        actions.append(
            SetLaunchConfiguration(
                "georeference_map_filename", georeference_filename
            )
        )
    actions.extend(
        _configure_lidar_map_initialization(
            context, map_directory, map_filename, georeference
        )
    )
    actions.extend(_configure_zone_files(context, map_directory))
    actions.extend(_configure_saved_lane_map(context, map_directory))
    return actions


def _configure_lidar_map_initialization(
    context, map_directory: Path, map_filename: str, georeference
):
    """Start stationary 360-degree scan-to-map alignment for this exact map."""

    layer_name = LaunchConfiguration("map_layer_name").perform(context).strip()
    if not layer_name or "/" in layer_name:
        raise RuntimeError(f"Invalid metric-map layer name: '{layer_name}'")

    configured_map = LaunchConfiguration(
        "lidar_localization_map_file"
    ).perform(context).strip()
    if configured_map:
        lidar_map = Path(configured_map).expanduser()
        if not lidar_map.is_absolute():
            lidar_map = map_directory / lidar_map
    else:
        # A map rebuilt directly on the continuous GNSS/IMU reference
        # trajectory has priority over the PLY exported from a deformed MOLA
        # map. The latter remains the compatibility fallback.
        reference_map = map_directory / "gnss_reference_localmap.ply"
        lidar_map = (
            reference_map
            if reference_map.is_file()
            else map_directory / f"{Path(map_filename).stem}_{layer_name}.ply"
        )

    mode = LaunchConfiguration("lidar_map_initialization").perform(context)
    mode = mode.strip().lower()
    if mode == "auto":
        enabled = lidar_map.is_file()
    elif mode in ("true", "1", "yes", "on"):
        enabled = True
    elif mode in ("false", "0", "no", "off"):
        enabled = False
    else:
        raise RuntimeError(
            "lidar_map_initialization must be 'auto', 'true', or 'false', "
            f"not '{mode}'"
        )

    if not enabled:
        message = "Stationary LiDAR map initialization disabled"
        if mode == "auto":
            message += f": derived PLY not found at '{lidar_map}'"
        return [LogInfo(msg=message)]
    if not lidar_map.is_file():
        raise RuntimeError(
            f"LiDAR localization map does not exist: '{lidar_map}'. Export "
            "the already-mapped MM layer first with: ros2 run "
            "mola_slam_mapping prepare_lidar_localization_map -- "
            f"'{map_directory / map_filename}'"
        )

    config_path = Path(
        LaunchConfiguration("lidar_map_localization_config").perform(context)
    ).expanduser()
    if not config_path.is_file():
        raise RuntimeError(
            f"LiDAR map-localization config does not exist: '{config_path}'"
        )

    return [
        LogInfo(
            msg=(
                f"Starting stationary LiDAR alignment with '{lidar_map}' and "
                f"the {Path(map_filename).name} georeference"
            )
        ),
        Node(
            package="krowd_lidar_odometry",
            executable="lidar_map_localization_node",
            name="mola_lidar_map_localization",
            output="screen",
            parameters=[
                str(config_path),
                {
                    "map_file": str(lidar_map),
                    "points_topic": LaunchConfiguration("lidar_topic"),
                    "gps_topic": LaunchConfiguration("gnss_topic"),
                    "odometry_topic": LaunchConfiguration(
                        "lidar_map_odometry_topic"
                    ),
                    "pose_topic": LaunchConfiguration("lidar_map_pose_topic"),
                    "quality_topic": LaunchConfiguration(
                        "lidar_map_quality_topic"
                    ),
                    "verified_topic": LaunchConfiguration(
                        "lidar_map_verified_topic"
                    ),
                    "status_topic": LaunchConfiguration(
                        "lidar_map_status_topic"
                    ),
                    "geodetic_origin_latitude": georeference.latitude,
                    "geodetic_origin_longitude": georeference.longitude,
                    "map_to_enu_x": georeference.x,
                    "map_to_enu_y": georeference.y,
                    "map_to_enu_yaw_deg": georeference.yaw_deg,
                },
            ],
        ),
    ]


def _configure_zone_files(context, map_directory: Path):
    """Prefer the zone files saved with the map being localized against.

    One file per zone kind - traffic_detect_zones.yaml, slow_down_zones.yaml,
    avoidance_zones.yaml, school_zones.yaml - each handled the same way. Zone numbers are
    waypoint indices of one specific route, so the map folder is where they
    belong; the mapping run leaves empty templates there for the operator to
    fill in. The package-wide files stay as the fallback for maps recorded
    before that (and for a deliberate shared list), and an explicit
    `<kind>_yaml` launch argument always wins.
    """

    package_config = (
        Path(get_package_share_directory("mola_slam_mapping")) / "config"
    )
    actions = []
    for kind in ZONE_KINDS:
        argument = f"{kind.parameter}_yaml"
        if LaunchConfiguration(argument).perform(context):
            continue
        saved = map_directory / kind.filename
        package_default = package_config / kind.filename
        actions.append(
            SetLaunchConfiguration(
                argument,
                str(saved if saved.is_file() else package_default),
            )
        )
    return actions


def _configure_saved_lane_map(context, map_directory: Path):
    """Publish the lanes and stop lines saved with this map, if it has any.

    Same lookup as the zone files: <map_dir>/lane_map.csv when the map folder
    has one, an explicit lane_map_csv argument always wins. Unlike the zones
    there is no package fallback - a map without lanes just draws none.
    """

    explicit = LaunchConfiguration("lane_map_csv").perform(context)
    if explicit:
        csv_path = Path(explicit).expanduser()
        if not csv_path.is_file():
            raise RuntimeError(f"lane_map_csv does not exist: '{csv_path}'")
    else:
        csv_path = map_directory / "lane_map.csv"
        if not csv_path.is_file():
            return [LogInfo(msg=f"No saved lane map ({csv_path}); none published")]
    return [
        Node(
            package="mola_slam_mapping",
            executable="saved_lane_map_node",
            name="saved_lane_map_node",
            output="screen",
            parameters=[{
                "lane_map_csv": str(csv_path),
                "topic": LaunchConfiguration("saved_lane_map_topic"),
            }],
        )
    ]


# Topics a localization bag replay needs. /tf and /tf_static are left out on
# purpose: this launch owns map/odom/base_link and the sensor static TFs, and a
# recorded copy would publish the same edges a second time.
BAG_REPLAY_TOPICS = [
    "/ouster/points",
    "/ouster/metadata",
    "/imu/data_ekf",
    "/imu/data",
    "/ublox_gps_node/fix",
    "/ublox_gps_node/fix_velocity",
    "/wheel_sensor",
    "/encoder",
    "/zed/zed_node/rgb/color/rect/image/compressed",
    "/zed/zed_node/rgb/color/rect/camera_info",
    # The recorded sensor-stack ESKF: its verified heading seeds the start.
    "/ekf/global_odometry",
    "/diagnostics",
]
ZED_IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image"


def _configure_bag_replay(context):
    """bag:=<path> replays it into this launch, camera overlay included.

    replay:=true prepares everything below WITHOUT playing a bag, for a bag
    played by hand in another terminal (see BAG_REPLAY_TOPICS for its --topics).

    Live, the ZED driver supplies two things the RViz camera overlay needs and
    a bag does not: the raw image (only .../image/compressed is recorded) and
    the zed_camera_link -> zed_left_camera_frame_optical TF, which the
    driver's own robot_state_publisher publishes from the ZED2i URDF. Both are
    started here only for a replay; live they would duplicate the driver's.
    The bag starts after bag_delay_sec so MOLA has loaded the map first.
    """

    bag = LaunchConfiguration("bag").perform(context).strip()
    replay = LaunchConfiguration("replay").perform(context).strip().lower() in ("true", "1")
    if not bag and not replay:
        return []
    bag_path = Path(bag).expanduser() if bag else None
    if bag_path is not None and not bag_path.exists():
        raise RuntimeError(f"bag:={bag} does not exist")

    import xacro

    zed_urdf = (
        Path(get_package_share_directory("zed_wrapper")) / "urdf" / "zed_descr.urdf.xacro"
    )
    robot_description = xacro.process_file(
        str(zed_urdf), mappings={"camera_name": "zed", "camera_model": "zed2i"}
    ).toxml()

    rate = LaunchConfiguration("bag_rate").perform(context)
    delay = float(LaunchConfiguration("bag_delay_sec").perform(context))
    prepared = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="zed_state_publisher",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="image_transport",
            executable="republish",
            name="zed_image_republish",
            arguments=["compressed", "raw"],
            remappings=[
                ("in/compressed", ZED_IMAGE_TOPIC + "/compressed"),
                ("out", ZED_IMAGE_TOPIC),
            ],
        ),
    ]
    if bag_path is None:
        return [
            LogInfo(
                msg="replay:=true - sensor TFs, ZED camera TF and image decoding are up; "
                "play the bag yourself WITHOUT /tf and /tf_static:\n  ros2 bag play <bag> "
                "--topics " + " ".join(BAG_REPLAY_TOPICS)
            ),
            *prepared,
        ]
    return [
        LogInfo(msg=f"bag replay: {bag_path} starts in {delay:.0f} s at rate {rate}"),
        *prepared,
        TimerAction(
            period=delay,
            actions=[
                ExecuteProcess(
                    cmd=["ros2", "bag", "play", str(bag_path), "--rate", rate,
                         "--topics", *BAG_REPLAY_TOPICS],
                    output="screen",
                )
            ],
        ),
    ]


def generate_launch_description():
    package_share = Path(get_package_share_directory("mola_slam_mapping"))
    # Source-tree YAML (see _source_config): edit config/*.yaml, relaunch.
    mapping_system_yaml = _source_config("mola_slam_mapping", "mapping_system.yaml")
    gnss_imu_pose_yaml = str(_source_config("mola_slam_mapping", "gnss_imu_pose.yaml"))
    lane_map_overlay_yaml = str(
        _source_config("mola_slam_mapping", "lane_map_overlay.yaml")
    )
    rviz_config = package_share / "rviz" / "mapping.rviz"
    mola_cli_executable, mola_runtime_environment = local_mola_runtime()

    default_lidar_pipeline = str(
        Path(get_package_share_directory("mola_lidar_odometry"))
        / "pipelines"
        / "lidar3d-dlio-like.yaml"
    )
    default_lidar_map_localization_config = str(
        _source_config("mola_slam_mapping", "lidar_map_localization_defaults.yaml")
    )

    return LaunchDescription([
        LogInfo(
            msg=f"localization parameters: {mapping_system_yaml}, "
            f"{gnss_imu_pose_yaml}, {default_lidar_map_localization_config}"
        ),
        # `map_dir` must be a run directory previously produced by mapping.launch.py
        # (contains map.simplemap, global_map.mm, waypoints.csv, corridor.csv,
        # georeference.yaml),
        # e.g. ~/krowd_maps/20260823_143000. No default: localizing without a map to
        # localize against is a user error, not something to silently paper over.
        DeclareLaunchArgument("map_dir"),
        DeclareLaunchArgument(
            "map_filename", default_value="rolling_local_map.mm"
        ),
        # Mapping and localization pipelines both expose the map as "localmap".
        DeclareLaunchArgument("map_layer_name", default_value="localmap"),
        # Empty means "the sidecar belonging to map_filename" - see
        # _georeference_filename(). Set it only to pair a map with a sidecar
        # that is not its own.
        DeclareLaunchArgument("georeference_map_filename", default_value=""),
        # Which pose owns the TF edge ending at base_link. Unlike mapping, this
        # node publishes no map->odom here (LiDAR Odometry already reports in
        # `map`), so without this the TF tree is two disconnected islands -
        # map/utm/enu and base_link/sensors - and only an RViz view whose Fixed
        # Frame is `map` can resolve anything. "auto" (default) publishes the
        # vehicle pose (vehicle_pose_topic, the one drawn as the vehicle) as
        # map->base_link, but only if no other node publishes base_link within
        # 2 s and only until one does: a bag replay (recorded /tf excluded) gets
        # a base_link-fixed view centred on the vehicle marker, while on the
        # live vehicle the ESKF keeps the edge and TF never sees two parents.
        # "lidar"/"state" force that pose; "none" never publishes the edge.
        DeclareLaunchArgument("base_link_tf_source", default_value="auto"),
        # Waypoint-index ranges where traffic-light detection is required (red
        # corridor in RViz, /traffic_detect_space true inside). Empty means
        # "<map_dir>/traffic_detect_zones.yaml if the mapping run left one,
        # otherwise the package default" - see _configure_zone_files().
        # Set it only to pair a map with a zone list that is not its own.
        DeclareLaunchArgument("traffic_detect_zones_yaml", default_value=""),
        # Same for the slow-down zones (orange, /slow_down_space) ...
        DeclareLaunchArgument("slow_down_zones_yaml", default_value=""),
        # ... and the obstacle-avoidance zones (sky blue, /avoidance_space).
        DeclareLaunchArgument("avoidance_zones_yaml", default_value=""),
        # Child-protection (school) zones: <map_dir>/school_zones.yaml, drawn yellow,
        # /school_zone_space true inside.
        DeclareLaunchArgument("school_zones_yaml", default_value=""),
        # Lanes + stop lines saved with the map (map frame, published latched
        # on saved_lane_map_topic). Empty means "<map_dir>/lane_map.csv if the
        # map folder has one, otherwise none" - see _configure_saved_lane_map().
        DeclareLaunchArgument("lane_map_csv", default_value=""),
        DeclareLaunchArgument("saved_lane_map_topic", default_value="/lane_map/saved"),

        # Same sensor topics / wheel_odom_source knobs as mapping.launch.py - see
        # comments there for what each means. Kept in sync manually since MOLA's own
        # pipeline YAML (not this launch file) is what's actually shared between the
        # two modes.
        DeclareLaunchArgument("lidar_topic", default_value="/ouster/points"),
        # "auto" starts the stationary 360-degree yaw initializer when the PLY
        # exported from the selected MM layer exists beside the map. "true"
        # makes that artifact mandatory; "false" keeps the former drive-based
        # GNSS/course initialization only.
        DeclareLaunchArgument("lidar_map_initialization", default_value="auto"),
        DeclareLaunchArgument("lidar_localization_map_file", default_value=""),
        DeclareLaunchArgument(
            "lidar_map_localization_config",
            default_value=default_lidar_map_localization_config,
        ),
        DeclareLaunchArgument(
            "lidar_map_odometry_topic", default_value="/ekf/odometry"
        ),
        DeclareLaunchArgument(
            "lidar_map_pose_topic",
            default_value="/mola/map_localization/odometry",
        ),
        DeclareLaunchArgument(
            "lidar_map_quality_topic",
            default_value="/mola/map_localization/quality",
        ),
        DeclareLaunchArgument(
            "lidar_map_verified_topic",
            default_value="/mola/map_localization/heading_verified",
        ),
        DeclareLaunchArgument(
            "lidar_map_status_topic",
            default_value="/mola/map_localization/status",
        ),
        # ESKF global odometry in the selected map frame (published only while
        # the live ESKF provides /ekf/georeference and /ekf/global_odometry).
        DeclareLaunchArgument(
            "reference_pose_topic", default_value="/localization/reference_pose"
        ),
        # Latched topic naming the run being localized against, so path
        # tracking can follow the same map without being told where it is.
        DeclareLaunchArgument(
            "map_dir_topic", default_value="/localization/map_dir"
        ),
        # The vehicle pose: drawn as the blue vehicle marker, fed to the route
        # monitor, and used as map->base_link by base_link_tf_source:=auto.
        # gnss_imu_pose_node's RTK pose in the selected map frame by default -
        # the same pose the path trackers steer on, available live and in bag
        # replay alike (after ~5 m of straight driving commits its heading).
        # Pass vehicle_pose_topic:=/localization/reference_pose for the ESKF pose.
        DeclareLaunchArgument("vehicle_pose_topic", default_value="/gnss_imu/pose"),
        DeclareLaunchArgument(
            "mola_seed_pose_topic", default_value="/localization/mola_seed_pose"
        ),
        DeclareLaunchArgument(
            "ekf_global_pose_topic", default_value="/ekf/global_odometry"
        ),
        DeclareLaunchArgument(
            "ekf_origin_topic", default_value="/ekf/georeference"
        ),
        # MOLA and the standalone krowd LiDAR odometry node otherwise both
        # publish /lidar_odometry/pose with different parent frames.
        DeclareLaunchArgument(
            "mola_lidar_pose_topic", default_value="/mola/lidar_odometry/pose"
        ),
        # See mapping.launch.py's comment - the EKF-timing-compensated IMU variant.
        DeclareLaunchArgument("imu_topic", default_value="/imu/data_ekf"),
        # See mapping.launch.py's comment - confirmed real-vehicle topic name.
        DeclareLaunchArgument("gnss_topic", default_value="/ublox_gps_node/fix"),
        DeclareLaunchArgument("gnss_velocity_topic", default_value="/ublox_gps_node/fix_velocity"),
        DeclareLaunchArgument(
            "fusion_imu_topic", default_value="/mola/imu/data_ekf_10hz"
        ),
        DeclareLaunchArgument("fusion_imu_rate_hz", default_value="10.0"),
        DeclareLaunchArgument(
            "gnss_fusion_topic", default_value="/mola/gnss/fix_sanitized"
        ),
        DeclareLaunchArgument("gnss_min_xy_sigma_m", default_value="0.05"),
        DeclareLaunchArgument("gnss_min_z_sigma_m", default_value="1.0"),
        # "can" (default) / "bag" / "none" - see mapping.launch.py's comment.
        DeclareLaunchArgument("wheel_odom_source", default_value="can"),
        DeclareLaunchArgument("wheel_speed_topic", default_value="/wheel_sensor"),
        DeclareLaunchArgument("steering_angle_topic", default_value="/encoder"),
        DeclareLaunchArgument("wheelbase_m", default_value="1.340"),
        DeclareLaunchArgument("left_wheel_topic", default_value="/wheel_left"),
        DeclareLaunchArgument("right_wheel_topic", default_value="/wheel_right"),
        DeclareLaunchArgument("track_width_m", default_value="1.080"),
        DeclareLaunchArgument("wheel_odom_topic", default_value="/wheel_odom"),
        DeclareLaunchArgument("lidar_pipeline", default_value=default_lidar_pipeline),
        DeclareLaunchArgument(
            "deskew_method", default_value="MotionCompensationMethod::Linear"
        ),
        DeclareLaunchArgument("publish_in_sim_time", default_value="true"),
        DeclareLaunchArgument("lo_pose_step_base_m", default_value="0.20"),
        DeclareLaunchArgument("lo_max_linear_speed_mps", default_value="12.0"),
        DeclareLaunchArgument("lo_rotation_step_base_deg", default_value="3.0"),
        DeclareLaunchArgument("lo_max_angular_speed_deg_s", default_value="60.0"),
        DeclareLaunchArgument(
            "initial_alignment_min_distance_m", default_value="5.0"
        ),
        DeclareLaunchArgument(
            "initial_alignment_min_straightness_ratio", default_value="0.95"
        ),
        # RViz only, matching mapping.launch.py: MOLA's built-in ImGui window
        # ("viz") draws the local SLAM frame with no route, corridor or reference
        # trajectory, so it duplicates the GPU cost without adding information.
        # Pass viz:=true to debug MOLA internals.
        DeclareLaunchArgument("viz", default_value="false"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_map_voxel_size_m", default_value="0.25"),
        DeclareLaunchArgument("corridor_half_width_m", default_value="3.0"),
        # Pure pursuit's `ld`: the follow target published on /lookahead_point
        # sits this far ahead ALONG the route (arc length, so it stays on the
        # route through a corner) and is drawn red in RViz.
        DeclareLaunchArgument("lookahead_distance_m", default_value="5.0"),
        # LidarOdometry starts at this fixed pose - the saved map origin - and
        # FromStateEstimator would be circular here (the estimator has no GNSS,
        # see state_estimator). It is only a placeholder for the first metres:
        # once the RTK heading commits (~5 m of straight driving),
        # rtk_initial_pose_node below re-seeds MOLA's relocalization from the
        # RTK pose, so a run may start anywhere on the map. Pass
        # seed_initial_pose_from_rtk:=false to keep the fixed pose only.
        DeclareLaunchArgument(
            "initial_localization_method",
            default_value="InitLocalization::FixedPose",
        ),
        DeclareLaunchArgument("seed_initial_pose_from_rtk", default_value="true"),
        # A single GNSS antenna has no heading at rest.  Use the nearest saved
        # route waypoint's intended direction only as a bounded /initialpose
        # prior; scan-to-map GICP must still verify it before ESKF accepts yaw.
        DeclareLaunchArgument("seed_initial_pose_from_route", default_value="true"),
        DeclareLaunchArgument("route_seed_max_distance_m", default_value="6.0"),
        # Route tangent and the actual parked/turning vehicle can differ by
        # ~20 deg. sigma=10 with the localizer's 3-sigma sweep searches +/-30
        # deg while still excluding the dangerous 180-deg opposite mode.
        DeclareLaunchArgument("route_seed_yaw_sigma_deg", default_value="10.0"),
        # The reverse direction too: once the stationary LiDAR alignment above
        # is verified, gnss_imu_pose_node takes its yaw as the initial heading
        # and publishes the RTK pose at a standstill, so path tracking can
        # start without the ~5 m of manual driving the RTK course needs. The
        # course still takes over once moving. false keeps the RTK-only start.
        # Production policy: absolute yaw is confirmed only by RTK course.
        # LiDAR localization remains available for position and independent
        # consistency checking, but cannot seed GNSS/IMU or ESKF yaw.
        DeclareLaunchArgument("seed_heading_from_lidar", default_value="false"),
        # Checked BEFORE LiDAR: a heading the sensor-stack ESKF has already
        # verified from the RTK course (vehicle driven to the start line after
        # all-sensor start) is used at once, at a standstill. Order: ESKF ->
        # LiDAR -> NEEDS_MOTION on /gnss_imu/heading_acquisition, on which path
        # tracking creeps straight for the RTK heading.
        DeclareLaunchArgument("seed_heading_from_eskf", default_value="true"),
        DeclareLaunchArgument("initial_x", default_value="0.0"),
        DeclareLaunchArgument("initial_y", default_value="0.0"),
        DeclareLaunchArgument("initial_z", default_value="0.0"),
        DeclareLaunchArgument("initial_yaw", default_value="0.0"),
        DeclareLaunchArgument("initial_pitch", default_value="0.0"),
        DeclareLaunchArgument("initial_roll", default_value="0.0"),
        # The sensor static TFs (base_link -> camera/LiDAR/IMU/GPS). Off by
        # default (2026-09-20): on the vehicle the sensor stack already owns
        # these edges, and a second publisher of base_link -> zed_camera_link
        # with a different pitch (0 here vs 2.79 deg measured) makes TF flip
        # between the two - camera overlay, lane distances and the LiDAR pose
        # all tilt with it. A bag replay (bag:=...) turns them on by itself;
        # pass true only for a replay started some other way.
        DeclareLaunchArgument("publish_sensor_static_tf", default_value="false"),
        # bag:=<path> replays a recorded drive into this launch (see
        # _configure_bag_replay): sensors, GNSS, IMU and the ZED image, plus the
        # camera TF and image decoding the RViz camera overlay needs.
        DeclareLaunchArgument("bag", default_value=""),
        # replay:=true: the same preparation as bag:= (sensor TFs, ZED camera
        # TF, image decoding) but no bag is played - play it yourself.
        DeclareLaunchArgument("replay", default_value="false"),
        DeclareLaunchArgument("bag_rate", default_value="1.0"),
        DeclareLaunchArgument("bag_delay_sec", default_value="12.0"),
        # Which NavStateFilter MOLA loads; see mapping.launch.py for why the
        # smoother is not the default. In this mode LiDAR odometry publishes
        # directly in map, so there is no odom->map drift to fuse and
        # pose_fusion_node deliberately stays silent - the purple overlay is a
        # mapping-run diagnostic.
        DeclareLaunchArgument(
            "state_estimator",
            default_value="mola::state_estimation_simple::StateEstimationSimple",
            description=(
                "MOLA NavStateFilter class. Either "
                "'mola::state_estimation_simple::StateEstimationSimple' "
                "(default) or "
                "'mola::state_estimation_smoother::StateEstimationSmoother' "
                "(fuses GNSS into SLAM; degrades mapping - see "
                "config/mapping_system.yaml)."
            ),
        ),
        SetEnvironmentVariable(
            "MOLA_STATE_ESTIMATOR", LaunchConfiguration("state_estimator")
        ),
        OpaqueFunction(function=_configure_saved_map_georeference),

        SetEnvironmentVariable("MOLA_LIDAR_TOPIC", LaunchConfiguration("lidar_topic")),
        SetEnvironmentVariable("MOLA_LO_IMU_TOPIC", LaunchConfiguration("imu_topic")),
        SetEnvironmentVariable(
            "MOLA_FUSION_IMU_TOPIC", LaunchConfiguration("fusion_imu_topic")
        ),
        SetEnvironmentVariable(
            "MOLA_GNSS_FUSION_TOPIC", LaunchConfiguration("gnss_fusion_topic")
        ),
        SetEnvironmentVariable("WHEEL_ODOM_TOPIC", LaunchConfiguration("wheel_odom_topic")),
        SetEnvironmentVariable(
            "MOLA_ODOMETRY_PIPELINE_YAML", LaunchConfiguration("lidar_pipeline")
        ),
        SetEnvironmentVariable(
            "MOLA_ROS2_PUBLISH_IN_SIM_TIME", LaunchConfiguration("publish_in_sim_time")
        ),
        SetEnvironmentVariable(
            "MOLA_MAX_ACCEPTED_POSE_STEP_BASE",
            LaunchConfiguration("lo_pose_step_base_m"),
        ),
        SetEnvironmentVariable(
            "MOLA_MAX_ACCEPTED_LINEAR_SPEED",
            LaunchConfiguration("lo_max_linear_speed_mps"),
        ),
        SetEnvironmentVariable(
            "MOLA_MAX_ACCEPTED_ROTATION_STEP_BASE_DEG",
            LaunchConfiguration("lo_rotation_step_base_deg"),
        ),
        SetEnvironmentVariable(
            "MOLA_MAX_ACCEPTED_ANGULAR_SPEED_DEG_S",
            LaunchConfiguration("lo_max_angular_speed_deg_s"),
        ),
        SetEnvironmentVariable("MOLA_WITH_GUI", LaunchConfiguration("viz")),
        SetEnvironmentVariable("MOLA_INCREMENTAL_MAP_MAX_SIZE", "0"),
        SetEnvironmentVariable("MOLA_LOCAL_MAP_MAX_SIZE", "0"),
        SetEnvironmentVariable(
            "MOLA_LOCALMAP_LAYER_NAME", LaunchConfiguration("map_layer_name")
        ),
        SetEnvironmentVariable("MOLA_IMU_NAME", "imu_lo"),
        SetEnvironmentVariable("MOLA_IMU_ATTITUDE_USE_YAW", "false"),
        SetEnvironmentVariable("MOLA_IMU_ATTITUDE_AZIMUTH_OFFSET", "-90.0"),
        SetEnvironmentVariable("MOLA_IMU_INITIAL_YAW_SIGMA", "30.0"),
        SetEnvironmentVariable("MOLA_IMU_GRAVITY_ALIGNMENT_SIGMA", "0.0"),
        SetEnvironmentVariable("MOLA_INITIAL_TWIST_SIGMA_LIN", "5.0"),
        SetEnvironmentVariable("MOLA_INITIAL_TWIST_SIGMA_ANG", "1.0"),
        SetEnvironmentVariable(
            "MOLA_DESKEW_METHOD", LaunchConfiguration("deskew_method")
        ),
        SetEnvironmentVariable("MOLA_DESKEW_IGNORE_ACCELEROMETER", "true"),
        SetEnvironmentVariable("MOLA_MAXIMUM_RANGE_FILTER", "75.0"),
        # The frozen map is published once and BridgeROS2 latches it for late RViz.
        SetEnvironmentVariable("MOLA_PUBLISH_LOCAL_MAP_UPDATES_EVERY_N", "0"),

        # Localization mode, as opposed to mapping.launch.py's mapping mode:
        #  - Freeze the local map instead of growing it, loading the one saved by a
        #    previous mapping run instead of building from scratch.
        #  - Don't (re)generate a .simplemap - there is nothing new to map.
        #  - Default to the saved run's origin pose. Arbitrary-start deployments
        #    can override the method and pose launch arguments above.
        SetEnvironmentVariable("MOLA_MAPPING_ENABLED", "false"),
        SetEnvironmentVariable("MOLA_GENERATE_SIMPLEMAP", "false"),
        SetEnvironmentVariable(
            "MOLA_LO_INITIAL_LOCALIZATION_METHOD",
            LaunchConfiguration("initial_localization_method"),
        ),
        SetEnvironmentVariable("MOLA_INITIAL_X", LaunchConfiguration("initial_x")),
        SetEnvironmentVariable("MOLA_INITIAL_Y", LaunchConfiguration("initial_y")),
        SetEnvironmentVariable("MOLA_INITIAL_Z", LaunchConfiguration("initial_z")),
        SetEnvironmentVariable(
            "MOLA_INITIAL_YAW", LaunchConfiguration("initial_yaw")
        ),
        SetEnvironmentVariable(
            "MOLA_INITIAL_PITCH", LaunchConfiguration("initial_pitch")
        ),
        SetEnvironmentVariable(
            "MOLA_INITIAL_ROLL", LaunchConfiguration("initial_roll")
        ),
        _sensor_static_tf_node(
            "localization_tf_base_to_imu",
            "base_link", "imu_link",
            (0.3938, 0.0, 0.876),
            (-0.00927259539161, 0.030217093795, 0.00028033096354, 0.99950030897),
        ),
        _sensor_static_tf_node(
            "localization_tf_base_to_zed",
            "base_link", "zed_camera_link",
            (0.461, 0.0, 0.8576),
            (0.0, 0.0, 0.0, 1.0),
        ),
        _sensor_static_tf_node(
            "localization_tf_zed_to_ouster",
            "zed_camera_link", "os_sensor",
            (-0.0414949508, 0.0504791616, -0.0901744425),
            (0.00486081091, -0.00173969287, -0.0150764324, 0.999873016),
        ),
        _sensor_static_tf_node(
            "localization_tf_ouster_to_lidar",
            "os_sensor", "os_lidar",
            (0.0, 0.0, 0.03618),
            (0.0, 0.0, 1.0, 0.0),
        ),
        _sensor_static_tf_node(
            "localization_tf_base_to_gps",
            "base_link", "gps_link",
            (2.5038, 0.0, 1.021),
            (0.0, 0.0, 0.0, 1.0),
        ),


        Node(
            package="mola_slam_mapping",
            executable="imu_downsample_node",
            name="imu_downsample_node",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("imu_topic"),
                "output_topic": LaunchConfiguration("fusion_imu_topic"),
                "target_rate_hz": ParameterValue(
                    LaunchConfiguration("fusion_imu_rate_hz"), value_type=float
                ),
            }],
        ),

        Node(
            package="mola_slam_mapping",
            executable="gnss_covariance_floor_node",
            name="gnss_covariance_floor_node",
            output="screen",
            parameters=[{
                "input_topic": LaunchConfiguration("gnss_topic"),
                "output_topic": LaunchConfiguration("gnss_fusion_topic"),
                "min_xy_sigma_m": ParameterValue(
                    LaunchConfiguration("gnss_min_xy_sigma_m"), value_type=float
                ),
                "min_z_sigma_m": ParameterValue(
                    LaunchConfiguration("gnss_min_z_sigma_m"), value_type=float
                ),
            }],
        ),
        Node(
            package="mola_slam_mapping",
            executable="gnss_imu_pose_node",
            name="gnss_imu_pose_node",
            output="screen",
            # Tuning in config/gnss_imu_pose.yaml; the dict below (launch
            # arguments and topic wiring) is applied after it and wins.
            parameters=[gnss_imu_pose_yaml, {
                "fix_topic": LaunchConfiguration("gnss_topic"),
                "velocity_topic": LaunchConfiguration("gnss_velocity_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "output_topic": "/gnss_imu/pose",
                # Latched selected-map metadata rebases the already-running
                # ESKF and converts verified LiDAR map yaw into ENU yaw.
                "georeference_topic": "/state_estimator/geo_ref_metadata",
                "map_dir": LaunchConfiguration("map_dir"),
                "georeference_map_filename": LaunchConfiguration(
                    "georeference_map_filename"
                ),
                "initial_alignment_min_distance_m": ParameterValue(
                    LaunchConfiguration("initial_alignment_min_distance_m"),
                    value_type=float,
                ),
                "initial_alignment_min_straightness_ratio": ParameterValue(
                    LaunchConfiguration(
                        "initial_alignment_min_straightness_ratio"
                    ),
                    value_type=float,
                ),
                "seed_heading_from_lidar": ParameterValue(
                    LaunchConfiguration("seed_heading_from_lidar"),
                    value_type=bool,
                ),
                "seed_heading_from_eskf": ParameterValue(
                    LaunchConfiguration("seed_heading_from_eskf"),
                    value_type=bool,
                ),
                "lidar_pose_topic": LaunchConfiguration("lidar_map_pose_topic"),
                "lidar_heading_verified_topic": LaunchConfiguration(
                    "lidar_map_verified_topic"
                ),
            }],
        ),

        # This selector is only for MOLA's one-shot /initialpose seed. Vehicle
        # display and the route monitor use vehicle_pose_topic, not this.
        # Candidate scan matches remain hidden until verified; drive-based
        # GNSS/course remains a fallback seed if LiDAR never verifies.
        Node(
            package="mola_slam_mapping",
            executable="localization_reference_pose_node",
            name="localization_reference_pose_node",
            output="screen",
            parameters=[{
                "primary_pose_topic": LaunchConfiguration(
                    "lidar_map_pose_topic"
                ),
                "primary_verified_topic": LaunchConfiguration(
                    "lidar_map_verified_topic"
                ),
                "fallback_pose_topic": "/gnss_imu/pose",
                "output_pose_topic": LaunchConfiguration(
                    "mola_seed_pose_topic"
                ),
            }],
        ),


        # Starts the same MOLA SLAM stack as mapping.launch.py (config/mapping_system.yaml
        # is shared between the two modes - see its header comment), just with the
        # above env vars switching mola_lidar_odometry into localization-only behavior.
        ExecuteProcess(
            name="mola_cli",
            output="screen",
            cmd=[
                mola_cli_executable,
                str(mapping_system_yaml),
                "--ros-args", "-r", "__node:=mola_cli",
                "-r",
                [
                    "/lidar_odometry/pose:=",
                    LaunchConfiguration("mola_lidar_pose_topic"),
                ],
            ],
            additional_env=mola_runtime_environment,
        ),

        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="mola_slam_mapping",
            executable="map_pointcloud_voxel_filter_node",
            name="map_pointcloud_voxel_filter_node",
            output="screen",
            parameters=[{
                "voxel_size_m": ParameterValue(
                    LaunchConfiguration("rviz_map_voxel_size_m"), value_type=float
                ),
            }],
        ),

        # han_krowd's live lane detections placed on this map with the
        # localized pose (not TF - see lane_map_overlay_node), plus the last
        # 10 s of them. Idle until han_krowd publishes /glane3d/lane_markers.
        Node(
            package="mola_slam_mapping",
            executable="lane_map_overlay_node",
            name="lane_map_overlay_node",
            output="screen",
            parameters=[lane_map_overlay_yaml],
        ),
        # Same complete operator view as mapping mode: loaded accumulated cloud,
        # fused vehicle pose, driven path, saved route/corridor, and on-track state.
        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", str(rviz_config)],
        ),

        # Accumulate the localization-run driven path in map. In localization
        # LO publishes the loaded cloud directly as map, so the helper detects
        # the equal parent frames and deliberately emits no map->map self-TF.
        Node(
            package="mola_slam_mapping",
            executable="map_odom_tf_node",
            name="map_odom_tf_node",
            output="screen",
            parameters=[{
                "state_pose_topic": LaunchConfiguration("vehicle_pose_topic"),
                "lidar_pose_topic": LaunchConfiguration(
                    "lidar_map_pose_topic"
                ),
                "lidar_path_state_proximity_m": 5.0,
                "require_lidar_pose_for_map_odom": False,
                # LiDAR map matching supplies only the verified initial yaw.
                # The driven/control trajectory is the RTK+IMU state path.
                "publish_lidar_path": False,
                "base_link_tf_source": LaunchConfiguration("base_link_tf_source"),
            }],
        ),

        Node(
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration("wheel_odom_source"), "' == 'can'"
                ])
            ),
            package="mola_slam_mapping",
            executable="wheel_odom_bicycle_node",
            name="wheel_odom_bicycle_node",
            output="screen",
            parameters=[{
                "wheel_speed_topic": LaunchConfiguration("wheel_speed_topic"),
                "steering_angle_topic": LaunchConfiguration("steering_angle_topic"),
                "output_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                "wheelbase_m": ParameterValue(
                    LaunchConfiguration("wheelbase_m"), value_type=float
                ),
            }],
        ),

        Node(
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration("wheel_odom_source"), "' == 'bag'"
                ])
            ),
            package="mola_slam_mapping",
            executable="wheel_odom_from_bag_node",
            name="wheel_odom_from_bag_node",
            output="screen",
            parameters=[{
                "left_wheel_topic": LaunchConfiguration("left_wheel_topic"),
                "right_wheel_topic": LaunchConfiguration("right_wheel_topic"),
                "output_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                "gnss_topic": LaunchConfiguration("gnss_topic"),
                "track_width_m": ParameterValue(
                    LaunchConfiguration("track_width_m"), value_type=float
                ),
            }],
        ),

        # The first trustworthy pose is published once on /initialpose, which
        # BridgeROS2 hands to LidarOdometry::relocalize_near_pose_pdf(). It is
        # normally the stationary scan-to-map result; drive-based GNSS/course
        # remains the fallback when that initializer is off.
        Node(
            condition=IfCondition(LaunchConfiguration("seed_initial_pose_from_rtk")),
            package="mola_slam_mapping",
            executable="rtk_initial_pose_node",
            name="rtk_initial_pose_node",
            output="screen",
            parameters=[{
                "rtk_pose_topic": LaunchConfiguration("mola_seed_pose_topic")
            }],
        ),

        # Loads the waypoints.csv saved by the mapping run being localized against,
        # and for each new fused pose reports the signed lateral offset from that
        # saved centerline plus whether the vehicle is still within the saved
        # drivable corridor - "load the map, compare against current position".
        # Also reports the traffic-light detection, slow-down and
        # obstacle-avoidance zones of that route.
        Node(
            package="mola_slam_mapping",
            executable="localization_monitor_node",
            name="localization_monitor_node",
            output="screen",
            parameters=[
                {
                    "input_odom_topic": LaunchConfiguration(
                        "vehicle_pose_topic"
                    ),
                    "waypoints_csv": PathJoinSubstitution(
                        [LaunchConfiguration("map_dir"), "waypoints.csv"]
                    ),
                    "corridor_csv": PathJoinSubstitution(
                        [LaunchConfiguration("map_dir"), "corridor.csv"]
                    ),
                    "corridor_half_width_m": ParameterValue(
                        LaunchConfiguration("corridor_half_width_m"),
                        value_type=float,
                    ),
                    "lookahead_distance_m": ParameterValue(
                        LaunchConfiguration("lookahead_distance_m"),
                        value_type=float,
                    ),
                },
                LaunchConfiguration("traffic_detect_zones_yaml"),
                LaunchConfiguration("slow_down_zones_yaml"),
                LaunchConfiguration("avoidance_zones_yaml"),
                LaunchConfiguration("school_zones_yaml"),
            ],
        ),
        OpaqueFunction(function=_configure_bag_replay),
    ])
