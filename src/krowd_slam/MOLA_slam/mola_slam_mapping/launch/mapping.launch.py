from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from mola_slam_mapping.local_mola_runtime import local_mola_runtime


def generate_launch_description():
    package_share = Path(get_package_share_directory("mola_slam_mapping"))
    mapping_system_yaml = package_share / "config" / "mapping_system.yaml"
    waypoint_saver_yaml = package_share / "config" / "waypoint_saver.yaml"
    rviz_config = package_share / "rviz" / "mapping.rviz"
    mola_cli_executable, mola_runtime_environment = local_mola_runtime()

    # Validated vehicle default: covariance-to-covariance registration with
    # DLIO-like IncrementalPointCloud map maintenance. The 0824_5 regression
    # replay showed that the KeyframePointCloudMap GICP pipeline can lose the
    # vehicle attitude and drive the smoother into fail-safe on this stack.
    default_lidar_pipeline = str(
        Path(get_package_share_directory("mola_lidar_odometry"))
        / "pipelines"
        / "lidar3d-dlio-like.yaml"
    )
    global_map_rebuild_pipeline = str(
        Path(get_package_share_directory("mola_lidar_odometry"))
        / "pipelines"
        / "sm2mm-global-keyframe.yaml"
    )

    # One run_id shared by mola-cli's simplemap output and waypoint_saver_node's output,
    # computed once here (not by either node independently) so both land in the same
    # per-run folder without needing a runtime handshake between the two processes.
    # Not exposed as a launch argument: both this and MOLA_SIMPLEMAP_OUTPUT below need a
    # plain resolved string at description-build time to join into a filesystem path,
    # which a LaunchConfiguration (only resolved at launch time, and overridable from the
    # CLI) can't give us without a lot of extra substitution machinery for little benefit.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_root = Path.home() / "krowd_maps"
    run_dir = map_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # A stable name for "the run that just started". scripts/map_and_retraject.sh
    # post-processes it without having to parse this file's output, and an
    # operator can `cd ~/krowd_maps/latest`. Only ever replaces a symlink, so a
    # real directory that happens to be called "latest" is left alone.
    latest_link = map_root / "latest"
    try:
        if latest_link.is_symlink():
            latest_link.unlink()
        if not latest_link.exists():
            latest_link.symlink_to(run_dir)
    except OSError:
        pass
    simplemap_path = str(run_dir / "map.simplemap")
    rolling_local_map_path = str(run_dir / "rolling_local_map.mm")
    global_map_path = str(run_dir / "global_map.mm")
    global_map_georef_path = str(run_dir / "global_map.georef.yaml")

    # In the legacy wheel-bag test path, the raw IMU is held until GNSS starts.
    # Both LiDAR deskew and the smoother downsampler must consume that same selected
    # stream, otherwise they would start at different epochs.
    selected_imu_topic = PythonExpression([
        "'", LaunchConfiguration("imu_topic"), "_gated' if '",
        LaunchConfiguration("wheel_odom_source"), "' == 'bag' else '",
        LaunchConfiguration("imu_topic"), "'",
    ])

    return LaunchDescription([
        # Sensor topics - these match the current vehicle wiring (Ouster LiDAR, dedicated
        # IMU, u-blox RTK GNSS); override if the wiring changes, e.g.:
        #   ros2 launch mola_slam_mapping mapping.launch.py lidar_topic:=/other/points
        DeclareLaunchArgument("lidar_topic", default_value="/ouster/points"),
        # Xsens publishes several IMU variants (see topic list); "/imu/data_ekf" is the
        # one already time-compensated for EKF-style fusion (14ms filter-delay
        # correction applied to "/imu/data_control") - the appropriate one to feed
        # mola_state_estimation_smoother, and the only IMU topic present in the
        # 2026-08-24 real-vehicle bag (0824_5; that recording has no "/imu/data").
        DeclareLaunchArgument("imu_topic", default_value="/imu/data_ekf"),
        # Real vehicle's u-blox driver node is named "ublox_gps_node", so its NavSatFix
        # is under that namespace - confirmed against both a live `ros2 topic list` and
        # the 2026-08-24 real-vehicle bag (0824_5), NOT "/ublox_gps/fix" (that was the
        # URBAN_C0 bag's own topic name, a different, unrelated recording).
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
        # Which wheel_odom source feeds mola_bridge_ros2:
        #   "can" (default) - production path. ros_can_bridge publishes raw rear wheel
        #     speed + front steering angle (no native nav_msgs/Odometry topic), so
        #     wheel_odom_bicycle_node converts those into wheel_odom_topic via a
        #     bicycle model.
        #   "bag" - TEST ONLY, for replaying old bags (e.g. URBAN_C0) that have no CAN
        #     topics at all. wheel_odom_from_bag_node instead derives wheel_odom_topic
        #     from the bag's independent left/right wheel TwistStamped topics via
        #     differential-drive kinematics (see left_wheel_topic/right_wheel_topic/
        #     track_width_m below). That bag's /vel topic looks like a candidate too
        #     but isn't usable: it's GPS ground speed (frame_id "/gps"), angular.z is
        #     always 0, so it carries no yaw information.
        #   "none" - TEST ONLY, for real-vehicle bags recorded without the CAN wheel
        #     topics (e.g. 0824_5). Neither node below launches, so WHEEL_ODOM_TOPIC
        #     simply has no publisher; mola_state_estimation_smoother fuses LiDAR
        #     odometry + IMU + GNSS only. This is not a hack: wheel_odom is one of
        #     several optional NavStateFilter inputs by design, not a required one -
        #     see mapping_system.yaml's data-flow comment. Do NOT synthesize a fake
        #     wheel_odom from GNSS velocity instead: /ublox_gps_node/fix_velocity is an
        #     ENU ground-track vector, not a body-frame forward speed, and is noisy/
        #     undefined at low speed or standstill - feeding it in would very likely
        #     hurt the estimate rather than help it.
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
        # Reject impossible ICP pose jumps before they can update LiDAR state,
        # enter the smoother, or contaminate either the rolling/global map.
        # At the 0824_5 Ouster rate (~8.8 Hz), these defaults allow about 1.6 m
        # per normal scan (roughly 43 km/h plus timestamp/registration margin).
        DeclareLaunchArgument(
            "lo_pose_step_base_m", default_value="0.20"
        ),
        DeclareLaunchArgument(
            "lo_max_linear_speed_mps", default_value="12.0"
        ),
        DeclareLaunchArgument(
            "lo_rotation_step_base_deg", default_value="3.0"
        ),
        DeclareLaunchArgument(
            "lo_max_angular_speed_deg_s", default_value="60.0"
        ),
        # RViz is the single visualization surface for this package; "viz" (MOLA's
        # own built-in ImGui window) is off by default so the two do not compete
        # for the GPU and there is only one place to look. It shows the local SLAM
        # frame only, with none of the route, corridor or reference trajectories,
        # so it adds nothing RViz does not already draw. Pass viz:=true to debug
        # MOLA internals (per-module panels, ICP inspection).
        DeclareLaunchArgument("viz", default_value="false"),
        # RViz2 runs with Fixed Frame="map" for stable float rendering. The route
        # and corridor are still stored in UTM and transformed into map for
        # display, while georeference.yaml preserves the map-to-UTM relationship.
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "base_link_tf_source",
            default_value="none",
            description=(
                "Which pose map_odom_tf_node publishes as the TF edge ending at "
                "base_link: 'none', 'lidar' (LiDAR Odometry, odom->base_link "
                "here), 'state' (the RTK pose) or 'auto' (the state pose, only "
                "while no other node publishes base_link). Required for an "
                "RViz view "
                "whose Fixed Frame is base_link. Leave 'none' when an "
                "ESKF/localization node already owns that edge - base_link may "
                "have exactly one parent."
            ),
        ),
        DeclareLaunchArgument("corridor_half_width_m", default_value="3.0"),
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
        # A full PointCloud2 copy grows with the accumulated map. This counter is
        # evaluated on scans while a map update is pending, not on keyframes.
        # Forty gives a useful preview without continuously copying a long-run map.
        DeclareLaunchArgument("map_publish_every_n", default_value="40"),
        # Half-size of the bounded live IncrementalPointCloud used by the
        # validated DLIO-like default. THIS ALSO BOUNDS THE SAVED LOCALIZATION
        # MAP: points farther than this from the vehicle are evicted continuously,
        # so rolling_local_map.mm written at shutdown holds only this ball around
        # the final pose - and it is the only map file the DLIO-like localization
        # pipeline can load (global_map.mm is a KeyframePointCloudMap, which the
        # Cov2Cov matcher rejects with a fatal exception). At the former 75 m,
        # localization on a 218 m loop with a 73 x 75 m footprint lost ICP exactly
        # 70 m from the origin and only recovered on the way back in. Keep this
        # larger than the farthest point of the course from where mapping ends.
        DeclareLaunchArgument(
            "incremental_map_half_size_m", default_value="200.0"
        ),
        # Secondary live-map/keyframe retention radius; same shutdown caveat, so
        # it must not undercut incremental_map_half_size_m. map.simplemap and the
        # shutdown-rebuilt global map remain full regardless.
        DeclareLaunchArgument(
            "keyframe_history_radius_m", default_value="250.0"
        ),
        DeclareLaunchArgument("map_removal_check_every_n", default_value="20"),
        # Revisited locations older than this window may create a fresh
        # SimpleMap keyframe, preserving both endpoints needed by loop closure.
        DeclareLaunchArgument("simplemap_revisit_window_s", default_value="30.0"),
        # Voxel de-duplication applied only to ROS/RViz map PointCloud2 output.
        # It does not alter the ICP map or the complete simplemap history.
        DeclareLaunchArgument("rviz_map_voxel_size_m", default_value="0.25"),
        # Debugging aid only: wraps mola-cli in gdb (batch mode - runs, and on a
        # crash/signal prints a full backtrace of every thread, then exits) so a
        # SIGSEGV shows where it actually happened instead of just "exit code -11".
        DeclareLaunchArgument("gdb", default_value="false"),
        # Which NavStateFilter MOLA loads. This module also supplies
        # LidarOdometry's motion-model prior, which is why the smoother is not the
        # default despite being the only one that fuses GNSS: on the 0824_5 bag it
        # measurably degrades the saved route. The purple RViz overlay does not
        # depend on this choice - pose_fusion_node produces it either way. The
        # measured comparison is in config/mapping_system.yaml.
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
        # How much the fused overlay trusts LiDAR dead reckoning between RTK
        # samples. pose_fusion_node's docstring carries the measured sweep that
        # picked 0.3; raise it towards 0.6 for tighter RTK agreement, lower it to
        # average away more reference noise.
        DeclareLaunchArgument(
            "correction_filter_weight", default_value="0.3"
        ),

        SetEnvironmentVariable(
            "MOLA_STATE_ESTIMATOR", LaunchConfiguration("state_estimator")
        ),
        SetEnvironmentVariable("MOLA_LIDAR_TOPIC", LaunchConfiguration("lidar_topic")),
        SetEnvironmentVariable("MOLA_LO_IMU_TOPIC", selected_imu_topic),
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
        SetEnvironmentVariable("MOLA_ESTIMATE_GEO_REFERENCE", "true"),
        SetEnvironmentVariable("MOLA_LINK_FIRST_POSE_SIGMA", "1e-6"),
        SetEnvironmentVariable("MOLA_FIXED_GEO_REF_ENABLED", "false"),
        SetEnvironmentVariable("MOLA_LO_PUBLISH_REF_FRAME", "odom"),
        # Per-point measurement noise assumed by the ICP (Censi) covariance
        # estimate. Upstream's 0.01 m makes LiDAR odometry report ~0.023 m per
        # frame, but measured against RTK ground truth it drifts roughly ten times
        # faster than that - so the smoother, fed an over-confident constraint ~9
        # times a second, out-voted the 0.02 m RTK fixes. 0.10 measured best here
        # (lowest position RMS, and it removed the sustained-ICP-failure events);
        # 0.30 was worse again. Override with MOLA_ICP_COV_DEFAULT_POINT_SIGMA.
        SetEnvironmentVariable(
            "MOLA_ICP_COV_DEFAULT_POINT_SIGMA",
            EnvironmentVariable("MOLA_ICP_COV_DEFAULT_POINT_SIGMA", default_value="0.10"),
        ),
        # /imu/data_ekf roll/pitch agrees with the mounted-IMU extrinsic, but its
        # absolute yaw differs by 11-28 degrees over this bag. Keep tilt only.
        SetEnvironmentVariable("MOLA_IMU_ATTITUDE_USE_YAW", "false"),
        # /imu/data_ekf is already ENU-like, so cancel the smoother's built-in
        # north-to-ENU +90 degree conversion. One weak initial yaw observation
        # removes the global heading gauge; all later drifting yaw is ignored.
        SetEnvironmentVariable("MOLA_IMU_ATTITUDE_AZIMUTH_OFFSET", "-90.0"),
        SetEnvironmentVariable("MOLA_IMU_INITIAL_YAW_SIGMA", "30.0"),
        # Quaternion tilt already supplies a gravity direction. A second 0.4-degree
        # factor from dynamic accelerometer samples over-constrains vehicle tilt/Z.
        SetEnvironmentVariable("MOLA_IMU_GRAVITY_ALIGNMENT_SIGMA", "0.0"),
        SetEnvironmentVariable("MOLA_INITIAL_TWIST_SIGMA_LIN", "5.0"),
        SetEnvironmentVariable("MOLA_INITIAL_TWIST_SIGMA_ANG", "1.0"),
        # Linear is the official pipeline default and avoids applying the
        # bag IMU yaw drift directly to each Ouster scan.
        SetEnvironmentVariable(
            "MOLA_DESKEW_METHOD", LaunchConfiguration("deskew_method")
        ),
        SetEnvironmentVariable("MOLA_DESKEW_IGNORE_ACCELEROMETER", "true"),
        # Ouster returns can extend beyond 150 m. Keeping only the useful 75 m
        # neighborhood reduces vertical far-beam clutter and ensures a newly
        # inserted scan cannot immediately refill the rolling map with points
        # beyond its configured half-size.
        SetEnvironmentVariable("MOLA_MAXIMUM_RANGE_FILTER", "75.0"),
        SetEnvironmentVariable("MOLA_IMU_NAME", "imu_lo"),
        # Enable simplemap recording (keyframes + observations + poses, including GNSS)
        # so this mapping run actually saves the point-cloud map, not just the waypoint
        # CSV. Path shares run_dir with waypoint_saver_node's output (see run_id above).
        SetEnvironmentVariable("MOLA_GENERATE_SIMPLEMAP", "true"),
        # Bound the validated DLIO-like IncrementalPointCloud live map.
        SetEnvironmentVariable(
            "MOLA_INCREMENTAL_MAP_MAX_SIZE",
            LaunchConfiguration("incremental_map_half_size_m"),
        ),
        # Bound only the live ICP map by this radius. Full history remains
        # independent in map.simplemap and is rebuilt into global_map.mm.
        SetEnvironmentVariable(
            "MOLA_LOCAL_MAP_MAX_SIZE",
            LaunchConfiguration("keyframe_history_radius_m"),
        ),
        SetEnvironmentVariable(
            "MOLA_MAP_REMOVAL_CHECK_EVERY_N",
            LaunchConfiguration("map_removal_check_every_n"),
        ),
        SetEnvironmentVariable(
            "MOLA_PUBLISH_LOCAL_MAP_UPDATES_EVERY_N",
            LaunchConfiguration("map_publish_every_n"),
        ),
        # Fixed keyframe spacing for the simplemap, instead of the pipeline's
        # default formula "(0.01 + |w|*0.1) * ESTIMATED_OBSERVATION_RADIUS": that
        # scales with the sensor's ~150 m range and with angular velocity, so
        # while turning it stretches to tens of metres and the run recorded only
        # 2 keyframes for a 190 m course - far too sparse to rebuild a map from.
        # The simplemap is the observation-rich full-route source used to rebuild
        # the final global map independently of rolling-map eviction.
        SetEnvironmentVariable(
            "MOLA_SIMPLEMAP_MIN_XYZ",
            EnvironmentVariable("MOLA_SIMPLEMAP_MIN_XYZ", default_value="1.0"),
        ),
        SetEnvironmentVariable(
            "MOLA_SIMPLEMAP_MIN_ROT",
            EnvironmentVariable("MOLA_SIMPLEMAP_MIN_ROT", default_value="10.0"),
        ),
        SetEnvironmentVariable(
            "MOLA_SIMPLEMAP_KF_TIME_WINDOW",
            LaunchConfiguration("simplemap_revisit_window_s"),
        ),
        SetEnvironmentVariable("MOLA_SIMPLEMAP_OUTPUT", simplemap_path),
        # Store deskewed observations in the simplemap so the global rebuild does
        # not reinsert both raw and deskewed copies of every scan.
        SetEnvironmentVariable("MOLA_SAVE_DESKEWED_SCANS", "true"),
        # rolling_local_map.mm is the bounded online DLIO-like map. At clean
        # shutdown, every accepted deskewed SimpleMap observation is rebuilt into
        # a non-evicting KeyframePointCloudMap named global_map.mm.
        SetEnvironmentVariable("MOLA_SAVE_MM", rolling_local_map_path),
        # Both maps use the same MOLA map frame, so the sidecar continuously
        # written by waypoint_saver is valid for the rolling diagnostic map too.
        SetEnvironmentVariable("MOLA_SAVE_MM_GEOREF", global_map_georef_path),
        SetEnvironmentVariable("MOLA_SAVE_GLOBAL_MM", global_map_path),
        SetEnvironmentVariable(
            "MOLA_SAVE_GLOBAL_MM_PIPELINE", global_map_rebuild_pipeline
        ),
        SetEnvironmentVariable(
            "MOLA_SAVE_GLOBAL_MM_GEOREF", global_map_georef_path
        ),

        Node(
            package="mola_slam_mapping",
            executable="imu_downsample_node",
            name="imu_downsample_node",
            output="screen",
            parameters=[{
                "input_topic": selected_imu_topic,
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
            parameters=[{
                "fix_topic": LaunchConfiguration("gnss_topic"),
                "velocity_topic": LaunchConfiguration("gnss_velocity_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "output_topic": "/gnss_imu/pose",
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
            }],
        ),


        # Starts the MOLA SLAM stack (lidar_odom + state_estimation + ros2_bridge)
        # exactly as declared in config/mapping_system.yaml. This is what actually
        # "starts mapping" when this launch file is run.
        ExecuteProcess(
            condition=UnlessCondition(LaunchConfiguration("gdb")),
            name="mola_cli",
            output="screen",
            cmd=[
                mola_cli_executable,
                str(mapping_system_yaml),
                "--ros-args", "-r", "__node:=mola_cli",
            ],
            additional_env=mola_runtime_environment,
            sigterm_timeout="180",
            sigkill_timeout="30",
        ),
        # gdb:=true variant of the same node: batch-mode gdb runs it, and on a
        # crash (or normal exit) prints every thread's backtrace before quitting.
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("gdb")),
            name="mola_cli",
            output="screen",
            cmd=[
                "gdb", "-q", "-batch", "-ex", "run", "-ex",
                "thread apply all bt", "-ex", "quit", "--args",
                mola_cli_executable,
                str(mapping_system_yaml),
                "--ros-args", "-r", "__node:=mola_cli",
            ],
            additional_env=mola_runtime_environment,
            sigterm_timeout="180",
            sigkill_timeout="30",
        ),

        # Keep visualization processing outside mola-cli so filtering a large
        # accumulated cloud cannot delay ICP or state-estimator threads.
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

        # Shows what the map would look like on the RTK-corrected trajectory,
        # while the run is going. Viz-only and subscribe-only: it keeps its own
        # copy of the scans placed by the fused pose and never touches the map
        # ICP registers against. The saved corrected map still comes from
        # scripts/retraject_map.sh after the run.
        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="mola_slam_mapping",
            executable="corrected_map_preview_node",
            name="corrected_map_preview_node",
            output="screen",
            parameters=[{
                "pointcloud_topic": LaunchConfiguration("lidar_topic"),
                "voxel_size_m": ParameterValue(
                    LaunchConfiguration("rviz_map_voxel_size_m"), value_type=float
                ),
            }],
        ),

        # Stable global-map view of the live run: complete accumulated point cloud,
        # fused vehicle pose, driven path, UTM-recorded reference route, and filled
        # drivable corridor. RViz uses map coordinates to avoid UTM float jitter.
        Node(
            condition=IfCondition(LaunchConfiguration("rviz")),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", str(rviz_config)],
        ),

        # Converts ros_can_bridge's WheelSpeedStamped + EncoderAngleStamped (rear wheel
        # speed + front steering angle) into nav_msgs/Odometry via a bicycle model, so
        # mola_bridge_ros2 has a wheel_odom source to fuse. Production path.
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
                # ParameterValue + value_type=float: DeclareLaunchArgument only ever
                # produces strings, and this node's `wheelbase_m` parameter is declared
                # with a float default (1.340) - passing the raw string would raise
                # InvalidParameterTypeException at node startup.
                "wheelbase_m": ParameterValue(
                    LaunchConfiguration("wheelbase_m"), value_type=float
                ),
            }],
        ),

        # TEST ONLY: derives wheel_odom_topic from a recorded bag's independent
        # left/right wheel TwistStamped topics via differential-drive kinematics.
        # Used in place of wheel_odom_bicycle_node when replaying old bags (e.g.
        # URBAN_C0) that have no CAN wheel-speed/steering-encoder topics. Withholds
        # its own output until the first GNSS fix - see its docstring.
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

        # TEST ONLY: republishes imu_topic on "<imu_topic>_gated" (what IMU_TOPIC above
        # actually points mola_bridge_ros2 at in bag mode), withheld until the first
        # GNSS fix - see imu_gate_node.py's docstring for why. Only launched alongside
        # wheel_odom_from_bag_node, for the same reason.
        Node(
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration("wheel_odom_source"), "' == 'bag'"
                ])
            ),
            package="mola_slam_mapping",
            executable="imu_gate_node",
            name="imu_gate_node",
            output="screen",
            parameters=[{
                "input_imu_topic": LaunchConfiguration("imu_topic"),
                "output_imu_topic": PythonExpression([
                    "'", LaunchConfiguration("imu_topic"), "_gated'"
                ]),
                "gnss_topic": LaunchConfiguration("gnss_topic"),
            }],
        ),

        # Visualization support: publishes the map->odom transform that is otherwise
        # missing here, so LidarOdometry's odom-frame cloud can be shown in the
        # stable map view, and republishes the driven trajectory as a map Path.
        # map->odom stays anchored to gnss_imu_pose_node whichever state estimator
        # is selected; the fused pose is picked up separately on its own default
        # topic and drawn as the purple comparison overlay only.
        Node(
            package="mola_slam_mapping",
            executable="map_odom_tf_node",
            name="map_odom_tf_node",
            output="screen",
            parameters=[{
                "state_pose_topic": "/gnss_imu/pose",
                "base_link_tf_source": LaunchConfiguration("base_link_tf_source"),
            }],
        ),

        # Fuses the drift-free-but-noisy RTK/IMU pose with the smooth-but-drifting
        # LiDAR odometry and publishes the result for the purple RViz overlay.
        # Subscribe-only by design: nothing in the SLAM stack consumes its output,
        # so it cannot affect the map or the saved route.
        Node(
            package="mola_slam_mapping",
            executable="pose_fusion_node",
            name="pose_fusion_node",
            output="screen",
            parameters=[{
                "correction_filter_weight": ParameterValue(
                    LaunchConfiguration("correction_filter_weight"),
                    value_type=float,
                ),
            }],
        ),

        # Buffers the complete 1m-spaced LiDAR-odom route, waits for initial
        # straight-motion/global-yaw stability, then reprojects every point together
        # into UTM/map and saves the route, corridor, and georeference metadata.
        Node(
            package="mola_slam_mapping",
            executable="waypoint_saver_node",
            name="waypoint_saver_node",
            output="screen",
            parameters=[
                str(waypoint_saver_yaml),
                {
                    "output_dir": str(map_root),
                    "run_id": run_id,
                    "corridor_half_width_m": ParameterValue(
                        LaunchConfiguration("corridor_half_width_m"), value_type=float
                    ),
                    "initial_alignment_min_distance_m": ParameterValue(
                        LaunchConfiguration(
                            "initial_alignment_min_distance_m"
                        ),
                        value_type=float,
                    ),
                    "initial_alignment_min_straightness_ratio": ParameterValue(
                        LaunchConfiguration(
                            "initial_alignment_min_straightness_ratio"
                        ),
                        value_type=float,
                    ),
                    "initial_alignment_yaw_window_sec": ParameterValue(
                        LaunchConfiguration(
                            "initial_alignment_yaw_window_sec"
                        ),
                        value_type=float,
                    ),
                    "initial_alignment_min_yaw_samples": ParameterValue(
                        LaunchConfiguration(
                            "initial_alignment_min_yaw_samples"
                        ),
                        value_type=int,
                    ),
                    "initial_alignment_max_yaw_spread_deg": ParameterValue(
                        LaunchConfiguration(
                            "initial_alignment_max_yaw_spread_deg"
                        ),
                        value_type=float,
                    ),
                    "global_map_filename": "global_map.mm",
                },
            ],
        ),

        # Keeps the stamped reference/fused/LiDAR pose streams as CSV in the
        # same run folder. The route has no timestamps and the simplemap has
        # only LiDAR poses, so this is what scripts/retraject_map.sh needs to
        # move the point-cloud map onto the RTK trajectory after the run.
        # Subscribe-only, like pose_fusion_node.
        Node(
            package="mola_slam_mapping",
            executable="trajectory_recorder_node",
            name="trajectory_recorder_node",
            output="screen",
            parameters=[{
                "output_dir": str(map_root),
                "run_id": run_id,
            }],
        ),
    ])
