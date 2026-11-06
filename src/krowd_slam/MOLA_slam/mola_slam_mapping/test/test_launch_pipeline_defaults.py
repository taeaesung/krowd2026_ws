from pathlib import Path

import pytest

from mola_slam_mapping.traffic_detect_zones import ZONE_KINDS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_mapping_launches_default_to_validated_dlio_pipeline():
    launch_files = (
        "mapping.launch.py",
        "mapping_bag.launch.py",
        "localization.launch.py",
    )

    for launch_file in launch_files:
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert '/ "lidar3d-dlio-like.yaml"' in text
        assert '/ "lidar3d-gicp.yaml"' not in text


def test_every_launch_visualizes_through_rviz_only():
    """RViz is the single visualization surface; MOLA's ImGui window stays off.

    It draws the local SLAM frame with no route, corridor or reference
    trajectories, so leaving it on costs a second GPU context and splits the
    operator's attention across two windows showing different things.
    """

    for launch_file in (
        "mapping.launch.py",
        "mapping_bag.launch.py",
        "localization.launch.py",
        "bag_test.launch.py",
    ):
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert (
            'DeclareLaunchArgument("viz", default_value="false")' in text
        ), launch_file
        assert (
            'DeclareLaunchArgument("rviz", default_value="true")' in text
        ), launch_file


def test_bag_replay_defaults_to_cpu_safe_half_rate():
    bag_launch = (PACKAGE_ROOT / "launch" / "mapping_bag.launch.py").read_text()
    assert 'DeclareLaunchArgument("bag_rate", default_value="0.5")' in bag_launch


def test_live_localization_does_not_duplicate_sensor_static_calibration():
    localization = (
        PACKAGE_ROOT / "launch" / "localization.launch.py"
    ).read_text()
    assert (
        'DeclareLaunchArgument("publish_sensor_static_tf", default_value="false")'
        in localization
    )
    for node_name in (
        "localization_tf_base_to_imu",
        "localization_tf_base_to_zed",
        "localization_tf_zed_to_ouster",
        "localization_tf_ouster_to_lidar",
        "localization_tf_base_to_gps",
    ):
        assert f'"{node_name}"' in localization


def test_localization_replay_starts_from_saved_map_origin_by_default():
    localization = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    assert '"map_filename", default_value="rolling_local_map.mm"' in localization
    assert 'map_filename == "rolling_local_map.mm"' in localization
    assert 'default_value="InitLocalization::FixedPose"' in localization
    assert 'LaunchConfiguration("initial_localization_method")' in localization
    for name in ("x", "y", "z", "yaw", "pitch", "roll"):
        assert f'DeclareLaunchArgument("initial_{name}", default_value="0.0")' in localization
        assert f'LaunchConfiguration("initial_{name}")' in localization


def test_initial_yaw_alignment_defaults_to_five_metres():
    expected = '"initial_alignment_min_distance_m", default_value="5.0"'
    assert expected in (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    assert expected in (
        PACKAGE_ROOT / "launch" / "mapping_bag.launch.py"
    ).read_text()
    assert "initial_alignment_min_distance_m: 5.0" in (
        PACKAGE_ROOT / "config" / "waypoint_saver.yaml"
    ).read_text()
    assert 'declare_parameter("initial_alignment_min_distance_m", 5.0)' in (
        PACKAGE_ROOT
        / "mola_slam_mapping"
        / "waypoint_saver_node.py"
    ).read_text()


def test_mapping_rviz_enables_red_fused_state_path():
    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    topic_offset = rviz.index("Value: /state_estimation/path")
    display = rviz[max(0, topic_offset - 800) : topic_offset + 80]
    assert "Name: GPS IMU Path" in display
    assert "Class: rviz_default_plugins/Path" in display
    assert "Color: 255; 0; 0" in display
    assert "Enabled: true" in display

    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py"
    ).read_text()
    assert 'declare_parameter("fused_path_topic", "/state_estimation/path")' in node
    assert "self._state_path.append(msg)" in node


def test_mapping_rviz_overlays_purple_fused_localization_path():
    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    topic_offset = rviz.index("Value: /fused_localization/path")
    display = rviz[max(0, topic_offset - 800) : topic_offset + 80]
    assert "Name: Fused Localization Path" in display
    assert "Class: rviz_default_plugins/Path" in display
    assert "Color: 170; 0; 255" in display
    assert "Enabled: true" in display

    # The overlay must stay a second, independent line: reusing the red path's
    # topic or its pose source would hide exactly the disagreement it is for.
    assert "Value: /state_estimation/path" in rviz

    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py"
    ).read_text()
    assert (
        'declare_parameter(\n'
        '            "fused_localization_pose_topic", "/fused_localization/pose"\n'
        '        )'
    ) in node
    assert (
        'declare_parameter(\n'
        '            "fused_localization_path_topic", "/fused_localization/path"\n'
        '        )'
    ) in node
    assert "self._fused_localization_path.append(msg)" in node


def test_fused_pose_never_drives_map_odom_or_the_slam_stack():
    """The fused pose is an output of the stack, never an input to it.

    It may be drawn and it may be recorded as the route, but nothing that feeds
    LidarOdometry (TF, map->odom, the sensor bridge) may depend on it - that is
    the failure mode that made the in-MOLA smoother unusable.
    """

    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py"
    ).read_text()
    fused_callback = node[
        node.index("def _on_fused_localization") : node.index("def _stamp_ns")
    ]
    assert "_broadcast_map_odom" not in fused_callback
    assert "sendTransform" not in fused_callback

    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    localization = (
        PACKAGE_ROOT / "launch" / "localization.launch.py"
    ).read_text()
    assert '"state_pose_topic": "/gnss_imu/pose"' in mapping
    assert (
        '"state_pose_topic": LaunchConfiguration("vehicle_pose_topic")'
        in localization
    )

    fusion_output = "/fused_localization/pose"
    consumers = [
        path.name
        for path in (PACKAGE_ROOT / "mola_slam_mapping").glob("*.py")
        if fusion_output in path.read_text()
    ]
    assert sorted(consumers) == [
        "corrected_map_preview_node.py",  # live viz-only corrected map
        "map_odom_tf_node.py",  # purple overlay
        "pose_fusion_node.py",  # producer
        "trajectory_recorder_node.py",  # stamped CSV for offline map re-trajectory
    ]


def test_mapping_records_stamped_trajectories_for_offline_map_retraject():
    """The map is built on LiDAR poses; moving it onto RTK is a post-process.

    That needs the reference pose with timestamps (the route has none), a
    tool to write optimized keyframe poses back into the simplemap, and a
    rebuild pipeline whose output layer the localization pipeline can load.
    """

    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    assert 'executable="trajectory_recorder_node"' in mapping
    setup = (PACKAGE_ROOT / "setup.py").read_text()
    assert "trajectory_recorder_node = mola_slam_mapping.trajectory_recorder_node:main" in setup
    assert "map_retraject = mola_slam_mapping.map_retraject:main" in setup

    script = (PACKAGE_ROOT / "scripts" / "retraject_map.sh").read_text()
    for step in ("sm-cli export-keyframes", "map_retraject", "sm-set-poses", "sm2mm"):
        assert step in script
    assert "sm2mm-retraject-incremental.yaml" in script


def test_one_command_bag_mapping_finishes_with_a_fail_closed_ready_map():
    wrapper = (PACKAGE_ROOT / "scripts" / "map_and_retraject.sh").read_text()
    finalizer = (PACKAGE_ROOT / "scripts" / "finalize_mapping_run.sh").read_text()

    assert "bag_path:=" in wrapper
    assert "finalize_mapping_run.sh" in wrapper
    for artifact in (
        "global_map_ready.mm",
        "gnss_reference_localmap.ply",
        "map_quality.yaml",
        "READY_FOR_LOCALIZATION",
    ):
        assert artifact in finalizer
    assert "repair_reference_anchor_outliers.py" in finalizer
    assert "validate_ready_map.py" in finalizer

    pipeline = (PACKAGE_ROOT / "config" / "sm2mm-retraject-incremental.yaml").read_text()
    assert "class: mola::IncrementalPointCloud" in pipeline
    assert 'target_layer: "localmap"' in pipeline
    # The stock sm2mm-global-incremental.yaml renames the result to "globalmap",
    # which localization.launch.py's map_layer_name would not find.
    assert "final_filters" not in pipeline
    assert 'output_pointcloud_layer: "globalmap"' not in pipeline

    fusion = (PACKAGE_ROOT / "mola_slam_mapping" / "pose_fusion_node.py").read_text()
    assert 'declare_parameter("lidar_increment_mismatch_samples", 3)' in fusion


def test_localization_anchors_the_gnss_node_to_the_loaded_maps_own_sidecar():
    """Both must mean the same thing by "map", whichever map is loaded.

    global_map.mm is stored in `odom`; a re-trajectoried
    global_map_corrected.mm is stored in `map`. On testweek_0905_3 those differ
    by 36.5 deg and 75 m, so hardcoding gnss_imu_pose_node to global_map.mm
    silently gave LidarOdometry and it two different frames both called "map".
    """

    launch_text = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    assert '"georeference_map_filename": "global_map.mm"' not in launch_text
    assert (
        '"georeference_map_filename": LaunchConfiguration(\n'
        '                    "georeference_map_filename"\n'
        "                ),"
    ) in launch_text
    # One rule, used by both the map loader and the GNSS node.
    assert launch_text.count("def _georeference_filename") == 1
    assert launch_text.count("_georeference_filename(map_filename)") == 1
    assert 'SetLaunchConfiguration(\n                "georeference_map_filename"' in launch_text


def test_corrected_map_preview_is_visualization_only():
    """The live preview must never become an input to the SLAM stack.

    Correcting the pose that feeds ICP is exactly what the smoother A/B
    measured as harmful, so this node subscribes only, publishes on its own
    viz topic, and is launched only alongside RViz.
    """

    preview = (
        PACKAGE_ROOT / "mola_slam_mapping" / "corrected_map_preview_node.py"
    ).read_text()
    assert "create_publisher" in preview
    # One publisher, and it is the viz topic.
    assert preview.count("create_publisher") == 1
    assert 'declare_parameter("output_topic", "/mola/viz/corrected_map_points")' in preview
    assert "sendTransform" not in preview
    assert "TransformBroadcaster" not in preview

    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    preview_node = mapping[
        mapping.index('executable="corrected_map_preview_node"') - 400 :
        mapping.index('executable="corrected_map_preview_node"') + 200
    ]
    assert 'IfCondition(LaunchConfiguration("rviz"))' in preview_node

    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    assert "/mola/viz/corrected_map_points" in rviz
    # Off by default: it would otherwise hide the rolling map it exists to be
    # compared against.
    assert "Name: Corrected Map Preview (fused pose)" in rviz


def test_mapping_marks_the_current_run_and_can_post_process_it():
    """One command must be able to map and then correct the map it just built."""

    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    assert 'latest_link = map_root / "latest"' in mapping
    assert "latest_link.symlink_to(run_dir)" in mapping

    wrapper = (PACKAGE_ROOT / "scripts" / "map_and_retraject.sh").read_text()
    assert "ros2 launch mola_slam_mapping" in wrapper
    assert "retraject_map.sh" in wrapper
    # Must wait for mola-cli's own SIGINT save path: the simplemap does not
    # exist until then, and a bare kill would orphan it.
    assert 'kill -INT -"$LAUNCH_PID"' in wrapper
    assert 'while kill -0 "$LAUNCH_PID"' in wrapper


def test_saved_route_follows_the_rtk_pose_from_the_first_sample():
    """Route and corridor come from gnss_imu_pose_node's RTK pose, in map.

    Routes saved from the fused pose sat up to 1.5 m from the run's own RTK
    track wherever LiDAR odometry lost tracking, and the path trackers steer on
    /gnss_imu/pose. The route must still start at the run's first pose, which
    gnss_imu_pose_node provides by releasing its buffered poses once the
    heading alignment commits.
    """

    yaml_text = (PACKAGE_ROOT / "config" / "waypoint_saver.yaml").read_text()
    assert 'input_odom_topic: "/gnss_imu/pose"' in yaml_text
    assert 'route_frame: "map"' in yaml_text
    assert 'pointcloud_map_frame: "odom"' in yaml_text

    saver = (
        PACKAGE_ROOT / "mola_slam_mapping" / "waypoint_saver_node.py"
    ).read_text()
    assert 'declare_parameter("input_odom_topic", "/gnss_imu/pose")' in saver
    assert 'declare_parameter("route_frame", "map")' in saver
    # The point-cloud sidecar and the route transform must not alias each other
    # now that they describe different frames.
    assert "self._last_route_transform = self._last_pointcloud_transform" not in saver
    assert "self._last_pointcloud_transform = route_transform" not in saver

    fusion = (
        PACKAGE_ROOT / "mola_slam_mapping" / "pose_fusion_node.py"
    ).read_text()
    first_fix = fusion[fusion.index("def _correct") : fusion.index("def _report_correction_health")]
    assert "self._backfill_history()" in first_fix


def test_waypoint_labels_are_a_window_that_follows_the_vehicle():
    """Only the waypoints around the car carry "#id  x, y" text in RViz.

    A 500 m route has ~400 waypoints; labelling all of them is unreadable and
    costs one RViz object each.
    """

    assert "waypoint_label_radius_m: 12.0" in (
        PACKAGE_ROOT / "config" / "waypoint_saver.yaml"
    ).read_text()

    saver = (
        PACKAGE_ROOT / "mola_slam_mapping" / "waypoint_saver_node.py"
    ).read_text()
    assert 'declare_parameter("waypoint_label_radius_m", 12.0)' in saver
    assert "label_center=self._label_center" in saver
    # Centred on the pose, not on the newest waypoint: waypoints are only
    # appended every min_waypoint_spacing_m, so the newest one lags the vehicle.
    assert "self._latest_valid_local_pose, visualization_transform" in saver


def test_a_base_link_fixed_rviz_view_has_a_tf_edge_and_fresh_transforms():
    """Two halves of the same fix, both required.

    (1) Someone must own the TF edge ending at base_link - in localization this
    node publishes no map->odom, so without it map/utm/enu and base_link/sensors
    are disconnected islands. (2) The latched displays must carry stamp 0, or
    they age out of RViz's TF buffer as soon as the fixed frame moves.
    """

    node = (PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py").read_text()
    assert 'declare_parameter("base_link_tf_source", "none")' in node
    assert "broadcast_lidar_pose_tf" not in node
    assert 'self._broadcast_base_link_tf("lidar", msg)' in node
    assert 'self._broadcast_base_link_tf("state", msg)' in node
    # auto rides on the pose drawn as the vehicle, so a base_link-fixed view
    # stays centred on the vehicle marker.
    assert 'self._effective_base_link_tf_source = "state"' in node

    defaults = {
        "mapping.launch.py": "none",      # a live ESKF owns base_link
        "mapping_bag.launch.py": "lidar",  # recorded /tf is excluded
        # auto: map->base_link for bag replay, yields to the live ESKF's edge
        "localization.launch.py": "auto",
    }
    for launch_file, default in defaults.items():
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert f'"base_link_tf_source"' in text
        assert f'default_value="{default}"' in text
        assert "broadcast_lidar_pose_tf" not in text

    monitor = (
        PACKAGE_ROOT / "mola_slam_mapping" / "localization_monitor_node.py"
    ).read_text()
    assert "Time().to_msg()," in monitor

    voxel = (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_pointcloud_voxel_filter_node.py"
    ).read_text()
    assert 'declare_parameter("publish_with_latest_transform_stamp", True)' in voxel
    assert "filtered.header.stamp = Time()" in voxel

    # The camera must ride on the Fixed Frame, not on a hardcoded `map`. With
    # Target Frame pinned to map, switching Fixed Frame to base_link changes the
    # coordinates but leaves the camera locked to the world, so the view looks
    # identical to the map one and the vehicle never centres.
    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    current_view = rviz[rviz.index("  Views:"):]
    assert "Target Frame: <Fixed Frame>" in current_view
    assert "Target Frame: map" not in current_view

    # And the publish-once displays must repeat, because changing Fixed Frame
    # calls reset() on every topic display WITHOUT resubscribing
    # (rviz_common/ros_topic_display.hpp), so a latched map/route is wiped and
    # never redelivered.
    assert "def republish_is_due(" in voxel
    assert 'declare_parameter("route_republish_interval_sec", 2.0)' in monitor
    assert "def _route_republish_is_due(self)" in monitor
    # The cloud repeat stays OFF: RViz rebuilds the entire cloud on every
    # message, so repeating hundreds of thousands of points is the stutter, not
    # the cure. Markers are re-placed per frame for free instead.
    assert 'declare_parameter("republish_interval_sec", 0.0)' in voxel
    assert "marker.frame_locked = True" in (
        PACKAGE_ROOT / "mola_slam_mapping" / "route_visualization.py"
    ).read_text()
    assert "marker.frame_locked = True" in (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py"
    ).read_text()
    assert "m.frame_locked = True" in monitor


def test_localization_publishes_the_route_and_follow_target_as_data():
    """A follower must not have to parse the drawing. The MarkerArray keeps
    coordinates as float32 offsets from a per-marker anchor and carries no
    heading; nav_msgs/Path and the lookahead point are the data interface."""

    monitor = (
        PACKAGE_ROOT / "mola_slam_mapping" / "localization_monitor_node.py"
    ).read_text()
    assert 'declare_parameter("reference_path_topic", "reference_path")' in monitor
    assert "self._reference_path_pub.publish(" in monitor
    assert 'declare_parameter("lookahead_topic", "lookahead_point")' in monitor
    assert 'declare_parameter("lookahead_distance_m", 5.0)' in monitor
    # Both route publications are latched, so a follower started later still
    # receives them.
    assert monitor.count("latched_qos,\n        )") >= 2

    launch = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    assert 'DeclareLaunchArgument("lookahead_distance_m", default_value="5.0")' in launch
    assert '"lookahead_distance_m": ParameterValue(' in launch

    route_io = (PACKAGE_ROOT / "mola_slam_mapping" / "route_io.py").read_text()
    assert "def route_to_path(" in route_io
    # Heading comes from the recorded yaw, not from differencing waypoints.
    assert "math.sin(point.yaw * 0.5)" in route_io

    # RViz can draw the controller's actual input, so a wrong frame/order/heading
    # shows up instead of being hidden behind a marker that looks right anyway.
    # Off by default: it overlays the centerline marker drawing the same route.
    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    assert "Name: Reference Path (controller input)" in rviz
    assert "Value: /reference_path" in rviz
    assert "Pose Style: Arrows" in rviz


def test_localization_seeds_mola_without_making_lidar_the_vehicle_pose():
    """RTK (or ESKF) locates the vehicle; a verified candidate only seeds MOLA.

    Measured 2026-09-09: both replays logged "Initial re-localization done with
    pose: (0,0,0, ...)" - the FixedPose placeholder, right only when the run
    starts where mapping did. MOLA's relocalization was advertised but nothing
    called it once the smoother (its GNSS layer) was switched off.
    """

    launch = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    assert 'DeclareLaunchArgument("seed_initial_pose_from_rtk", default_value="true")' in launch
    assert 'executable="rtk_initial_pose_node"' in launch
    assert 'IfCondition(LaunchConfiguration("seed_initial_pose_from_rtk"))' in launch
    assert '"rtk_pose_topic": LaunchConfiguration("mola_seed_pose_topic")' in launch
    assert 'executable="global_odometry_map_pose_node"' in launch
    assert '"input_topic": LaunchConfiguration("ekf_global_pose_topic")' in launch
    assert '"output_topic": LaunchConfiguration("reference_pose_topic")' in launch
    # The drawn vehicle pose defaults to the RTK pose: the ESKF pose needs the
    # live ESKF's latched /ekf/georeference, which no recorded bag contains.
    assert 'DeclareLaunchArgument("vehicle_pose_topic", default_value="/gnss_imu/pose")' in launch
    assert '"require_lidar_pose_for_map_odom": False' in launch
    # The selector must not replace the continuously published ESKF pose when
    # LiDAR eventually verifies; it feeds only the one-shot MOLA seed.
    seed_output = (
        '"output_pose_topic": LaunchConfiguration(\n'
        '                    "mola_seed_pose_topic"'
    )
    reference_output = (
        '"output_pose_topic": LaunchConfiguration(\n'
        '                    "reference_pose_topic"'
    )
    assert seed_output in launch
    assert reference_output not in launch

    node = (PACKAGE_ROOT / "mola_slam_mapping" / "rtk_initial_pose_node.py").read_text()
    # The seed goes to the topic BridgeROS2 forwards to relocalize_near_pose_pdf().
    assert 'declare_parameter("initial_pose_topic", "/initialpose")' in node
    assert 'declare_parameter("rtk_pose_topic", "/gnss_imu/pose")' in node
    # Seeds from the NEWEST pose of the heading-commit burst, not the stale first.
    assert 'declare_parameter("settle_time_sec", 0.5)' in node
    # sqrt(2) * 1.0 = 1.41 m ICP search, under the pipeline's 2.0 m maximum.
    assert 'declare_parameter("position_sigma_m", 1.0)' in node

    setup = (PACKAGE_ROOT / "setup.py").read_text()
    assert "rtk_initial_pose_node = mola_slam_mapping.rtk_initial_pose_node:main" in setup


@pytest.mark.parametrize(
    "parameter", ["traffic_detect_zones", "slow_down_zones", "avoidance_zones", "school_zones"]
)
def test_route_zones_live_with_the_map_not_with_the_code(parameter):
    """Zone numbers are waypoint indices of ONE route - every zone kind.

    The mapping run leaves a template in the map folder, localization reads it
    from there, and the package-wide file is an empty fallback: baking one
    map's numbers into it would silently act at the wrong place on another.
    """

    zones_yaml = (PACKAGE_ROOT / "config" / f"{parameter}.yaml").read_text()
    assert "localization_monitor_node:" in zones_yaml
    assert f"{parameter}: []" in zones_yaml

    saver = (
        PACKAGE_ROOT / "mola_slam_mapping" / "waypoint_saver_node.py"
    ).read_text()
    assert "self._save_zone_templates()" in saver
    assert "template_may_be_rewritten(path, kind)" in saver

    launch = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    assert f'DeclareLaunchArgument("{parameter}_yaml", default_value="")' in launch
    assert "saved = map_directory / kind.filename" in launch
    assert "saved if saved.is_file() else package_default" in launch
    assert f'LaunchConfiguration("{parameter}_yaml"),' in launch


def test_localization_monitor_publishes_one_bool_per_zone_kind():
    kinds = {kind.parameter: kind.topic for kind in ZONE_KINDS}
    assert kinds == {
        "traffic_detect_zones": "traffic_detect_space",
        "slow_down_zones": "slow_down_space",
        "avoidance_zones": "avoidance_space",
        "school_zones": "school_zone_space",
    }

    monitor = (
        PACKAGE_ROOT / "mola_slam_mapping" / "localization_monitor_node.py"
    ).read_text()
    assert 'self.declare_parameter(f"{kind.topic}_topic", kind.topic)' in monitor
    assert "check_ranges_fit_route(\n                    self._zones[kind]" in monitor
    assert "highlight_index_ranges=self._zone_index_pairs(TRAFFIC_DETECT)" in monitor
    assert "slow_down_index_ranges=self._zone_index_pairs(SLOW_DOWN)" in monitor
    assert "avoidance_index_ranges=self._zone_index_pairs(AVOIDANCE)" in monitor


def test_corridor_half_width_defaults_to_3_m_everywhere():
    """A 6 m drivable band; every place that carries the default must agree."""

    assert "corridor_half_width_m: 3.0" in (
        PACKAGE_ROOT / "config" / "waypoint_saver.yaml"
    ).read_text()
    for launch_file in ("mapping.launch.py", "localization.launch.py"):
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert 'DeclareLaunchArgument("corridor_half_width_m", default_value="3.0")' in text
    for node_file in ("waypoint_saver_node.py", "localization_monitor_node.py"):
        text = (PACKAGE_ROOT / "mola_slam_mapping" / node_file).read_text()
        assert 'declare_parameter("corridor_half_width_m", 3.0)' in text


def test_mapping_keeps_the_saved_localization_map_whole():
    """The live-map eviction radius is also the saved rolling map's extent.

    At 75 m, localization on a 218 m loop (73 x 75 m footprint) lost ICP exactly
    70 m from the origin because the far side had been evicted before the map
    was written. global_map.mm is not an alternative: the DLIO-like matcher
    rejects its KeyframePointCloudMap type with a fatal exception.
    """

    # mapping_bag.launch.py re-declares both arguments and forwards them, so its
    # defaults silently override mapping.launch.py's - both must agree.
    for launch_file in ("mapping.launch.py", "mapping_bag.launch.py"):
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert '"incremental_map_half_size_m", default_value="200.0"' in text, launch_file
        assert '"keyframe_history_radius_m", default_value="250.0"' in text, launch_file


def test_state_estimator_stays_simple_until_the_smoother_is_fixed():
    """The smoother degrades the saved route; keep it opt-in, not default."""

    simple = "mola::state_estimation_simple::StateEstimationSimple"
    # The doubled-"mola_" spelling is unregistered and aborts mola-cli at startup.
    typo = "mola::mola_state_estimation_smoother::"

    system_yaml = (PACKAGE_ROOT / "config" / "mapping_system.yaml").read_text()
    assert f"type: ${{MOLA_STATE_ESTIMATOR|{simple}}}" in system_yaml
    assert typo not in system_yaml.replace('"mola::mola_state_estimation_smoother::..."', "")

    for launch_file in (
        "mapping.launch.py",
        "mapping_bag.launch.py",
        "localization.launch.py",
    ):
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert f'default_value="{simple}"' in text
    for launch_file in ("mapping.launch.py", "localization.launch.py"):
        text = (PACKAGE_ROOT / "launch" / launch_file).read_text()
        assert (
            'SetEnvironmentVariable(\n'
            '            "MOLA_STATE_ESTIMATOR", LaunchConfiguration("state_estimator")\n'
            '        )'
        ) in text


def test_pose_fusion_node_is_launched_for_mapping():
    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    assert 'executable="pose_fusion_node"' in mapping
    assert 'DeclareLaunchArgument(\n            "correction_filter_weight", default_value="0.3"\n        )' in mapping
    # A weight this low was measured to leave most of a metre of standing error
    # against the reference, because LiDAR dead reckoning drifts ~0.2 m/s here.
    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "pose_fusion_node.py"
    ).read_text()
    assert 'declare_parameter("correction_filter_weight", 0.3)' in node


def test_pose_fusion_applies_corrections_from_the_lidar_callback():
    """On-arrival matching alone drops nearly every correction; see the docstring."""

    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "pose_fusion_node.py"
    ).read_text()
    lidar_callback = node[
        node.index("def _on_lidar") : node.index("def _on_reference")
    ]
    assert "self._apply_pending_references()" in lidar_callback

    setup = (PACKAGE_ROOT / "setup.py").read_text()
    assert (
        "pose_fusion_node = mola_slam_mapping.pose_fusion_node:main" in setup
    )


def test_smoother_does_not_contest_the_gnss_node_georeference_tf():
    """Both would publish enu->map and enu->utm; only one may own them."""

    system_yaml = (PACKAGE_ROOT / "config" / "mapping_system.yaml").read_text()
    assert (
        "publish_estimated_georef_on_convergence: "
        "${MOLA_PUBLISH_ESTIMATED_GEOREF|false}"
    ) in system_yaml
    # Still estimated internally: without it the smoother cannot place a GNSS fix
    # in the map frame and builds no GNSS factors at all.
    assert (
        'estimate_geo_reference: "${MOLA_ESTIMATE_GEO_REFERENCE|true}"'
    ) in system_yaml


def test_mapping_rviz_offers_vehicle_follow_views():
    """Saved views in the Views panel let the operator ride along with base_link."""

    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    saved = rviz[rviz.index("Saved:") : rviz.index("Window Geometry:")]
    assert "Class: rviz_default_plugins/ThirdPersonFollower" in saved
    assert saved.count("Target Frame: base_link") == 2
    assert "Name: Follow base_link (chase)" in saved
    assert "Name: Follow base_link (top-down)" in saved
    # The default view must stay the map overview so the route is visible on
    # start - which it is, because Global Options still opens on Fixed Frame
    # `map` and the camera now rides on whatever that is. Pinning it to `map`
    # instead made a base_link Fixed Frame look like it changed nothing.
    current = rviz[rviz.index("Current:") : rviz.index("Saved:")]
    assert "Target Frame: <Fixed Frame>" in current
    assert "Fixed Frame: map" in rviz[: rviz.index("  Views:")]


def test_mapping_rviz_uses_a_vehicle_marker_instead_of_a_pose_arrow():
    rviz = (PACKAGE_ROOT / "rviz" / "mapping.rviz").read_text()
    assert "Name: Vehicle\n" in rviz
    assert "Value: /vehicle_marker" in rviz
    assert "Name: Vehicle Pose" not in rviz

    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "map_odom_tf_node.py"
    ).read_text()
    assert 'declare_parameter("vehicle_marker_topic", "/vehicle_marker")' in node
    # Colour is a parameter, not a hardcoded literal, so it can be changed
    # without editing the node.
    assert 'declare_parameter(\n            "vehicle_marker_color_rgba"' in node
    assert "VEHICLE_MARKER_COLOR_RGBA = (0.0, 0.24, 1.0, 0.9)" in node


def test_mapping_enables_pre_map_lidar_pose_continuity_gate():
    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    assert '"lo_pose_step_base_m", default_value="0.20"' in mapping
    assert '"lo_max_linear_speed_mps", default_value="12.0"' in mapping
    assert '"lo_rotation_step_base_deg", default_value="3.0"' in mapping
    assert '"lo_max_angular_speed_deg_s", default_value="60.0"' in mapping
    assert '"MOLA_MAX_ACCEPTED_POSE_STEP_BASE"' in mapping
    assert '"MOLA_MAX_ACCEPTED_LINEAR_SPEED"' in mapping
    assert '"MOLA_MAX_ACCEPTED_ROTATION_STEP_BASE_DEG"' in mapping
    assert '"MOLA_MAX_ACCEPTED_ANGULAR_SPEED_DEG_S"' in mapping

    bag = (PACKAGE_ROOT / "launch" / "mapping_bag.launch.py").read_text()
    assert '"lo_pose_step_base_m", default_value="0.20"' in bag
    assert '"lo_max_linear_speed_mps", default_value="12.0"' in bag
    assert '"lo_rotation_step_base_deg", default_value="3.0"' in bag
    assert '"lo_max_angular_speed_deg_s", default_value="60.0"' in bag
    assert (
        '"lo_pose_step_base_m": LaunchConfiguration('
        in bag
    )
    assert (
        '"lo_max_linear_speed_mps": LaunchConfiguration('
        in bag
    )


def test_gnss_imu_pose_drives_global_visualization_and_corridor():
    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    localization = (
        PACKAGE_ROOT / "launch" / "localization.launch.py"
    ).read_text()
    bag = (PACKAGE_ROOT / "launch" / "mapping_bag.launch.py").read_text()

    for launch_text in (mapping, localization):
        assert 'executable="gnss_imu_pose_node"' in launch_text
        assert '"velocity_topic": LaunchConfiguration("gnss_velocity_topic")' in launch_text

    assert '"state_pose_topic": "/gnss_imu/pose"' in mapping
    assert (
        '"state_pose_topic": LaunchConfiguration("vehicle_pose_topic")'
        in localization
    )
    assert '"input_odom_topic": LaunchConfiguration(' in localization
    assert '"vehicle_pose_topic"' in localization
    assert '"map_dir": LaunchConfiguration("map_dir")' in localization
    assert "LaunchConfiguration(\"gnss_velocity_topic\")" in bag


def test_localization_can_initialize_stationary_against_the_selected_map():
    launch = (PACKAGE_ROOT / "launch" / "localization.launch.py").read_text()
    config = (
        PACKAGE_ROOT / "config" / "lidar_map_localization_defaults.yaml"
    ).read_text()
    setup = (PACKAGE_ROOT / "setup.py").read_text()

    assert '"lidar_map_initialization", default_value="auto"' in launch
    assert 'executable="lidar_map_localization_node"' in launch
    assert 'name="mola_lidar_map_localization"' in launch
    assert '"map_to_enu_yaw_deg": georeference.yaw_deg' in launch
    assert 'executable="localization_reference_pose_node"' in launch
    assert '"/lidar_odometry/pose:="' in launch
    assert 'LaunchConfiguration("mola_lidar_pose_topic")' in launch
    assert "mola_lidar_map_localization:" in config
    assert "initial_yaw_step_deg: 15.0" in config
    assert "localization_reference_pose_node =" in setup
    assert "prepare_lidar_localization_map =" in setup


def test_gnss_imu_and_lidar_pose_use_the_same_sensor_clock():
    node = (
        PACKAGE_ROOT / "mola_slam_mapping" / "gnss_imu_pose_node.py"
    ).read_text()
    system = (PACKAGE_ROOT / "config" / "mapping_system.yaml").read_text()
    mapping = (PACKAGE_ROOT / "launch" / "mapping.launch.py").read_text()
    localization = (
        PACKAGE_ROOT / "launch" / "localization.launch.py"
    ).read_text()

    assert (
        "output.header.stamp.sec = sample.stamp_ns // 1_000_000_000"
        in node
    )
    assert "output.header.stamp = self.get_clock().now().to_msg()" not in node
    assert "mola_pose_topic" not in node
    assert "/mola/gnss_imu/pose_sensor_time" not in system
    for launch_text in (mapping, localization):
        assert '"publish_in_sim_time", default_value="true"' in launch_text
        assert '"MOLA_ROS2_PUBLISH_IN_SIM_TIME"' in launch_text
