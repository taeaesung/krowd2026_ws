from glob import glob
from setuptools import find_packages, setup


package_name = "mola_slam_mapping"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        (
            "share/" + package_name + "/scripts",
            glob("scripts/*.sh") + glob("scripts/*.py"),
        ),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="krowd",
    maintainer_email="krowd2026@gmail.com",
    description="Global MOLA mapping, localization, route, and corridor visualization.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "waypoint_saver_node = mola_slam_mapping.waypoint_saver_node:main",
            "map_odom_tf_node = mola_slam_mapping.map_odom_tf_node:main",
            "pose_fusion_node = mola_slam_mapping.pose_fusion_node:main",
            "mark_waypoint_keyboard_node = mola_slam_mapping.mark_waypoint_keyboard_node:main",
            "localization_monitor_node = mola_slam_mapping.localization_monitor_node:main",
            "wheel_odom_bicycle_node = mola_slam_mapping.wheel_odom_bicycle_node:main",
            "wheel_odom_from_bag_node = mola_slam_mapping.wheel_odom_from_bag_node:main",
            "imu_gate_node = mola_slam_mapping.imu_gate_node:main",
            "imu_downsample_node = mola_slam_mapping.imu_downsample_node:main",
            "gnss_covariance_floor_node = mola_slam_mapping.gnss_covariance_floor_node:main",
            "gnss_imu_pose_node = mola_slam_mapping.gnss_imu_pose_node:main",
            "map_pointcloud_voxel_filter_node = mola_slam_mapping.map_pointcloud_voxel_filter_node:main",
            "trajectory_recorder_node = mola_slam_mapping.trajectory_recorder_node:main",
            "corrected_map_preview_node = mola_slam_mapping.corrected_map_preview_node:main",
            "map_retraject = mola_slam_mapping.map_retraject:main",
            "rtk_initial_pose_node = mola_slam_mapping.rtk_initial_pose_node:main",
            "route_initial_pose_node = mola_slam_mapping.route_initial_pose_node:main",
            "localization_reference_pose_node = mola_slam_mapping.localization_reference_pose_node:main",
            "global_odometry_map_pose_node = mola_slam_mapping.global_odometry_map_pose_node:main",
            "selected_map_node = mola_slam_mapping.selected_map_node:main",
            "prepare_lidar_localization_map = mola_slam_mapping.prepare_lidar_localization_map:main",
            "lane_map_overlay_node = mola_slam_mapping.lane_map_overlay_node:main",
            "saved_lane_map_node = mola_slam_mapping.saved_lane_map_node:main",
            "build_localization_point_map = mola_slam_mapping.build_localization_point_map:main",
        ],
    },
)
