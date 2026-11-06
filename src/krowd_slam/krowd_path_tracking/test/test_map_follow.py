"""The map is named once, in localization; path tracking follows it."""

from pathlib import Path


PATH_TRACKING = Path(__file__).resolve().parents[1]
MAPPING = PATH_TRACKING.parents[0] / "MOLA_slam" / "mola_slam_mapping"


def test_path_tracking_defaults_to_following_the_localization_map():
    launch = (PATH_TRACKING / "launch" / "path_tracking.launch.py").read_text()
    assert '"map_dir",\n            default_value="",' in launch

    node = (
        PATH_TRACKING / "krowd_path_tracking" / "route_publisher_node.py"
    ).read_text()
    assert 'declare_parameter("map_dir_topic", "/localization/map_dir")' in node
    # An explicit route still wins over the announced one.
    assert 'self._load_route(str(self.get_parameter("map_dir").value))' in node
    assert "self.create_subscription(String, topic, self._on_map_dir, latched)" in node


def test_localization_announces_the_run_it_loaded():
    launch = (MAPPING / "launch" / "localization.launch.py").read_text()
    assert 'executable="selected_map_node"' in launch
    assert '"map_dir": str(map_directory)' in launch
    announcement = (
        'DeclareLaunchArgument(\n            "map_dir_topic", '
        'default_value="/localization/map_dir"\n        )'
    )
    assert announcement in launch

    setup = (MAPPING / "setup.py").read_text()
    assert "selected_map_node = mola_slam_mapping.selected_map_node:main" in setup


def test_the_announcement_is_latched_for_a_late_subscriber():
    """Path tracking normally starts after localization has already said it."""

    node = (MAPPING / "mola_slam_mapping" / "selected_map_node.py").read_text()
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in node
    start = node.index("self._publisher = self.create_publisher")
    publisher = node[start:]
    assert "latched" in publisher.split("\n")[0] + publisher.split("\n")[1]
