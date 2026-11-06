import io
from types import SimpleNamespace

import numpy as np
import yaml

from mola_slam_mapping.waypoint_saver_node import (
    WaypointSaverNode,
    _mrpt_diagonal_covariance,
    _yaml_builtin_scalars,
)


def test_yaml_builtin_scalars_converts_nested_numpy_numbers():
    source = {
        "transform": [np.float64(4.538472402917085), np.float32(-1.25)],
        "stamp": np.int64(123),
        "valid": True,
    }

    converted = _yaml_builtin_scalars(source)

    assert type(converted["transform"][0]) is float
    assert type(converted["transform"][1]) is float
    assert type(converted["stamp"]) is int
    assert type(converted["valid"]) is bool

    stream = io.StringIO()
    yaml.safe_dump(converted, stream, sort_keys=False)
    assert "4.538472402917085" in stream.getvalue()


def test_mrpt_covariance_uses_matrix_map_schema():
    covariance = _mrpt_diagonal_covariance(1e-6)

    assert covariance["rows"] == 6
    assert covariance["cols"] == 6
    assert len(covariance["data"]) == 36
    assert [covariance["data"][index * 6 + index] for index in range(6)] == [
        1e-6
    ] * 6
    assert sum(value != 0.0 for value in covariance["data"]) == 6


def test_lidar_georeference_cannot_overwrite_authoritative_estimator_value():
    authoritative = SimpleNamespace(valid=True, source="state_estimator")
    lagging_lidar = SimpleNamespace(valid=True, source="lidar_odometry")
    calls = []
    node = SimpleNamespace(
        _authoritative_georeference_received=False,
        _georeference_msg=None,
        _refresh_pointcloud_transform=lambda: calls.append("refresh"),
        _last_pointcloud_enu_transform=None,
        _save_georeference=lambda: calls.append("save"),
    )
    node._store_georeference = lambda msg: WaypointSaverNode._store_georeference(
        node, msg
    )

    WaypointSaverNode._on_map_georeference(node, lagging_lidar)
    assert node._georeference_msg is lagging_lidar

    WaypointSaverNode._on_georeference(node, authoritative)
    assert node._georeference_msg is authoritative
    assert node._authoritative_georeference_received

    WaypointSaverNode._on_map_georeference(node, lagging_lidar)
    assert node._georeference_msg is authoritative
    assert calls == ["refresh", "refresh"]


def test_marker_publish_is_skipped_after_rclpy_shutdown(monkeypatch):
    published = []
    node = SimpleNamespace(
        _marker_pub=SimpleNamespace(publish=lambda msg: published.append(msg))
    )
    monkeypatch.setattr(
        "mola_slam_mapping.waypoint_saver_node.rclpy.ok", lambda: False
    )

    WaypointSaverNode._publish_markers(node)

    assert published == []
