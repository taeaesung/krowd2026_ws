import math

import pytest
import yaml

from mola_slam_mapping.georeference_io import load_fixed_georeference


def _write_yaml(path, value):
    with open(path, "w") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def test_native_georeference_is_loaded_for_localization(tmp_path):
    _write_yaml(
        tmp_path / "global_map.georef.yaml",
        {
            "type": "mp2p_icp::Georeferencing",
            "defined": True,
            "geo_coord": {"lat": 37.1, "lon": 127.2, "altitude": 42.0},
            "T_enu_to_map": {
                "mean": {
                    "x": 12.0,
                    "y": -4.0,
                    "z": 0.5,
                    "yaw_deg": 15.0,
                    "pitch_deg": 1.0,
                    "roll_deg": -2.0,
                },
                "cov": [[0.0] * 6 for _ in range(6)],
            },
        },
    )
    fixed = load_fixed_georeference(tmp_path)
    assert fixed.latitude == 37.1
    assert fixed.x == 12.0
    assert fixed.yaw_deg == 15.0
    assert fixed.as_environment()["MOLA_FIXED_GEO_REF_ENABLED"] == "true"


def test_manifest_fallback_composes_enu_utm_and_utm_map(tmp_path):
    half = math.sqrt(0.5)
    _write_yaml(
        tmp_path / "georeference.yaml",
        {
            "schema_version": 2,
            "geodetic_origin_wgs84": {
                "latitude": 37.0,
                "longitude": 127.0,
                "height": 10.0,
            },
            "enu_to_utm": {
                "translation": {"x": 100.0, "y": 200.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": half, "w": half},
            },
            "pointcloud_map_to_global": {
                "source_to_target": {
                    "translation": {"x": 10.0, "y": 0.0, "z": 0.0},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
        },
    )
    fixed = load_fixed_georeference(tmp_path)
    assert (fixed.x, fixed.y, fixed.z) == pytest.approx((100.0, 210.0, 0.0))
    assert fixed.yaw_deg == pytest.approx(90.0)


def test_old_manifest_without_map_alignment_is_rejected(tmp_path):
    _write_yaml(tmp_path / "georeference.yaml", {"schema_version": 1})
    with pytest.raises(ValueError, match="predates"):
        load_fixed_georeference(tmp_path)


def test_custom_map_uses_matching_native_sidecar(tmp_path):
    _write_yaml(
        tmp_path / "alternate.georef.yaml",
        {
            "type": "mp2p_icp::Georeferencing",
            "defined": True,
            "geo_coord": {"lat": 37.1, "lon": 127.2, "altitude": 42.0},
            "T_enu_to_map": {
                "mean": {
                    "x": 7.0,
                    "y": 8.0,
                    "z": 0.0,
                    "yaw_deg": 0.0,
                    "pitch_deg": 0.0,
                    "roll_deg": 0.0,
                },
                "cov": [[0.0] * 6 for _ in range(6)],
            },
        },
    )
    fixed = load_fixed_georeference(tmp_path, "alternate.mm")
    assert (fixed.x, fixed.y) == (7.0, 8.0)


def test_manifest_for_different_map_is_rejected(tmp_path):
    _write_yaml(
        tmp_path / "georeference.yaml",
        {
            "schema_version": 2,
            "artifacts": {"pointcloud_map": {"file": "global_map.mm"}},
        },
    )
    with pytest.raises(ValueError, match="not requested map"):
        load_fixed_georeference(tmp_path, "alternate.mm")
