"""Read a mapping run's saved ENU alignment for localization startup."""

from dataclasses import dataclass
import math
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FixedGeoreference:
    latitude: float
    longitude: float
    altitude: float
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    def as_environment(self) -> dict[str, str]:
        values = {
            "MOLA_FIXED_GEO_REF_ENABLED": "true",
            "MOLA_FIXED_GEO_REF_LATITUDE": self.latitude,
            "MOLA_FIXED_GEO_REF_LONGITUDE": self.longitude,
            "MOLA_FIXED_GEO_REF_ALTITUDE": self.altitude,
            "MOLA_FIXED_GEO_REF_X": self.x,
            "MOLA_FIXED_GEO_REF_Y": self.y,
            "MOLA_FIXED_GEO_REF_Z": self.z,
            "MOLA_FIXED_GEO_REF_YAW_DEG": self.yaw_deg,
            "MOLA_FIXED_GEO_REF_PITCH_DEG": self.pitch_deg,
            "MOLA_FIXED_GEO_REF_ROLL_DEG": self.roll_deg,
        }
        return {key: str(value) for key, value in values.items()}


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite georeference value '{label}'")
    return result


def _normalized_quaternion(rotation: dict) -> tuple[float, float, float, float]:
    quaternion = tuple(
        _finite(rotation[key], f"rotation.{key}") for key in ("x", "y", "z", "w")
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("georeference quaternion has zero norm")
    return tuple(value / norm for value in quaternion)


def _euler_degrees(quaternion) -> tuple[float, float, float]:
    x, y, z, w = quaternion
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return tuple(math.degrees(angle) for angle in (yaw, pitch, roll))


def _multiply_quaternions(first, second):
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return _normalized_quaternion(
        {
            "x": w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            "y": w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            "z": w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            "w": w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        }
    )


def _rotate(quaternion, vector):
    x, y, z, w = quaternion
    vx, vy, vz = vector
    # Equivalent to q * [v,0] * q^-1, expanded to avoid dependencies.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def _pose_from_dict(pose: dict):
    translation = pose["translation"]
    return (
        tuple(
            _finite(translation[key], f"translation.{key}")
            for key in ("x", "y", "z")
        ),
        _normalized_quaternion(pose["rotation"]),
    )


def _compose(first, second):
    """Compose target<-middle with middle<-source into target<-source."""

    first_translation, first_rotation = first
    second_translation, second_rotation = second
    rotated = _rotate(first_rotation, second_translation)
    return (
        tuple(a + b for a, b in zip(first_translation, rotated)),
        _multiply_quaternions(first_rotation, second_rotation),
    )


def _from_native(path: Path) -> FixedGeoreference:
    with open(path) as stream:
        data = yaml.safe_load(stream)
    if data.get("type") != "mp2p_icp::Georeferencing" or not data.get("defined"):
        raise ValueError(f"'{path}' is not a defined mp2p_icp georeference")
    origin = data["geo_coord"]
    mean = data["T_enu_to_map"]["mean"]
    return FixedGeoreference(
        latitude=_finite(origin["lat"], "latitude"),
        longitude=_finite(origin["lon"], "longitude"),
        altitude=_finite(origin["altitude"], "altitude"),
        x=_finite(mean["x"], "x"),
        y=_finite(mean["y"], "y"),
        z=_finite(mean["z"], "z"),
        yaw_deg=_finite(mean["yaw_deg"], "yaw_deg"),
        pitch_deg=_finite(mean["pitch_deg"], "pitch_deg"),
        roll_deg=_finite(mean["roll_deg"], "roll_deg"),
    )


def load_fixed_georeference(
    map_directory: Path, map_filename: str = "global_map.mm"
) -> FixedGeoreference:
    """Load the sidecar paired with map_filename, then try its run manifest."""

    native_filename = Path(map_filename).with_suffix(".georef.yaml").name
    native_path = map_directory / native_filename
    if native_path.exists():
        return _from_native(native_path)

    manifest_path = map_directory / "georeference.yaml"
    with open(manifest_path) as stream:
        data = yaml.safe_load(stream)
    if int(data.get("schema_version", 0)) < 2:
        raise ValueError(
            f"'{manifest_path}' predates pointcloud-map global alignment metadata"
        )
    manifest_map = (
        data.get("artifacts", {}).get("pointcloud_map", {}).get("file")
    )
    if manifest_map is not None:
        if Path(str(manifest_map)).name != Path(map_filename).name:
            raise ValueError(
                f"'{manifest_path}' describes map '{manifest_map}', not "
                f"requested map '{map_filename}'"
            )
    elif Path(map_filename).name != "global_map.mm":
        raise ValueError(
            f"'{manifest_path}' cannot be paired safely with custom map "
            f"'{map_filename}'"
        )
    origin = data["geodetic_origin_wgs84"]
    if "pointcloud_map_to_enu" in data:
        transform = _pose_from_dict(
            data["pointcloud_map_to_enu"]["source_to_target"]
        )
    else:
        # Compatibility with early schema-v2 runs:
        # ENU<-saved_map = ENU<-UTM * UTM<-saved_map.
        transform = _compose(
            _pose_from_dict(data["enu_to_utm"]),
            _pose_from_dict(
                data["pointcloud_map_to_global"]["source_to_target"]
            ),
        )
    translation, quaternion = transform
    yaw, pitch, roll = _euler_degrees(quaternion)
    return FixedGeoreference(
        latitude=_finite(origin["latitude"], "latitude"),
        longitude=_finite(origin["longitude"], "longitude"),
        altitude=_finite(origin["height"], "altitude"),
        x=translation[0],
        y=translation[1],
        z=translation[2],
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
    )
