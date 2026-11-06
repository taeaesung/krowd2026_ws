"""Per-map profiles: which mission table, segment table and lane map go with
which mapping run.

    config/maps/<run_name>/profile.yaml       one directory per map
    config/maps/<run_name>/missions.yaml      (named in profile.yaml; optional)
    config/maps/<run_name>/segments.yaml      (optional)
    config/maps/<run_name>/lane_boundaries.csv (optional)

`run_name` is the basename of the map directory localization announces on
/localization/map_dir (symlinks resolved, so `latest` names the real run). A
run WITHOUT a profile directory runs `route_only`: no missions, no segment
overrides, no lane walls (the route corridor is the only virtual wall) and a
stop at the route's end - any freshly mapped test course is drivable without
writing a profile first. A run WITH a profile directory whose files do not
load, or whose waypoints.csv does not match the recorded SHA-256, is a
BLOCKING problem: the controller holds a stop rather than silently driving
the course without its missions.

The waypoints.csv the profile checksums is the localization map's own
(<map_dir>/waypoints.csv), the same file the route publisher reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Optional

import yaml

from .geometry import sha256_of_file

PROFILE_FILE = "profile.yaml"


@dataclass
class MapProfile:
    run_name: str
    display_name: str
    directory: Optional[str]  # profile directory; None for route_only
    map_dir: Optional[str]  # localization map directory (holds waypoints.csv)
    route_only: bool
    problem: Optional[str] = None  # set when a profile exists but cannot be used (start blocker)
    frame_id: str = "map"
    waypoints_csv: Optional[str] = None
    waypoints_sha256: Optional[str] = None
    expected_waypoint_count: Optional[int] = None
    missions_path: Optional[str] = None
    segments_path: Optional[str] = None
    lane_boundaries_path: Optional[str] = None
    lane_boundaries_sha256: Optional[str] = None
    expected_points_per_boundary: Optional[int] = None
    left_namespace_key: str = "leftBoundary"
    right_namespace_key: str = "rightBoundary"
    utm_cross_check_tolerance_m: float = 2.0
    notes: str = ""
    warnings: list = field(default_factory=list)

    @property
    def has_lane_map(self) -> bool:
        return self.lane_boundaries_path is not None

    @property
    def has_missions(self) -> bool:
        return self.missions_path is not None

    def label(self) -> str:
        if self.route_only:
            return f"{self.run_name}(no profile) route_only"
        return f"{self.run_name}({self.display_name})"

    def mission_config(self) -> dict:
        """The profile in the mission-config shape the controller modules read."""

        route = {"frame_id": self.frame_id}
        if self.waypoints_csv:
            route["waypoints_csv"] = self.waypoints_csv
        if self.waypoints_sha256:
            route["sha256"] = self.waypoints_sha256
        if self.expected_waypoint_count:
            route["expected_waypoint_count"] = self.expected_waypoint_count
            route["expected_trajectory_id_prefix"] = f"{self.run_name}:{self.expected_waypoint_count}wp"
        lanes = None
        if self.lane_boundaries_path:
            lanes = {
                "csv": self.lane_boundaries_path,
                "sha256": self.lane_boundaries_sha256,
                "left_namespace_key": self.left_namespace_key,
                "right_namespace_key": self.right_namespace_key,
                "expected_points_per_boundary": self.expected_points_per_boundary,
            }
        missions = {"list": [], "accept_pass_end_candidates": False}
        if self.missions_path:
            missions = _load_section(self.missions_path, "missions")
        segments = {"list": []}
        if self.segments_path:
            segments = _load_section(self.segments_path, "segments")
        return {"route": route, "lane_boundaries": lanes, "missions": missions, "segments": segments}


def _load_section(path: str, key: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict) or key not in data or not isinstance(data[key], dict):
        raise ValueError(f"{path}: expected a top-level '{key}:' mapping")
    return data[key]


def run_name_of(map_dir: str) -> str:
    """Basename of the map directory with symlinks resolved (`latest` -> the run)."""

    path = os.path.expanduser(str(map_dir).strip().rstrip("/"))
    if not path:
        return ""
    return os.path.basename(os.path.realpath(path))


def route_only_profile(run_name: str, map_dir: Optional[str]) -> MapProfile:
    profile = MapProfile(run_name=run_name, display_name="route_only", directory=None, map_dir=map_dir, route_only=True)
    if map_dir:
        candidate = os.path.join(os.path.expanduser(map_dir), "waypoints.csv")
        if os.path.isfile(candidate):
            profile.waypoints_csv = candidate
    return profile


def load_profile(directory: str, run_name: str, map_dir: Optional[str]) -> MapProfile:
    """Read <directory>/profile.yaml. Raises ValueError for anything unusable."""

    directory = os.path.abspath(os.path.expanduser(directory))
    path = os.path.join(directory, PROFILE_FILE)
    with open(path, "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")

    def resolve(name):
        value = raw.get(name)
        if value is None or str(value).strip() == "":
            return None
        value = os.path.expanduser(str(value))
        if not os.path.isabs(value):
            value = os.path.join(directory, value)
        if not os.path.isfile(value):
            raise ValueError(f"{path}: {name} '{value}' not found")
        return value

    def sha(name):
        # str(), not truthiness: YAML reads an all-digit digest as an integer.
        value = raw.get(name)
        if value is None or str(value).strip() == "":
            return None
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"{path}: {name} is not a 64-hex-digit SHA-256: {value!r}")
        return digest

    profile = MapProfile(
        run_name=run_name,
        display_name=str(raw.get("name", run_name)),
        directory=directory,
        map_dir=map_dir,
        route_only=False,
        frame_id=str(raw.get("frame_id", "map")),
        waypoints_sha256=sha("waypoints_sha256"),
        expected_waypoint_count=(int(raw["expected_waypoint_count"]) if raw.get("expected_waypoint_count") else None),
        missions_path=resolve("missions"),
        segments_path=resolve("segments"),
        lane_boundaries_path=resolve("lane_boundaries"),
        lane_boundaries_sha256=sha("lane_boundaries_sha256"),
        expected_points_per_boundary=(int(raw["expected_points_per_boundary"]) if raw.get("expected_points_per_boundary") else None),
        left_namespace_key=str(raw.get("left_namespace_key", "leftBoundary")),
        right_namespace_key=str(raw.get("right_namespace_key", "rightBoundary")),
        utm_cross_check_tolerance_m=float(raw.get("utm_cross_check_tolerance_m", 2.0)),
        notes=str(raw.get("notes", "")),
    )
    declared_run = raw.get("run_name")
    if declared_run is not None and not isinstance(declared_run, str):
        # YAML reads 20260913_103732 as the integer 20260913103732.
        raise ValueError(f"{path}: run_name must be QUOTED in YAML (read as {declared_run!r}, expected '{run_name}')")
    if declared_run and declared_run != run_name:
        raise ValueError(f"{path}: run_name '{declared_run}' does not match the directory '{run_name}'")

    waypoints = raw.get("waypoints_csv")
    if waypoints:
        waypoints = os.path.expanduser(str(waypoints))
    elif map_dir:
        waypoints = os.path.join(os.path.expanduser(map_dir), "waypoints.csv")
    if waypoints and os.path.isfile(waypoints):
        profile.waypoints_csv = waypoints
        if profile.waypoints_sha256:
            actual = sha256_of_file(waypoints)
            if actual != profile.waypoints_sha256:
                raise ValueError(
                    f"waypoints.csv SHA-256 mismatch for '{waypoints}': profile expects "
                    f"{profile.waypoints_sha256[:12]}..., file is {actual[:12]}... - this is a DIFFERENT "
                    "waypoints file; the mission waypoint indices would land in the wrong places"
                )
    elif waypoints:
        profile.warnings.append(f"waypoints.csv not found at '{waypoints}' (SHA check deferred to the trajectory)")
    if profile.lane_boundaries_path and profile.lane_boundaries_sha256:
        actual = sha256_of_file(profile.lane_boundaries_path)
        if actual != profile.lane_boundaries_sha256:
            raise ValueError(
                f"lane_boundaries SHA-256 mismatch for '{profile.lane_boundaries_path}': "
                f"expected {profile.lane_boundaries_sha256[:12]}..., got {actual[:12]}..."
            )
    return profile


def find_profile(maps_dir: str, run_name: str, map_dir: Optional[str]) -> MapProfile:
    """The profile for `run_name` under `maps_dir`, or a route_only one.

    A directory that exists but fails to load returns a profile with `problem`
    set (and route_only False): the controller must not drive on it.
    """

    if not run_name:
        return route_only_profile("", map_dir)
    directory = os.path.join(os.path.expanduser(maps_dir), run_name)
    if not os.path.isfile(os.path.join(directory, PROFILE_FILE)):
        return route_only_profile(run_name, map_dir)
    try:
        return load_profile(directory, run_name, map_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        profile = MapProfile(run_name=run_name, display_name="INVALID", directory=directory, map_dir=map_dir, route_only=False)
        profile.problem = f"map profile '{directory}' unusable: {error}"
        return profile


def list_profiles(maps_dir: str) -> list[str]:
    root = os.path.expanduser(maps_dir)
    if not os.path.isdir(root):
        return []
    return sorted(
        name for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, PROFILE_FILE))
    )
