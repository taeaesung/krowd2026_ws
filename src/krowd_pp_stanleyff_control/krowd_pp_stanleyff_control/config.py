"""Configuration loading and readiness validation.

Loaded with PyYAML (not through ros__parameters, which cannot express `null`):

  * control config  - vehicle, speed policy, steering table, blend gains,
                      braking model, corridor margins, stop-line, signal and
                      heading-gate parameters. Map independent.
  * map profile     - per mapping run (map_profile.py): route checksum,
                      mission table, segment table, lane boundaries. The
                      profile is turned into the `mission` dict shape below.

`readiness()` classifies every unresolved value as BLOCKING (the controller
holds a NOT_READY stop) or WARNING (driving allowed, reported in the status).
A blocking value is never replaced by a default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import math
import os

import yaml

from .geometry import sha256_of_file

PACKAGE_NAME = "krowd_pp_stanleyff_control"
CONTROL_CONFIG = "control.yaml"


def default_config_dir() -> str:
    """The package's config directory.

    Preference order: the SOURCE tree (so an edited YAML is picked up without
    a rebuild, exactly as the launch file does), then the installed share copy.
    """

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(here, "config")
    if os.path.isdir(source) and os.path.isfile(os.path.join(here, "package.xml")):
        return source
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory(PACKAGE_NAME))
        workspace = share.parents[3]
        for depth in range(4):
            pattern = "src/" + "*/" * depth + PACKAGE_NAME
            for candidate in sorted(workspace.glob(pattern)):
                if (candidate / "package.xml").is_file() and (candidate / "config").is_dir():
                    return str(candidate / "config")
        return str(share / "config")
    except Exception:  # pragma: no cover - outside a ROS environment
        return source


def default_maps_dir() -> str:
    return os.path.join(default_config_dir(), "maps")


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict, overrides: dict) -> dict:
    """`overrides` written over `base`, recursing into nested mappings."""

    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_speed_mode(path) -> dict:
    """A config/speed_modes/<mode>.yaml: `route_publisher` (speed_profile.yaml
    parameters) and `control` (merged over control.yaml) sections."""

    data = load_yaml(path)
    for section in ("route_publisher", "control"):
        if not isinstance(data.get(section) or {}, dict):
            raise ValueError(f"{path}: '{section}' must be a mapping")
    return data


def load_control_config(path, overrides_path=None, speed_mode_path=None) -> dict:
    """control.yaml, then the speed mode's `control` section, then the overrides
    (the simulator's) - the last one written wins."""

    cfg = load_yaml(path)
    if speed_mode_path and str(speed_mode_path).strip():
        cfg = deep_merge(cfg, load_speed_mode(speed_mode_path).get("control") or {})
    if overrides_path and str(overrides_path).strip():
        cfg = deep_merge(cfg, load_yaml(overrides_path))
    return cfg


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


@dataclass
class Readiness:
    blocking: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        lines = ["READY" if self.ok else f"NOT READY ({len(self.blocking)} blocking)"]
        for item in self.blocking:
            lines.append(f"  BLOCKING: {item}")
        for item in self.warnings:
            lines.append(f"  warning : {item}")
        for key, value in self.info.items():
            lines.append(f"  info    : {key} = {value}")
        return "\n".join(lines)


# Values the team must measure or decide. None -> hold a NOT_READY stop.
BLOCKING_NUMBERS = (
    ("braking.safe_deceleration_mps2", "제동 안전 감속도 (m/s^2) - 대회 노면 통합시험으로 확정"),
    ("braking.total_latency_s", "감지->판단->CAN->제동 전체 지연 (s) - 통합시험 실측"),
    ("braking.distance_margin_m", "정지 여유거리 (m)"),
    ("stopline.stop_buffer_m", "stop anchor 앞 추가 정지 여유 (m)"),
    ("corridor.localization_error_m", "가상벽 축소용 localization 오차 (m)"),
    ("corridor.map_error_m", "가상벽 축소용 지도 오차 (m)"),
    ("corridor.tracking_error_m", "가상벽 축소용 경로추종 오차 (m)"),
    ("corridor.physical_margin_m", "가상벽 축소용 물리 여유 (m)"),
)

WARNING_NUMBERS = (
    ("vehicle.control_point_to_front_bumper_m", "control point->앞범퍼 실측값 (없으면 보수적 상한 사용)"),
    ("vehicle.control_point_to_rear_bumper_m", "control point->뒷범퍼 실측값 (없으면 전장-앞범퍼 거리 사용)"),
    ("stopline.live_ttl_s", "실시간 정지선 TTL (없으면 실시간 정지선 미사용, map anchor만 사용)"),
    ("stopline.map_live_association_tolerance_m", "map/live 정지선 연관 허용오차 (없으면 실시간 정지선 미사용)"),
)


def front_offset_m(vehicle: dict) -> tuple[float, str]:
    """Control point -> front bumper distance and where it came from.

    The measured value wins. Without it, the LARGER of the two documented
    bounds is used so the vehicle stops EARLIER, never later:
      vehicle length (2.464 m) and wheelbase + front-axle-to-bumper
      (1.340 + 1.194 = 2.534 m).
    """

    measured = vehicle.get("control_point_to_front_bumper_m")
    if is_number(measured):
        return float(measured), "measured"
    length = float(vehicle["length_m"])
    axle_bound = float(vehicle["wheelbase_m"]) + float(vehicle["front_axle_to_front_bumper_m"])
    return max(length, axle_bound), "conservative_bound(max(length, wheelbase+front_axle_to_bumper))"


def rear_offset_m(vehicle: dict, front: float) -> tuple[float, str]:
    measured = vehicle.get("control_point_to_rear_bumper_m")
    if is_number(measured):
        return float(measured), "measured"
    return max(float(vehicle["length_m"]) - front, 0.0), "derived(length - front)"


def _positive(control: dict, key: str, r: Readiness, allow_zero: bool = False) -> None:
    value = get(control, key)
    if not is_number(value) or float(value) < 0.0 or (float(value) == 0.0 and not allow_zero):
        r.blocking.append(f"{key} 는 {'0 이상' if allow_zero else '양수'}여야 함 (got {value!r})")


def readiness(control: dict, mission: dict, segments: dict | None = None, *, check_files: bool = True) -> Readiness:
    r = Readiness()

    for key, why in BLOCKING_NUMBERS:
        value = get(control, key)
        if not is_number(value):
            r.blocking.append(f"{key} 미입력: {why}")
        elif float(value) < 0.0:
            r.blocking.append(f"{key} 음수")
    for key, why in WARNING_NUMBERS:
        value = get(control, key)
        if not is_number(value):
            r.warnings.append(f"{key} 미입력: {why}")

    timeout = get(control, "signal.timeout_departure_s")
    if timeout is not None and (not is_number(timeout) or float(timeout) <= 0.0):
        r.blocking.append("signal.timeout_departure_s 는 null 또는 양수(초)여야 함")
    r.info["timeout_departure"] = (
        "disabled" if timeout is None else f"{float(timeout):.0f}s without a detected signal while waiting"
        + (" (detector must be alive)" if get(control, "signal.timeout_requires_detector_alive", True) else "")
    )
    commit_mode = get(control, "signal.commit_mode", "last_stoppable_point")
    if commit_mode not in ("last_stoppable_point", "front_bumper_at_line"):
        r.blocking.append("signal.commit_mode 는 last_stoppable_point 또는 front_bumper_at_line")
    elif commit_mode == "front_bumper_at_line":
        r.warnings.append("signal.commit_mode=front_bumper_at_line: 정지선 직전 신호 변경 시 위반 가능 (접근속도 감속 없음)")
    _positive(control, "speed.comfort_decel_mps2", r)
    _positive(control, "speed.start_ramp_accel_mps2", r, allow_zero=True)
    _positive(control, "speed.brake_release_hold_s", r, allow_zero=True)
    _positive(control, "steering.max_road_wheel_rate_deg_s", r)
    _positive(control, "corridor.fallback_half_width_m", r)

    # ---- blend --------------------------------------------------------------
    blend = control.get("blend") or {}
    try:
        from .blended_steering import BlendParams

        BlendParams(
            wheelbase_m=float(get(control, "vehicle.wheelbase_m", 0.0)),
            straight_radius_m=float(blend.get("straight_radius_m", 50.0)),
            curve_radius_m=float(blend.get("curve_radius_m", 5.0)),
            max_stanley_weight=float(blend.get("max_stanley_weight", 0.5)),
            heading_gain=float(blend.get("heading_gain", 0.6)),
            cross_track_gain=float(blend.get("cross_track_gain", 0.3)),
            softening_mps=float(blend.get("cross_track_softening_kmh", 10.8)) / 3.6,
            heading_window_m=float(blend.get("heading_window_m", 4.0)),
            yaw_rate_damping_gain_s=float(blend.get("yaw_rate_damping_gain_s", 0.15)),
            feedforward_gain=float(blend.get("feedforward_gain", 1.0)),
        ).validate()
        r.info["blend"] = (
            f"stanley weight 0 @R>={blend.get('straight_radius_m', 50.0)}m .. "
            f"{blend.get('max_stanley_weight', 0.5)} @R<={blend.get('curve_radius_m', 5.0)}m, "
            f"feedforward gain {blend.get('feedforward_gain', 1.0)}"
        )
    except (KeyError, TypeError, ValueError) as error:
        r.blocking.append(f"blend parameters invalid: {error}")

    # ---- cone-avoidance Stanley -------------------------------------------
    avoidance = control.get("avoidance_stanley") or {}
    if bool(avoidance.get("enabled", False)):
        for key, allow_zero in (
            ("avoidance_stanley.signal_timeout_s", False),
            ("avoidance_stanley.k_heading", True),
            ("avoidance_stanley.k_stanley", True),
            ("avoidance_stanley.k_soft_mps", False),
            ("avoidance_stanley.latency_compensation_s", True),
            ("avoidance_stanley.heading_window_m", True),
            ("avoidance_stanley.max_road_wheel_deg", False),
            ("avoidance_stanley.max_road_wheel_rate_deg_s", False),
        ):
            _positive(control, key, r, allow_zero=allow_zero)
        max_wheel = avoidance.get("max_road_wheel_deg")
        if is_number(max_wheel) and float(max_wheel) >= 80.0:
            r.blocking.append("avoidance_stanley.max_road_wheel_deg 는 80도 미만이어야 함")
        topic = str(avoidance.get("mode_topic", "")).strip()
        if not topic:
            r.blocking.append("avoidance_stanley.mode_topic 이 비어 있음")
        r.info["avoidance_stanley"] = (
            f"enabled by {topic}, 0924 gains "
            f"k={avoidance.get('k_stanley')}, heading={avoidance.get('k_heading')}, "
            f"latency={avoidance.get('latency_compensation_s')}s"
        )
    else:
        r.info["avoidance_stanley"] = "disabled"

    vehicle = control.get("vehicle", {})
    try:
        front, front_src = front_offset_m(vehicle)
        rear, rear_src = rear_offset_m(vehicle, front)
        r.info["front_offset_m"] = f"{front:.3f} ({front_src})"
        r.info["rear_offset_m"] = f"{rear:.3f} ({rear_src})"
    except (KeyError, TypeError, ValueError) as error:
        r.blocking.append(f"vehicle geometry incomplete: {error}")

    if not bool(get(control, "steering.table_verified", False)):
        cap = get(control, "speed.cap_until_steering_verified_kmh")
        r.warnings.append(
            "steering table not verified (2026-09-23 bag fit + hand-measured ends): "
            f"speed capped at {cap} km/h until steering.table_verified is true"
        )
    if get(control, "steering.zero_offset_rad") is None:
        r.warnings.append("steering.zero_offset_rad unset (0 assumed): re-measure after every /steering_center")

    live_ttl = get(control, "stopline.live_ttl_s")
    tol = get(control, "stopline.map_live_association_tolerance_m")
    r.info["live_stopline"] = "enabled" if (is_number(live_ttl) and is_number(tol)) else "disabled (map anchor only)"

    heading = control.get("heading") or {}
    r.info["heading_gate"] = (
        f"required: {heading.get('verified_topic', '/ekf/heading_verified')}=true"
        + (f", source={heading.get('required_source')}" if heading.get("required_source") else "")
        + (f"; until then HEADING_ALIGN (steering 0, {heading.get('align_speed_kmh', 5.0)} km/h)"
           if bool(heading.get("align_drive", False)) else "")
        if bool(heading.get("verified_required", True))
        else "DISABLED"
    )
    pause = control.get("pause") or {}
    r.info["pause"] = (
        f"zone-brake style: brake at stopping distance + {pause.get('stop_margin_m', 0.0)} m, front bumper "
        f"{pause.get('stop_gap_m', 0.5)} m before stop_idx, hold {pause.get('hold_s', 3.0)} s, "
        f"check from {pause.get('brake_window_m', 15.0)} m before the zone"
    )
    # ---- files ------------------------------------------------------------
    if check_files:
        for section, name in (("route", "waypoints_csv"), ("lane_boundaries", "csv")):
            block = mission.get(section) or {}
            path = block.get(name)
            expected = block.get("sha256")
            if not path:
                r.info[section] = "none (route_only)" if section == "lane_boundaries" else "not checked"
                continue
            path = os.path.expanduser(str(path))
            if not Path(path).is_file():
                r.blocking.append(f"{section}: file not found: {path}")
                continue
            actual = sha256_of_file(path)
            if expected and actual != expected:
                r.blocking.append(f"{section}: SHA-256 mismatch for {path} (expected {expected[:12]}..., got {actual[:12]}...)")
            else:
                r.info[f"{section}_sha256"] = actual[:16] + "..."

    # ---- missions --------------------------------------------------------
    from .missions import load_missions  # local import keeps config importable alone

    try:
        missions = load_missions(mission.get("missions", {}) or {})
        r.info["missions"] = ", ".join(
            f"{m.mission_id}({'on' if m.enabled else 'OFF'})" for m in missions
        ) or "none"
    except (KeyError, ValueError, TypeError) as error:
        r.blocking.append(f"mission table invalid: {error}")

    if segments is not None:
        from .segments import load_segments

        try:
            specs = load_segments(segments.get("segments", {}) or {})
            overrides = sum(
                1 for sp in specs
                if any(v is not None for v in (sp.speed_kmh, sp.lookahead_straight_m, sp.lookahead_curve_m,
                                                sp.straight_radius_m, sp.curve_radius_m, sp.max_road_wheel_rate_deg_s))
            )
            r.info["segments"] = f"{len(specs)} rows, {overrides} with overrides"
        except (KeyError, ValueError, TypeError) as error:
            r.blocking.append(f"segment table invalid: {error}")

    return r
