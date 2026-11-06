"""CLI: offline report on one map profile's static inputs (no ROS needed).

Reads the map's waypoints.csv and georeference.yaml plus the profile's
mission table, segment table and lane boundaries and prints:
  * mission waypoint -> arc length table (what the controller will use)
  * boundary orientation / width / coverage of the route (uncovered waypoints
    -> those stretches are supervised by the route corridor instead)
  * lateral clearance of the route centreline to each wall
  * the static stop-line segments nearest to each stop anchor (information only)

    ros2 run krowd_pp_stanleyff_control static_input_report --map 20260913_103732 --map-dir ~/krowd_maps/20260913_103732
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import yaml

from .config import default_maps_dir
from .config_check import resolve_profile
from .corridor import StaticCorridor, load_boundaries, route_boundaries, validate_boundaries
from .missions import load_missions, resolve_missions
from .segments import load_segments, resolve_segments
from .route import RouteModel


def load_waypoints(path):
    rows = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append((float(row["utm_x"]), float(row["utm_y"]), float(row["utm_z"]), float(row["yaw_rad"])))
    return np.array(rows)


def utm_to_map(points_xyz, georeference):
    t = georeference["route_local_frame_to_global"]["source_to_target"]
    tr = np.array([t["translation"][k] for k in "xyz"])
    qx, qy, qz, qw = [t["rotation"][k] for k in "xyzw"]
    rot = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    return (rot.T @ (points_xyz - tr).T).T


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", default=default_maps_dir())
    parser.add_argument("--map", required=True, help="run name under maps-dir, or a profile directory")
    parser.add_argument("--map-dir", required=True, help="localization map directory (waypoints.csv, georeference.yaml)")
    parser.add_argument("--fallback-half-width", type=float, default=3.0)
    args = parser.parse_args(argv)
    profile = resolve_profile(args.maps_dir, args.map, args.map_dir)
    if profile.problem:
        print(profile.problem)
        return 1
    mission = profile.mission_config()

    wp_path = os.path.join(os.path.expanduser(args.map_dir), "waypoints.csv")
    wps = load_waypoints(wp_path)
    route = RouteModel.from_arrays(wps[:, :2], wps[:, 3], np.zeros(len(wps)), np.zeros(len(wps)))
    print(f"map profile: {profile.label()}")
    print(f"route: {len(wps)} waypoints, {route.length:.1f} m, from {wp_path}")
    if profile.expected_waypoint_count and profile.expected_waypoint_count != len(wps):
        print(f"  !! profile expects {profile.expected_waypoint_count} waypoints - the mission indices do NOT fit this file")

    table = load_missions(mission.get("missions", {}) or {})
    resolved, warnings = resolve_missions(table, route, bool((mission.get("missions") or {}).get("accept_pass_end_candidates", False)))
    print("\nmission arc-length table (control point = rear axle):")
    for rm in resolved:
        m = rm.mission
        if rm.kind == "signal":
            print(
                f"  {rm.mission_id} {m.direction:8s} detect #{m.detect_idx}={rm.start_s:7.2f} m  "
                f"decision #{m.decision_idx}={rm.decision_s:7.2f} m  stop #{m.stop_idx}={rm.stop_anchor_s:7.2f} m  "
                f"(decision->stop {rm.stop_anchor_s - rm.decision_s:4.2f} m)  end={rm.end_s:7.2f} m [{rm.end_source}]"
            )
        else:
            print(f"  {rm.mission_id} {rm.kind:8s} start #{m.start_idx}={rm.start_s:7.2f} m  end #{m.end_idx}={rm.end_s:7.2f} m")
    if not resolved:
        print("  (none - route_only)")
    for w in warnings:
        print(f"  warning: {w}")
    utm_points = [(m.mission_id, k, x, y) for m in table for k, x, y in m.utm_points]
    if utm_points:
        print("\nmission *_utm cross-check (UTM route, no TF needed):")
        for mid, key, x, y in utm_points:
            proj = route.project(x, y, None, None)
            idx = getattr(next(m for m in table if m.mission_id == mid), f"{key}_idx")
            s_idx = route.s_at_index(idx) if idx is not None else float("nan")
            print(f"  {mid}.{key}: utm -> s={proj.s if proj else float('nan'):.1f} m ({proj.distance if proj else float('nan'):.2f} m off), "
                  f"#{idx} -> s={s_idx:.1f} m, diff {abs((proj.s if proj else float('nan')) - s_idx):.1f} m")

    specs = load_segments(mission.get("segments", {}) or {})
    if specs:
        segs = resolve_segments(specs, route)
        print("\nsegments:")
        covered_s = 0.0
        for seg in segs:
            sp = seg.spec
            overrides = ", ".join(
                f"{k}={v}" for k, v in (
                    ("v_kmh", sp.speed_kmh), ("ld_straight", sp.lookahead_straight_m), ("ld_curve", sp.lookahead_curve_m),
                    ("R_straight", sp.straight_radius_m), ("R_curve", sp.curve_radius_m), ("rate_deg_s", sp.max_road_wheel_rate_deg_s),
                ) if v is not None
            ) or "inherit"
            covered_s += seg.end_s - seg.start_s
            print(f"  #{sp.start_idx:3d}-{sp.end_idx:3d}  s={seg.start_s:6.1f}..{min(seg.end_s, route.length):6.1f} m  {sp.name}: {overrides}")
        print(f"  route covered by segment rows: {min(covered_s, route.length):.0f} / {route.length:.0f} m (gaps use global values)")

    georef_path = os.path.join(os.path.dirname(wp_path), "georeference.yaml")
    lanes = mission.get("lane_boundaries")
    if not lanes:
        print("\nlane boundaries: none - the whole route is supervised by the route corridor")
        return 0
    bounds = load_boundaries(lanes["csv"], lanes.get("left_namespace_key") or "leftBoundary", lanes.get("right_namespace_key") or "rightBoundary")
    print(f"\nlane boundaries: {lanes['csv']}")
    print(f"  {validate_boundaries(bounds)}")
    if not os.path.isfile(georef_path):
        print(f"  (no {georef_path}; coverage check skipped)")
        return 0
    with open(georef_path) as stream:
        georef = yaml.safe_load(stream)
    wp_map = utm_to_map(wps[:, :3], georef)
    corridor = StaticCorridor(bounds, 0.0, 45)
    fallback = StaticCorridor(route_boundaries(wp_map[:, :2], args.fallback_half_width), 10.0, 45)
    uncovered = []
    left_cl, right_cl = [], []
    for i, (x, y) in enumerate(wp_map[:, :2]):
        corridor.reset()
        corridor.locate(x, y)
        cl = corridor.clearance(np.array([[x, y]]))
        left_cl.append(cl.left_min)
        right_cl.append(cl.right_min)
        if not cl.covered:
            uncovered.append(i)
    left_cl, right_cl = np.array(left_cl), np.array(right_cl)
    print(f"  centreline clearance to LEFT wall : min {left_cl.min():.2f} med {np.median(left_cl):.2f} max {left_cl.max():.2f} m")
    print(f"  centreline clearance to RIGHT wall: min {right_cl.min():.2f} med {np.median(right_cl):.2f} max {right_cl.max():.2f} m")
    print(f"  waypoints outside lane coverage (route corridor +/-{args.fallback_half_width:.1f} m there): {uncovered if uncovered else 'none'}")
    del fallback

    print("\nstatic stop-line segments in the lane map (information only; NOT used for braking):")
    for k, seg in enumerate(bounds.stoplines):
        c = seg.mean(axis=0)
        d = np.hypot(*(wp_map[:, :2] - c).T)
        i = int(np.argmin(d))
        near = ""
        for rm in resolved:
            if rm.kind == "signal" and abs(route.s_at_index(i) - rm.stop_anchor_s) < 15.0:
                near = f"  <- {rm.mission_id} anchor #{rm.mission.stop_idx} is {route.s_at_index(i) - rm.stop_anchor_s:+.1f} m before this line"
        print(f"  stopline {k}: nearest wp #{i} (s={route.s_at_index(i):.1f} m, {d[i]:.2f} m off){near}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
