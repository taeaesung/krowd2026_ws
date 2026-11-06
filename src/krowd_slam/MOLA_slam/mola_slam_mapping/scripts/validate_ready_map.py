#!/usr/bin/env python3
"""Fail closed unless a finalized mapping run is suitable for localization."""

import argparse
import os
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from mola_slam_mapping.map_retraject import interpolate_reference, read_tum
from mola_slam_mapping.route_io import load_route_files
from mola_slam_mapping.trajectory_io import read_trajectory_csv


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
}


def load_ply_xyz(path):
    with path.open("rb") as stream:
        count = None
        properties = []
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"incomplete PLY header: {path}")
            line = raw.decode("ascii").strip()
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property ") and count is not None:
                _, kind, name = line.split()
                properties.append((name, PLY_TYPES[kind]))
            elif line == "end_header":
                break
        if count is None or not {"x", "y", "z"}.issubset(dict(properties)):
            raise ValueError(f"PLY has no xyz vertex layer: {path}")
        data = np.fromfile(stream, dtype=np.dtype(properties), count=count)
    xyz = np.column_stack([data[key] for key in ("x", "y", "z")]).astype(float)
    return xyz[np.isfinite(xyz).all(axis=1)]


def cloud_quality(path, sample_count=4000):
    points = load_ply_xyz(path)
    if len(points) < 20:
        raise ValueError(f"too few finite points in {path}: {len(points)}")
    rng = np.random.default_rng(20260923)
    sample = points[rng.choice(len(points), min(sample_count, len(points)), False)]
    _, neighbours = cKDTree(points).query(sample, k=20, workers=-1)
    thickness = []
    for indices in neighbours:
        eigenvalues = np.linalg.eigvalsh(np.cov(points[indices], rowvar=False))
        thickness.append(np.sqrt(max(0.0, eigenvalues[0])))
    thickness = np.asarray(thickness)
    return {
        "points": int(len(points)),
        "surface_thickness_median_m": float(np.median(thickness)),
        "surface_thickness_p95_m": float(np.percentile(thickness, 95)),
        "span_xyz_m": np.ptp(points, axis=0).tolist(),
    }


def validate(run_dir: Path):
    failures = []
    required = [
        "global_map_ready.mm", "global_map_ready.georef.yaml",
        "global_map_ready_localmap.ply", "gnss_reference_localmap.ply",
        "keyframes_ready.tum", "trajectory_reference.csv", "georeference.yaml",
        "waypoints.csv", "corridor.csv", "gnss_reference_build.yaml",
    ]
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"missing_or_empty:{name}")

    report = {"run_dir": str(run_dir), "required_files": required}
    if failures:
        report["failures"] = failures
        return report

    stamps, positions, _ = read_tum(run_dir / "keyframes_ready.tum")
    ref_ns, frames, ref_positions, ref_quaternions = read_trajectory_csv(
        run_dir / "trajectory_reference.csv"
    )
    if set(frames) != {"map"}:
        failures.append(f"reference_frames:{sorted(set(frames))}")
    reference, _, valid = interpolate_reference(
        ref_ns * 1e-9, ref_positions, ref_quaternions, stamps, 0.5
    )
    errors = np.linalg.norm(positions[valid] - reference[valid], axis=1)
    trajectory = {
        "keyframes": len(stamps),
        "matched_keyframes": int(valid.sum()),
        "matched_fraction": float(valid.mean()),
        "position_error_median_m": float(np.median(errors)),
        "position_error_p95_m": float(np.percentile(errors, 95)),
        "position_error_max_m": float(errors.max()),
    }
    report["trajectory"] = trajectory
    if trajectory["matched_fraction"] < 0.98:
        failures.append("trajectory_matched_fraction_below_0.98")
    if trajectory["position_error_median_m"] > 0.20:
        failures.append("trajectory_median_error_above_0.20m")
    if trajectory["position_error_p95_m"] > 0.75:
        failures.append("trajectory_p95_error_above_0.75m")
    if trajectory["position_error_max_m"] > 1.00:
        failures.append("trajectory_max_error_above_1.00m")

    route = load_route_files(run_dir / "waypoints.csv", run_dir / "corridor.csv", 3.0)
    route_points = np.asarray([[p.x, p.y] for p in route.points])
    route_length = float(np.linalg.norm(np.diff(route_points, axis=0), axis=1).sum())
    report["route"] = {
        "waypoints": len(route.points), "length_m": route_length,
        "corridor_valid": route.used_saved_corridor, "utm_zone": route.utm_zone,
    }
    if len(route.points) < 2 or not route.used_saved_corridor or route_length < 5.0:
        failures.append("route_or_corridor_invalid")

    with (run_dir / "gnss_reference_build.yaml").open() as stream:
        raw = yaml.safe_load(stream)
    report["raw_lidar"] = raw
    if raw["accepted_fraction"] < 0.95:
        failures.append("gnss_reference_scan_acceptance_below_0.95")
    if raw["points"] < 50000:
        failures.append("gnss_reference_cloud_below_50000_points")
    maximum_gap = raw.get("maximum_raw_scan_gap_s")
    if maximum_gap is None or maximum_gap > 0.25:
        failures.append("raw_lidar_gap_above_0.25s")

    report["mola_map_cloud"] = cloud_quality(run_dir / "global_map_ready_localmap.ply")
    report["gnss_reference_cloud"] = cloud_quality(run_dir / "gnss_reference_localmap.ply")
    for label in ("mola_map_cloud", "gnss_reference_cloud"):
        cloud = report[label]
        if cloud["surface_thickness_median_m"] > 0.20:
            failures.append(f"{label}_median_thickness_above_0.20m")
        if cloud["surface_thickness_p95_m"] > 0.60:
            failures.append(f"{label}_p95_thickness_above_0.60m")

    report["failures"] = failures
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    ready = run_dir / "READY_FOR_LOCALIZATION"
    not_ready = run_dir / "MAP_NOT_READY"
    ready.unlink(missing_ok=True)
    not_ready.unlink(missing_ok=True)
    report = validate(run_dir)
    passed = not report["failures"]
    report["ready_for_localization"] = passed
    report_path = run_dir / "map_quality.yaml"
    temporary = Path(str(report_path) + ".tmp")
    with temporary.open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    os.replace(temporary, report_path)
    marker = ready if passed else not_ready
    marker.write_text(
        ("PASS\nmap_file=global_map_ready.mm\n" if passed else
         "FAIL\nsee=map_quality.yaml\n")
    )
    print(f"ready_for_localization={passed} report={report_path} marker={marker}")
    if not passed:
        for failure in report["failures"]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
