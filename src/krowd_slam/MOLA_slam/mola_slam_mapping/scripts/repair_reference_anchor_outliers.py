#!/usr/bin/env python3
"""Repair only isolated end-keyframe placement failures in a corrected map.

This leaves all well-aligned keyframes and every orientation unchanged.  A
large interior mismatch is treated as a mapping failure, not silently patched.
Write to a new TUM; never overwrite the original corrected trajectory.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from mola_slam_mapping.map_retraject import interpolate_reference, read_tum, write_tum
from mola_slam_mapping.trajectory_io import read_trajectory_csv


def repair(run_dir: Path, output: Path, report: Path, threshold_m: float = 2.0):
    stamps, positions, quaternions = read_tum(run_dir / "keyframes_corrected.tum")
    ref_ns, frames, ref_positions, ref_quaternions = read_trajectory_csv(
        run_dir / "trajectory_reference.csv"
    )
    if set(frames) != {"map"}:
        raise ValueError("reference is not in map frame")
    matched_pos, _, valid = interpolate_reference(
        ref_ns * 1e-9, ref_positions, ref_quaternions, stamps, 0.5
    )
    # The first keyframe can precede the first 5 Hz reference by a few ms.
    # The vehicle is stationary there; a <=100 ms nearest-sample anchor is
    # safer than leaving a 20 m erroneous placement in the map.
    if 0 < ref_ns[0] * 1e-9 - stamps[0] <= 0.1:
        matched_pos[0] = ref_positions[0]
        valid[0] = True
    errors = np.linalg.norm(positions - matched_pos, axis=1)
    outliers = np.flatnonzero(valid & (errors > threshold_m))
    allowed = set(range(3)) | set(range(len(stamps) - 3, len(stamps)))
    if not set(outliers).issubset(allowed):
        raise ValueError(f"interior outliers present: {sorted(set(outliers) - allowed)}")
    if len(outliers) > 6:
        raise ValueError("too many end outliers")
    changed = [
        {"keyframe": int(index), "stamp": float(stamps[index]),
         "old_error_m": float(errors[index]),
         "old_position": positions[index].tolist(),
         "reference_position": matched_pos[index].tolist()}
        for index in outliers
    ]
    positions[outliers] = matched_pos[outliers]
    write_tum(output, stamps, positions, quaternions)
    with report.open("w") as stream:
        yaml.safe_dump(
            {"source": "keyframes_corrected.tum", "reference": "trajectory_reference.csv",
             "threshold_m": threshold_m, "repaired_keyframes": changed,
             "note": "Positions only; original orientations and all other poses unchanged."},
            stream, sort_keys=False,
        )
    print(f"repaired={len(outliers)} keyframes={outliers.tolist()} output={output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--threshold-m", type=float, default=2.0)
    args = parser.parse_args()
    if args.threshold_m <= 0.0:
        parser.error("threshold-m must be positive")
    repair(args.run_dir, args.output, args.report, args.threshold_m)


if __name__ == "__main__":
    main()
