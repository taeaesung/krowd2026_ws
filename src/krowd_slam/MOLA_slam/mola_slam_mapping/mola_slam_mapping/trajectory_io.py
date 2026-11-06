"""Stamped-trajectory CSV format shared by the recorder and the offline tools.

One row per pose message; ``stamp_ns`` keeps the exact ROS stamp so a row can
be matched against simplemap keyframe timestamps without float rounding.
"""

import csv
from pathlib import Path

import numpy as np

TRAJECTORY_COLUMNS = ("stamp_ns", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")


def trajectory_row(stamp_ns: int, frame: str, pose) -> list:
    """CSV row for a geometry_msgs Pose (position + quaternion)."""

    position, orientation = pose.position, pose.orientation
    return [
        int(stamp_ns),
        frame,
        f"{position.x:.4f}",
        f"{position.y:.4f}",
        f"{position.z:.4f}",
        f"{orientation.x:.7f}",
        f"{orientation.y:.7f}",
        f"{orientation.z:.7f}",
        f"{orientation.w:.7f}",
    ]


def read_trajectory_csv(path):
    """Return (stamp_ns int array, frames list, positions (N,3), quaternions (N,4) xyzw).

    Rows are returned stamp-sorted; a duplicated stamp keeps its first row.
    """

    stamps, frames, positions, quaternions = [], [], [], []
    with open(Path(path), newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            stamps.append(int(row["stamp_ns"]))
            frames.append(row["frame"])
            positions.append([float(row[key]) for key in ("x", "y", "z")])
            quaternions.append([float(row[key]) for key in ("qx", "qy", "qz", "qw")])
    stamps = np.asarray(stamps, dtype=np.int64)
    order = np.unique(stamps, return_index=True)[1]
    return (
        stamps[order],
        [frames[index] for index in order],
        np.asarray(positions, dtype=float).reshape(-1, 3)[order],
        np.asarray(quaternions, dtype=float).reshape(-1, 4)[order],
    )
