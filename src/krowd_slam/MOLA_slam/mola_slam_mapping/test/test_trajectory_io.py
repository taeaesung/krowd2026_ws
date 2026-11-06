import csv

import numpy as np
from geometry_msgs.msg import Pose

from mola_slam_mapping.trajectory_io import (
    TRAJECTORY_COLUMNS,
    read_trajectory_csv,
    trajectory_row,
)


def _pose(x, y, z, qz=0.0, qw=1.0):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = x, y, z
    pose.orientation.z, pose.orientation.w = qz, qw
    return pose


def test_rows_round_trip_sorted_and_deduplicated(tmp_path):
    path = tmp_path / "trajectory_reference.csv"
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TRAJECTORY_COLUMNS)
        writer.writerow(trajectory_row(2_000_000_000, "map", _pose(2.0, 0.0, 0.5)))
        writer.writerow(trajectory_row(1_000_000_000, "map", _pose(1.0, 0.0, 0.25, 0.7071068, 0.7071068)))
        # A repeated stamp keeps the first row seen for it.
        writer.writerow(trajectory_row(2_000_000_000, "map", _pose(9.0, 9.0, 9.0)))

    stamps, frames, positions, quaternions = read_trajectory_csv(path)

    assert stamps.tolist() == [1_000_000_000, 2_000_000_000]
    assert frames == ["map", "map"]
    assert np.allclose(positions, [[1.0, 0.0, 0.25], [2.0, 0.0, 0.5]])
    assert np.allclose(quaternions[0], [0.0, 0.0, 0.7071068, 0.7071068])
    assert np.allclose(quaternions[1], [0.0, 0.0, 0.0, 1.0])


def test_row_keeps_the_exact_stamp_and_frame():
    row = trajectory_row(1_700_000_000_123_456_789, "odom", _pose(0.12345, -1.0, 0.0))

    assert row[0] == 1_700_000_000_123_456_789
    assert row[1] == "odom"
    assert row[2] == "0.1235"
    assert len(row) == len(TRAJECTORY_COLUMNS)
