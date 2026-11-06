"""Rebuild a localization PLY from raw LiDAR scans and a trusted pose CSV.

This deliberately bypasses the mapping-time MOLA trajectory.  A missed early
scan-matching segment deformed that trajectory even after pose-graph
retrajecting; the saved GNSS/IMU reference trajectory is continuous and is the
right coordinate backbone for a GNSS-aided localization map.
"""

import argparse
import csv
import math
import os
import struct
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def pose_matrix(xyz, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(quaternion)
    matrix[:3, 3] = xyz
    return matrix


def compose_sensor_extrinsic():
    base_from_zed = pose_matrix((0.461, 0.0, 0.8576), (0.0, 0.0, 0.0, 1.0))
    zed_from_sensor = pose_matrix(
        (-0.0414949508, 0.0504791616, -0.0901744425),
        (0.00486081091, -0.00173969287, -0.0150764324, 0.999873016),
    )
    sensor_from_lidar = pose_matrix(
        (0.0, 0.0, 0.03618), (0.0, 0.0, 1.0, 0.0)
    )
    return base_from_zed @ zed_from_sensor @ sensor_from_lidar


def load_trajectory(path):
    samples = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            samples.append((
                int(row["stamp_ns"]),
                np.array([float(row[k]) for k in ("x", "y", "z")]),
                np.array([float(row[k]) for k in ("qx", "qy", "qz", "qw")]),
            ))
    if len(samples) < 2:
        raise ValueError("trajectory needs at least two samples")
    return samples


def interpolate_pose(samples, stamp_ns):
    stamps = [sample[0] for sample in samples]
    index = int(np.searchsorted(stamps, stamp_ns))
    if index <= 0 or index >= len(samples):
        return None
    before, after = samples[index - 1], samples[index]
    span = after[0] - before[0]
    if span <= 0 or min(stamp_ns-before[0], after[0]-stamp_ns) > 300_000_000:
        return None
    ratio = (stamp_ns - before[0]) / span
    xyz = before[1] * (1.0-ratio) + after[1] * ratio
    q0, q1 = before[2], after[2]
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    q = q0 * (1.0-ratio) + q1 * ratio
    return pose_matrix(xyz, q)


def deduplicate(points, voxel):
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def write_binary_ply(path, points):
    temporary = Path(str(path) + ".tmp")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    ).encode("ascii")
    with open(temporary, "wb") as stream:
        stream.write(header)
        stream.write(np.asarray(points, dtype="<f4").tobytes())
    os.replace(temporary, path)


def build(args):
    trajectory = load_trajectory(args.trajectory)
    base_from_lidar = compose_sensor_extrinsic()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.topic]))
    accumulated = np.zeros((0, 3), dtype=np.float32)
    pending = []
    scan_index = placed = skipped = 0
    while reader.has_next():
        _, raw, _ = reader.read_next()
        scan_index += 1
        if (scan_index - 1) % args.scan_stride:
            continue
        message = deserialize_message(raw, PointCloud2)
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        map_from_base = interpolate_pose(trajectory, stamp)
        if map_from_base is None:
            skipped += 1
            continue
        values = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        points = np.column_stack((values["x"], values["y"], values["z"])).astype(float)
        ranges = np.linalg.norm(points, axis=1)
        points = points[(ranges >= args.min_range) & (ranges <= args.max_range)]
        points = points @ base_from_lidar[:3, :3].T + base_from_lidar[:3, 3]
        points = points[(points[:, 2] >= args.min_z) & (points[:, 2] <= args.max_z)]
        points = deduplicate(points, args.voxel)
        points = points @ map_from_base[:3, :3].T + map_from_base[:3, 3]
        pending.append(np.asarray(points, dtype=np.float32))
        placed += 1
        if len(pending) >= 20:
            accumulated = deduplicate(
                np.vstack([accumulated, *pending]), args.voxel
            ).astype(np.float32)
            pending.clear()
            print(f"placed {placed} scans, {len(accumulated)} voxels", flush=True)
    if pending:
        accumulated = deduplicate(
            np.vstack([accumulated, *pending]), args.voxel
        ).astype(np.float32)
    write_binary_ply(args.output, accumulated)
    print(
        f"wrote {len(accumulated)} points from {placed} scans "
        f"({skipped} without a bracketing pose) to {args.output}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/ouster/points")
    parser.add_argument("--scan-stride", type=int, default=2)
    parser.add_argument("--voxel", type=float, default=0.30)
    parser.add_argument("--min-range", type=float, default=2.0)
    parser.add_argument("--max-range", type=float, default=70.0)
    parser.add_argument("--min-z", type=float, default=-2.5)
    parser.add_argument("--max-z", type=float, default=5.0)
    args = parser.parse_args(argv)
    build(args)


if __name__ == "__main__":
    main()
