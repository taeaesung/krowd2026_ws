#!/usr/bin/env python3
"""Build a scan-to-map PLY from a bag and a mapping run's GNSS/IMU trajectory.

The resulting cloud is in the run's ``map`` frame.  It deliberately does not
reuse MOLA's keyframe poses: an odometry dropout must not move a scan tens of
metres away from its GNSS/IMU reference position.  The original MM is untouched.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation, Slerp
from sensor_msgs.msg import PointCloud2
from tf2_msgs.msg import TFMessage


def _trajectory(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {row["frame"] for row in rows} != {"map"}:
        raise ValueError("trajectory_reference.csv must contain poses in map")
    stamps = np.array([int(row["stamp_ns"]) * 1e-9 for row in rows])
    positions = np.array([[float(row[key]) for key in ("x", "y", "z")] for row in rows])
    quaternions = np.array(
        [[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows]
    )
    return stamps, positions, Slerp(stamps, Rotation.from_quat(quaternions))


def _pose_in_reference(stamp, times, positions, slerp):
    index = np.searchsorted(times, stamp)
    if index < 1 or index >= len(times) or times[index] - times[index - 1] > 0.5:
        return None
    fraction = (stamp - times[index - 1]) / (times[index] - times[index - 1])
    return positions[index - 1] * (1 - fraction) + positions[index] * fraction, slerp([stamp])[0]


def _static_lidar_to_base(transforms, frame):
    translation = np.zeros(3)
    rotation = Rotation.identity()
    visited = set()
    while frame != "base_link":
        if frame in visited or frame not in transforms:
            raise ValueError(f"no static TF chain from {frame} to base_link")
        visited.add(frame)
        tf = transforms[frame]
        t = tf.transform.translation
        q = tf.transform.rotation
        parent_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        translation = parent_rotation.apply(translation) + [t.x, t.y, t.z]
        rotation = parent_rotation * rotation
        frame = tf.header.frame_id
    return translation, rotation


def _point_array(message):
    dtype = np.dtype({
        "names": ["x", "y", "z", "t"],
        "formats": ["<f4", "<f4", "<f4", "<u4"],
        "offsets": [0, 4, 8, 20],
        "itemsize": message.point_step,
    })
    return np.frombuffer(message.data, dtype=dtype, count=message.width * message.height)


def build(bag, run_dir, output, voxel_m=0.20, stride=4):
    times, reference_positions, slerp = _trajectory(run_dir / "trajectory_reference.csv")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    static_tf = {}
    chunks = []
    scan_stamps = []
    accepted_scans = skipped_scans = 0
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic == "/tf_static":
            for tf in deserialize_message(serialized, TFMessage).transforms:
                static_tf[tf.child_frame_id] = tf
            continue
        if topic != "/ouster/points":
            continue
        cloud = deserialize_message(serialized, PointCloud2)
        try:
            lidar_translation, lidar_rotation = _static_lidar_to_base(
                static_tf, cloud.header.frame_id
            )
        except ValueError:
            skipped_scans += 1
            continue
        scan_start = cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9
        scan_stamps.append(scan_start)
        samples = _point_array(cloud)[::stride]
        xyz = np.column_stack((samples["x"], samples["y"], samples["z"]))
        ranges = np.linalg.norm(xyz, axis=1)
        valid = np.isfinite(xyz).all(axis=1) & (ranges >= 1.5) & (ranges <= 60.0)
        xyz = xyz[valid]
        offsets = samples["t"][valid] * 1e-9
        if len(xyz) == 0:
            skipped_scans += 1
            continue
        transformed = []
        # 10 ms motion compensation: at 6 m/s the residual translation is <=3 cm.
        bins = np.floor(offsets / 0.01).astype(np.int16)
        for bin_id in np.unique(bins):
            mask = bins == bin_id
            stamp = scan_start + float(np.median(offsets[mask]))
            pose = _pose_in_reference(stamp, times, reference_positions, slerp)
            if pose is None:
                continue
            position, orientation = pose
            base_points = lidar_rotation.apply(xyz[mask]) + lidar_translation
            transformed.append((orientation.apply(base_points) + position).astype("<f4"))
        if transformed:
            chunks.extend(transformed)
            accepted_scans += 1
        else:
            skipped_scans += 1
    if not chunks:
        raise ValueError("no usable LiDAR scans overlap the reference trajectory")
    points = np.concatenate(chunks)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    # Keep the first physical return in each voxel, without averaging edges.
    keys = np.floor(points / voxel_m).astype(np.int32)
    _, indices = np.unique(keys, axis=0, return_index=True)
    points = points[np.sort(indices)]
    with output.open("wb") as stream:
        stream.write(
            ("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {len(points)}\n"
             "property float x\nproperty float y\nproperty float z\nend_header\n").encode()
        )
        points.tofile(stream)
    gaps = np.diff(scan_stamps)
    result = {
        "bag": str(bag),
        "output": str(output),
        "raw_lidar_scans": len(scan_stamps),
        "accepted_scans": accepted_scans,
        "skipped_scans": skipped_scans,
        "accepted_fraction": accepted_scans / max(1, len(scan_stamps)),
        "points": len(points),
        "voxel_m": voxel_m,
        "stride": stride,
        "maximum_raw_scan_gap_s": float(gaps.max()) if len(gaps) else None,
        "raw_scan_gaps_over_0_25_s": int((gaps > 0.25).sum()),
    }
    print(
        f"accepted_scans={accepted_scans} skipped_scans={skipped_scans} "
        f"points={len(points)} max_gap_s={result['maximum_raw_scan_gap_s']} output={output}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-m", type=float, default=0.20)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.voxel_m <= 0 or args.stride < 1:
        parser.error("voxel-m and stride must be positive")
    result = build(args.bag, args.run_dir, args.output, args.voxel_m, args.stride)
    if args.report:
        temporary = Path(str(args.report) + ".tmp")
        with temporary.open("w") as stream:
            yaml.safe_dump(result, stream, sort_keys=False)
        os.replace(temporary, args.report)


if __name__ == "__main__":
    main()
