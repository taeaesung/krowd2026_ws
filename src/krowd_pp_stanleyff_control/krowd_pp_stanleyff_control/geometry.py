"""Planar geometry helpers shared by the controller modules.

Everything here is plain numpy / math with no ROS types, so it can be unit
tested and reused offline (static input report, bag analysis).

Conventions (REP-103): +x forward, +y left, yaw counter-clockwise from +x.
The control point is the rear-axle centre, which sensor_extrinsics.yaml
declares to be the origin of `base_link`.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np


def yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation about +z."""

    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def to_body_frame(
    origin_x: float, origin_y: float, yaw: float, target_x: float, target_y: float
) -> tuple[float, float]:
    """A world point expressed in the frame at (origin, yaw)."""

    dx = target_x - origin_x
    dy = target_y - origin_y
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def footprint_corners(
    x: float,
    y: float,
    yaw: float,
    front_m: float,
    rear_m: float,
    half_width_m: float,
) -> np.ndarray:
    """The four corners of the rotated vehicle rectangle, shape (4, 2).

    `front_m` is the distance from the control point (rear axle) to the front
    bumper along +x, `rear_m` the distance back to the rear bumper. Both are
    positive numbers.
    """

    c = math.cos(yaw)
    s = math.sin(yaw)
    local = np.array(
        [
            [front_m, half_width_m],
            [front_m, -half_width_m],
            [-rear_m, -half_width_m],
            [-rear_m, half_width_m],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def bicycle_step(
    x: float,
    y: float,
    yaw: float,
    speed_mps: float,
    steering_rad: float,
    wheelbase_m: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """Kinematic bicycle model integrated at the rear axle."""

    x_next = x + speed_mps * math.cos(yaw) * dt_s
    y_next = y + speed_mps * math.sin(yaw) * dt_s
    yaw_next = yaw + speed_mps / wheelbase_m * math.tan(steering_rad) * dt_s
    return x_next, y_next, wrap_angle(yaw_next)


def cumulative_length(xy: np.ndarray) -> np.ndarray:
    """Arc length at each vertex of a polyline, starting at 0."""

    if len(xy) == 0:
        return np.zeros(0)
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(seg)])


def nearest_on_segments(points: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Vectorised point-to-segment projection.

    points: (K, 2); a, b: (S, 2) segment ends.
    Returns (dist (K,S), t_clamped (K,S), t_raw (K,S), cross (K,S)) where
    cross > 0 means the point lies LEFT of the directed segment a->b.
    """

    ab = b - a
    ap = points[:, None, :] - a[None, :, :]
    denom = np.maximum((ab * ab).sum(axis=1), 1e-12)
    t_raw = (ap * ab[None, :, :]).sum(axis=2) / denom[None, :]
    t = np.clip(t_raw, 0.0, 1.0)
    q = a[None, :, :] + t[..., None] * ab[None, :, :]
    diff = points[:, None, :] - q
    dist = np.hypot(diff[..., 0], diff[..., 1])
    cross = ab[None, :, 0] * ap[..., 1] - ab[None, :, 1] * ap[..., 0]
    return dist, t, t_raw, cross


def sha256_of_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def interpolate_table(xs, ys, x: float) -> float:
    """Piecewise-linear interpolation, HELD (not extrapolated) beyond the ends."""

    if not math.isfinite(x):
        return float(x)
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    index = int(np.searchsorted(xs, x, side="right")) - 1
    index = min(max(index, 0), len(xs) - 2)
    x0, x1 = xs[index], xs[index + 1]
    y0, y1 = ys[index], ys[index + 1]
    if x1 == x0:
        return float(y0)
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
