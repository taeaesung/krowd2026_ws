"""The Stanley terms used by the blend, with no ROS types in sight.

Pure functions of numbers, unit-tested against hand-computed values.
Steering angles follow REP-103 - positive is a LEFT turn - while the route
projection's lateral offset is positive when a point lies LEFT of the route,
so a vehicle left of the route must steer negative (right).

Carried over from krowd_stanley_control (2026-09-20). Deliberately NOT
carried over: the /encoder lead limit (the steer Teensy is the position
controller; the main PC does not close that loop again) and the projection
window helpers (RouteModel.project owns continuity here).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import wrap_angle


@dataclass(frozen=True)
class FrontAxlePose:
    """Front-axle pose predicted with the same bicycle model as cone_avoidance_0924_3."""

    x: float
    y: float
    yaw: float


def front_axle_position(
    x: float, y: float, yaw: float, rear_axle_offset_m: float, wheelbase_m: float
) -> tuple[float, float]:
    """Slide a base_link position forward to the front axle along the body x axis.

    Stanley measures the cross-track error at the FRONT axle - the point the
    steered wheels move - and its convergence result is derived there.
    rear_axle_offset_m is the rear axle relative to base_link along +x.
    """

    distance = rear_axle_offset_m + wheelbase_m
    return (x + distance * math.cos(yaw), y + distance * math.sin(yaw))


def predict_front_axle_pose(
    x_front: float,
    y_front: float,
    yaw: float,
    speed_mps: float,
    previous_steering_rad: float,
    wheelbase_m: float,
    latency_s: float,
) -> FrontAxlePose:
    """Predict the front axle through the steering-system latency.

    This is the controller prediction used by ``cone_avoidance_0924_3``:
    recover the rear axle, advance a bicycle model while holding the previous
    road-wheel command, then rebuild the front axle.  Keeping this pure makes
    the integrated controller directly testable against the proven equation.
    """

    if wheelbase_m <= 0.0:
        raise ValueError(f"wheelbase must be positive: {wheelbase_m}")
    if latency_s < 0.0:
        raise ValueError(f"latency must be non-negative: {latency_s}")
    x_rear = x_front - wheelbase_m * math.cos(yaw)
    y_rear = y_front - wheelbase_m * math.sin(yaw)
    yaw_rate = speed_mps / wheelbase_m * math.tan(previous_steering_rad)
    yaw_mid = yaw + 0.5 * yaw_rate * latency_s
    x_rear += speed_mps * math.cos(yaw_mid) * latency_s
    y_rear += speed_mps * math.sin(yaw_mid) * latency_s
    yaw_predicted = yaw + yaw_rate * latency_s
    return FrontAxlePose(
        x_rear + wheelbase_m * math.cos(yaw_predicted),
        y_rear + wheelbase_m * math.sin(yaw_predicted),
        yaw_predicted,
    )


def cross_track_steering(
    signed_offset_m: float, speed_mps: float, gain: float, softening_mps: float
) -> float:
    """Stanley's second term, atan(k e / (k_soft + v)), steering back to the route.

    signed_offset_m is positive LEFT of the route, so the term is negated:
    left of the route steers right. The softening speed keeps the term finite
    at a standstill, where plain atan(k e / v) is not, and stops a crawling
    vehicle being told to turn hard for a small offset. Reversing is not
    modelled; negative speed counts as zero.
    """

    if softening_mps <= 0.0:
        raise ValueError(f"softening speed must be positive: {softening_mps}")
    return -math.atan2(gain * signed_offset_m, softening_mps + max(speed_mps, 0.0))


def steering_angle(
    heading_error_rad: float,
    signed_offset_m: float,
    speed_mps: float,
    cross_track_gain: float,
    softening_mps: float,
    heading_gain: float = 1.0,
) -> float:
    """delta = heading_gain * (route yaw - vehicle yaw) + atan(k e / (k_soft + v)).

    heading_gain is 1.0 in Stanley's law. It is exposed because this vehicle's
    heading comes from an RTK velocity course that is noisy at low speed, and
    at unity gain that noise reaches the steering undiminished.
    """

    return heading_gain * wrap_angle(heading_error_rad) + cross_track_steering(
        signed_offset_m, speed_mps, cross_track_gain, softening_mps
    )


def yaw_rate_damping(
    speed_mps: float,
    path_curvature_per_m: float,
    measured_yaw_rate_rad_s: float,
    gain_s: float,
) -> float:
    """Steering correction (rad, road wheel) from the yaw-rate error.

        gain_s * (speed * curvature - measured yaw rate)

    The basic laws only see where the vehicle IS; with a steering motor that
    lags its command the vehicle is already turning back by the time the
    offset changes sign, and the law adds to that turn - the wide weaving of
    the 2026-09-20 drives. This term opposes the rotation the path does not
    ask for: on a straight, a vehicle yawing left at 10 deg/s gets 1.5 deg to
    the right at gain 0.15 s. On a curve driven correctly it is zero, because
    the path's own yaw rate is subtracted first (Thrun et al. 2006, 9.2).

    Curvature and yaw rate are both positive to the LEFT. A negative speed
    counts as zero. gain_s <= 0 disables the term.
    """

    if gain_s <= 0.0:
        return 0.0
    desired = max(speed_mps, 0.0) * path_curvature_per_m
    return gain_s * (desired - measured_yaw_rate_rad_s)
