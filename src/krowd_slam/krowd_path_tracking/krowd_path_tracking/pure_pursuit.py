"""The Pure Pursuit geometry itself, with no ROS types in sight.

Everything here is a pure function of numbers so it can be unit-tested against
hand-computed values and reused unchanged if the node ever moves to another
middleware.
"""

import math


def rear_axle_position(
    x: float, y: float, yaw: float, rear_axle_offset_m: float
) -> tuple[float, float]:
    """Slide a base_link position back to the rear axle along the body x axis."""

    return (
        x + rear_axle_offset_m * math.cos(yaw),
        y + rear_axle_offset_m * math.sin(yaw),
    )


def to_body_frame(
    origin_x: float, origin_y: float, yaw: float, target_x: float, target_y: float
) -> tuple[float, float]:
    """Express a point in the frame whose origin is (origin_x, origin_y) at `yaw`."""

    dx = target_x - origin_x
    dy = target_y - origin_y
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)


def steering_angle(
    target_x_body: float, target_y_body: float, wheelbase_m: float
) -> float:
    """Pure Pursuit: delta = atan(2 L sin(alpha) / ld), as atan2(2 L y, ld^2).

    Using the squared distance rather than dividing keeps the expression finite
    when the target collapses onto the rear axle, where the ratio would blow up.
    """

    squared_distance = target_x_body * target_x_body + target_y_body * target_y_body
    if squared_distance <= 0.0:
        return 0.0
    return math.atan2(2.0 * wheelbase_m * target_y_body, squared_distance)


def lookahead_distance(
    speed_mps: float, gain_s: float, minimum_m: float, maximum_m: float
) -> float:
    """Speed-scaled lookahead, ld = clamp(gain * v, minimum, maximum).

    The gain is a time: it is how many seconds ahead of itself the vehicle
    steers. Scaling with speed is what makes one tune work at both ends of the
    speed range - a fixed lookahead that is calm at 30 km/h oscillates at 50 and
    cuts every corner at 10.

    It is also this controller's only filter on heading noise. A yaw error of
    dpsi moves the commanded angle by roughly (2 L / ld) * dpsi, so a longer
    lookahead attenuates it - which matters here because the vehicle's heading
    comes from an RTK velocity course whose accuracy collapses at low speed.
    """

    if minimum_m > maximum_m:
        raise ValueError(
            f"lookahead minimum {minimum_m} exceeds maximum {maximum_m}"
        )
    return max(minimum_m, min(maximum_m, gain_s * max(speed_mps, 0.0)))


def curvature_lookahead(
    max_abs_curvature: float,
    straight_m: float,
    curve_m: float,
    straight_radius_m: float,
    curve_radius_m: float,
) -> float:
    """Lookahead from how tight the route ahead is, independent of speed.

    straight_m where the tightest radius ahead is straight_radius_m or more,
    curve_m where it is curve_radius_m or less, and linear in curvature
    (1 / radius) in between, so the lookahead shrinks smoothly on the way into
    a bend instead of jumping at a threshold.
    """

    straight_curvature = 1.0 / straight_radius_m
    curve_curvature = 1.0 / curve_radius_m
    weight = (abs(max_abs_curvature) - straight_curvature) / (
        curve_curvature - straight_curvature
    )
    weight = min(1.0, max(0.0, weight))
    return straight_m + weight * (curve_m - straight_m)


def rate_limited(
    previous_rad: float, desired_rad: float, max_rate_rad_s: float, dt_s: float
) -> float:
    """Clamp how far the steering command may move in one control period.

    Pure Pursuit recomputes its angle from scratch every frame, so a single
    outlier pose becomes a full-amplitude step at the actuator. MPC had this
    constraint built in (MAX_TAU); here it has to be applied explicitly.
    """

    if max_rate_rad_s <= 0.0 or dt_s <= 0.0:
        return desired_rad
    step = max_rate_rad_s * dt_s
    return max(previous_rad - step, min(previous_rad + step, desired_rad))


def clamp_steering(angle_rad: float, limit_rad: float) -> float:
    return max(-limit_rad, min(limit_rad, angle_rad))
