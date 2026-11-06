"""The Stanley steering law itself, with no ROS types in sight.

Same split as pure_pursuit.py: pure functions of numbers, unit-tested against
hand-computed values. Steering angles follow REP-103 - positive is a LEFT turn -
while RouteProjection.signed_offset is positive when a point lies LEFT of the
route, so a vehicle left of the route must steer negative (right).
"""

import bisect
import math


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def front_axle_position(
    x: float, y: float, yaw: float, rear_axle_offset_m: float, wheelbase_m: float
) -> tuple[float, float]:
    """Slide a base_link position forward to the front axle along the body x axis.

    Stanley measures the cross-track error at the FRONT axle - the point the
    steered wheels move - and its convergence result is derived there.
    rear_axle_offset_m is Vehicle's: the rear axle relative to base_link along +x.
    """

    distance = rear_axle_offset_m + wheelbase_m
    return (x + distance * math.cos(yaw), y + distance * math.sin(yaw))


def _point_at_arc(
    points: list[tuple[float, float]], cumulative: list[float], arc_m: float
) -> tuple[float, float]:
    arc_m = min(max(arc_m, 0.0), cumulative[-1])
    index = min(max(bisect.bisect_right(cumulative, arc_m) - 1, 0), len(points) - 2)
    length = cumulative[index + 1] - cumulative[index]
    fraction = (arc_m - cumulative[index]) / length if length > 0.0 else 0.0
    start, end = points[index], points[index + 1]
    return (
        start[0] + fraction * (end[0] - start[0]),
        start[1] + fraction * (end[1] - start[1]),
    )


def path_heading(
    points: list[tuple[float, float]],
    segment_index: int,
    fraction: float,
    window_m: float,
) -> float:
    """Direction of the route at a projected position, from its GEOMETRY.

    The chord between the points window_m / 2 behind and ahead along the arc,
    clamped to the route's ends. Deliberately not the recorded waypoint yaw:
    that is the mapping drive's own heading estimate, noise included, and
    Stanley passes the heading error to the steering at unity gain. With ~1 m
    waypoint spacing a single segment's direction also steps at every waypoint;
    the window averages that out. window_m <= 0 uses the projected segment alone.
    """

    if len(points) < 2:
        raise ValueError("a route needs at least two points to have a heading")
    index = min(max(segment_index, 0), len(points) - 2)
    clamped = min(max(fraction, 0.0), 1.0)

    if window_m > 0.0:
        cumulative = [0.0]
        for start, end in zip(points, points[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(end[0] - start[0], end[1] - start[1])
            )
        here = cumulative[index] + clamped * (cumulative[index + 1] - cumulative[index])
        behind = _point_at_arc(points, cumulative, here - 0.5 * window_m)
        ahead = _point_at_arc(points, cumulative, here + 0.5 * window_m)
        if math.hypot(ahead[0] - behind[0], ahead[1] - behind[1]) > 1e-6:
            return math.atan2(ahead[1] - behind[1], ahead[0] - behind[0])

    start, end = points[index], points[index + 1]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def cross_track_steering(
    signed_offset_m: float, speed_mps: float, gain: float, softening_mps: float
) -> float:
    """Stanley's second term, atan(k e / (k_soft + v)), steering back to the route.

    signed_offset_m uses RouteProjection's convention (positive = left of the
    route), so the term is negated: left of the route steers right. The
    softening speed keeps the term finite at a standstill, where plain
    atan(k e / v) is not, and stops a crawling vehicle being told to turn hard
    for a small offset. Reversing is not modelled; negative speed counts as zero.
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
    heading comes from an RTK velocity course that is noisy at low speed, and at
    unity gain that noise reaches the steering undiminished.
    """

    return heading_gain * wrap_angle(heading_error_rad) + cross_track_steering(
        signed_offset_m, speed_mps, cross_track_gain, softening_mps
    )


def heading_not_aligned(heading_error_rad: float, limit_rad: float) -> bool:
    """True when the vehicle and route directions differ by more than limit_rad.

    limit_rad <= 0 disables the check.
    """

    return limit_rad > 0.0 and abs(wrap_angle(heading_error_rad)) > limit_rad
