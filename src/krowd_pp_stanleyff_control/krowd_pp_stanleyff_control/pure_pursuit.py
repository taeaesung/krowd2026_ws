"""Pure Pursuit geometry on the Global Path (pure functions)."""

from __future__ import annotations

import math


def curvature_lookahead(
    max_abs_curvature: float,
    straight_m: float,
    curve_m: float,
    straight_radius_m: float,
    curve_radius_m: float,
) -> float:
    """straight_m on straights, curve_m in tight bends, linear in curvature between."""

    straight_k = 1.0 / straight_radius_m
    curve_k = 1.0 / curve_radius_m
    weight = (abs(max_abs_curvature) - straight_k) / (curve_k - straight_k)
    weight = min(1.0, max(0.0, weight))
    return straight_m + weight * (curve_m - straight_m)


def time_floored_lookahead(lookahead_m: float, speed_mps: float, time_s: float) -> float:
    """lookahead_m, but never shorter than time_s seconds of travel (0 = off).

    The curvature rule shortens the lookahead in a bend whatever the speed;
    at 40 km/h a 3-5 m lookahead is under half a second ahead. 0.6 s was the
    best of 0.6-1.2 s in the 2026-10-02 closed-loop sweep (0.24 s steering
    dead time): longer cut the bends of 20260913_103732. Up to 25 km/h it
    rarely binds (only while still fast at a corner's entry) and the sweep's
    errors there were unchanged.
    """

    if time_s <= 0.0:
        return lookahead_m
    return max(lookahead_m, time_s * max(speed_mps, 0.0))


def pure_pursuit_steering(target_x_body: float, target_y_body: float, wheelbase_m: float) -> float:
    """delta = atan2(2 L y, x^2 + y^2): finite even when the target collapses."""

    d2 = target_x_body * target_x_body + target_y_body * target_y_body
    if d2 <= 1e-9:
        return 0.0
    return math.atan2(2.0 * wheelbase_m * target_y_body, d2)


def rate_limited(previous: float, desired: float, max_rate: float, dt_s: float) -> float:
    if max_rate <= 0.0 or dt_s <= 0.0:
        return desired
    step = max_rate * dt_s
    return max(previous - step, min(previous + step, desired))
