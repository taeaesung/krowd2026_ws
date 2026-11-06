"""Pure Pursuit + Stanley blended steering WITH curvature feedforward (pure functions).

    kappa = route curvature at the base_link (rear-axle) projection
    w     = stanley_weight(kappa): 0 on a straight (R >= straight_radius_m),
            max_stanley_weight in a tight bend (R <= curve_radius_m), linear in
            curvature between
    delta_pp      = atan2(2 L y, x^2 + y^2) toward the lookahead point
    delta_stanley = feedforward_gain * atan(L kappa)
                    + heading_gain * heading_err + atan(k e / (k_soft + v))
                    (heading_err and e measured at the FRONT axle)
    delta = (1 - w) delta_pp + w delta_stanley + yaw-rate damping

The difference from krowd_pp_stanley_control is the first Stanley term. Pure
Pursuit on a constant-radius arc returns exactly atan(L kappa) - the geometry a
corner needs - so without that term the blend removes the fraction w of the
required steering as soon as the vehicle is on the correct line, and Stanley
can only recover it by letting an error build up (on R 4.9 m that is metres of
offset: the 2026-09-22 wide exits). With it, both laws carry the same geometric
term and the weight only chooses HOW the residual error is corrected, so the
Stanley share may be raised above 0.5 safely. feedforward_gain 0.0 reproduces
the sibling package exactly (a test pins that).

Everything here is numbers in, numbers out; the node supplies the two route
projections and the lookahead target.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .stanley import cross_track_steering, yaw_rate_damping
from .geometry import wrap_angle


@dataclass(frozen=True)
class BlendParams:
    wheelbase_m: float
    straight_radius_m: float = 50.0
    curve_radius_m: float = 5.0
    max_stanley_weight: float = 0.5
    heading_gain: float = 0.6
    cross_track_gain: float = 0.3
    softening_mps: float = 3.0  # 10.8 km/h
    heading_window_m: float = 4.0
    yaw_rate_damping_gain_s: float = 0.15
    feedforward_gain: float = 1.0  # 0 = the no-feedforward sibling, 1 = full atan(L kappa)
    # Cap on the damping's loop gain gain * v / L (see damping_gain); 0 = no cap.
    yaw_rate_damping_max_loop_gain: float = 0.0

    def validate(self) -> None:
        if self.wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be positive")
        if not 0.0 <= self.max_stanley_weight <= 1.0:
            raise ValueError(f"max_stanley_weight must lie in [0, 1]: {self.max_stanley_weight}")
        if self.straight_radius_m <= 0.0 or self.curve_radius_m <= 0.0:
            raise ValueError("blend radii must be positive")
        if self.curve_radius_m >= self.straight_radius_m:
            raise ValueError(
                f"curve_radius_m {self.curve_radius_m} must be smaller than straight_radius_m {self.straight_radius_m}"
            )
        if self.softening_mps <= 0.0:
            raise ValueError("cross_track softening speed must be positive")
        if self.heading_gain < 0.0 or self.cross_track_gain < 0.0 or self.yaw_rate_damping_gain_s < 0.0:
            raise ValueError("gains must not be negative")
        if self.heading_window_m < 0.0:
            raise ValueError("heading_window_m must not be negative")
        if not 0.0 <= self.feedforward_gain <= 1.5:
            raise ValueError(f"feedforward_gain must lie in [0, 1.5]: {self.feedforward_gain}")
        if self.yaw_rate_damping_max_loop_gain < 0.0:
            raise ValueError("yaw_rate_damping_max_loop_gain must not be negative")


def damping_gain(p: BlendParams, speed_mps: float) -> float:
    """The yaw-rate damping gain to use at this speed.

    The measured yaw rate is v tan(delta) / L, so the damping term feeds the
    steering back on itself with a loop gain of about gain * v / L. At 0.15 s
    and L 1.34 m that reaches 1 at 8.9 m/s (32 km/h): above it the term drives
    the steering into oscillation - 35 deg/s of chatter at 40 km/h in the
    2026-10-02 closed-loop sweep, with or without the steering dead time.
    The gain is therefore lowered above the speed where gain * v / L reaches
    yaw_rate_damping_max_loop_gain; 0.78 is what 0.15 s already has at
    25 km/h, so nothing changes up to there.
    """

    gain = p.yaw_rate_damping_gain_s
    if p.yaw_rate_damping_max_loop_gain > 0.0 and speed_mps > 0.0:
        gain = min(gain, p.yaw_rate_damping_max_loop_gain * p.wheelbase_m / speed_mps)
    return gain


def stanley_weight(
    curvature_per_m: float, straight_radius_m: float, curve_radius_m: float, max_weight: float
) -> float:
    """0 at |R| >= straight_radius_m, max_weight at |R| <= curve_radius_m, linear in curvature."""

    if max_weight <= 0.0 or straight_radius_m <= 0.0 or curve_radius_m <= 0.0:
        return 0.0
    straight_k = 1.0 / straight_radius_m
    curve_k = 1.0 / curve_radius_m
    if curve_k <= straight_k:
        return 0.0
    weight = (abs(curvature_per_m) - straight_k) / (curve_k - straight_k)
    return min(1.0, max(0.0, weight)) * max_weight


def blend(delta_pp: float, delta_stanley: float, weight: float) -> float:
    weight = min(1.0, max(0.0, weight))
    return (1.0 - weight) * delta_pp + weight * delta_stanley


def geometric_steering(wheelbase_m: float, curvature_per_m: float) -> float:
    """atan(L kappa): the road-wheel angle that drives a bicycle along curvature kappa.

    Pure Pursuit on a constant-radius arc returns exactly this value; it is
    the feedforward term of the Stanley law here.
    """

    return math.atan(wheelbase_m * curvature_per_m)


@dataclass(frozen=True)
class SteeringBreakdown:
    """Every term of one steering computation, for the debug topic and tests."""

    delta: float  # blended + damping, BEFORE the limit clamp and slew
    delta_pp: float
    delta_stanley: float
    weight: float
    curvature: float
    feedforward: float  # feedforward_gain * atan(L kappa)
    heading_term: float
    cross_track_term: float
    yaw_damping: float
    heading_error: float  # rad, route yaw - vehicle yaw at the front-axle projection
    cross_track_error: float  # m at the front axle, +left of the route


def compute_steering(
    delta_pp: float,
    heading_error_rad: float,
    cross_track_m: float,
    speed_mps: float,
    curvature_per_m: float,
    measured_yaw_rate_rad_s: float,
    p: BlendParams,
) -> SteeringBreakdown:
    heading_error = wrap_angle(heading_error_rad)
    feedforward = p.feedforward_gain * geometric_steering(p.wheelbase_m, curvature_per_m)
    heading_term = p.heading_gain * heading_error
    cte_term = cross_track_steering(cross_track_m, speed_mps, p.cross_track_gain, p.softening_mps)
    delta_stanley = feedforward + heading_term + cte_term
    weight = stanley_weight(curvature_per_m, p.straight_radius_m, p.curve_radius_m, p.max_stanley_weight)
    damping = yaw_rate_damping(speed_mps, curvature_per_m, measured_yaw_rate_rad_s, damping_gain(p, speed_mps))
    delta = blend(delta_pp, delta_stanley, weight) + damping
    return SteeringBreakdown(
        delta=delta,
        delta_pp=delta_pp,
        delta_stanley=delta_stanley,
        weight=weight,
        curvature=curvature_per_m,
        feedforward=feedforward,
        heading_term=heading_term,
        cross_track_term=cte_term,
        yaw_damping=damping,
        heading_error=heading_error,
        cross_track_error=cross_track_m,
    )
