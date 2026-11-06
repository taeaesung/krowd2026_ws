"""Turn a route's curvature into a per-waypoint target speed.

This is the half of path tracking that Pure Pursuit does not do. MPC solved
steering and speed together under one set of constraints; a geometric steering
law solves neither, so the longitudinal plan has to be built explicitly - and
built ahead of time, over the whole route, because braking for a corner has to
start before the corner is anywhere near the lookahead point.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpeedLimits:
    """What the vehicle may do, longitudinally and laterally."""

    max_speed_mps: float
    # The cornering budget. v = sqrt(a_lat / k) is the whole speed profile in
    # one line: everything else here only limits how fast it may be reached or
    # left. Set it from measured grip, not from comfort - on a 220 kg vehicle
    # with 155-section tyres the real ceiling arrives well before 1 g.
    max_lateral_accel_mps2: float
    max_accel_mps2: float
    max_decel_mps2: float
    # Floor under the CURVATURE ceiling only, so a noisy tight corner cannot
    # profile the vehicle down to a crawl. The braking ramp into a stop goes
    # through it - a goal speed of zero is still zero.
    min_speed_mps: float = 0.0
    # Speed ceiling for curves: applied wherever the turning radius is below
    # curve_radius_m. The grip formula alone lets a gentle bend run nearly as
    # fast as a straight; this bounds what "a curve" may be driven at,
    # however gentle. Both 0 disables it.
    curve_max_speed_mps: float = 0.0
    curve_radius_m: float = 0.0
    # Reach each corner's speed this far BEFORE the corner: every waypoint's
    # ceiling is pulled down to the lowest ceiling within curve_lead_m ahead
    # of it, before the braking ramp is laid in front of that. Gives the
    # drivetrain, which only coasts down to a lower target, a margin so the
    # vehicle is actually at corner speed at the corner entry. 0 disables.
    curve_lead_m: float = 0.0
    # Leave each corner at straight speed: within curve_exit_lead_m of a
    # curve's end, but never before its tightest point, the ceiling is
    # released to max_speed_mps so the acceleration ramp starts inside the
    # corner. 0 disables.
    curve_exit_lead_m: float = 0.0

    @property
    def curve_cap_enabled(self) -> bool:
        return self.curve_max_speed_mps > 0.0 and self.curve_radius_m > 0.0

    def validate(self) -> None:
        if self.max_speed_mps <= 0.0:
            raise ValueError(f"max_speed_mps must be positive: {self.max_speed_mps}")
        if self.max_lateral_accel_mps2 <= 0.0:
            raise ValueError(
                f"max_lateral_accel_mps2 must be positive: {self.max_lateral_accel_mps2}"
            )
        if self.max_accel_mps2 <= 0.0:
            raise ValueError(f"max_accel_mps2 must be positive: {self.max_accel_mps2}")
        if self.max_decel_mps2 <= 0.0:
            raise ValueError(f"max_decel_mps2 must be positive: {self.max_decel_mps2}")
        if not 0.0 <= self.min_speed_mps <= self.max_speed_mps:
            raise ValueError(
                f"min_speed_mps {self.min_speed_mps} must lie in "
                f"[0, {self.max_speed_mps}]"
            )
        if self.curve_max_speed_mps < 0.0 or self.curve_radius_m < 0.0:
            raise ValueError(
                "curve_max_speed_mps and curve_radius_m must not be negative: "
                f"{self.curve_max_speed_mps}, {self.curve_radius_m}"
            )
        if (self.curve_max_speed_mps > 0.0) != (self.curve_radius_m > 0.0):
            raise ValueError(
                "set both curve_max_speed_mps and curve_radius_m, or neither: "
                f"got {self.curve_max_speed_mps} and {self.curve_radius_m}"
            )
        if self.curve_lead_m < 0.0:
            raise ValueError(f"curve_lead_m must not be negative: {self.curve_lead_m}")
        if self.curve_exit_lead_m < 0.0:
            raise ValueError(
                f"curve_exit_lead_m must not be negative: {self.curve_exit_lead_m}"
            )
        if self.curve_cap_enabled and self.curve_max_speed_mps < self.min_speed_mps:
            raise ValueError(
                f"curve_max_speed_mps {self.curve_max_speed_mps} is below "
                f"min_speed_mps {self.min_speed_mps}: the floor would override "
                "the curve cap"
            )


def curvature_speed_ceiling(
    curvatures: list[float], limits: SpeedLimits
) -> list[float]:
    """The fastest each waypoint may be taken: lateral grip, then the curve cap."""

    ceiling = []
    for curvature in curvatures:
        magnitude = abs(curvature)
        if magnitude < 1e-6:
            ceiling.append(limits.max_speed_mps)
            continue
        speed = min(
            limits.max_speed_mps,
            math.sqrt(limits.max_lateral_accel_mps2 / magnitude),
        )
        if limits.curve_cap_enabled and magnitude > 1.0 / limits.curve_radius_m:
            speed = min(speed, limits.curve_max_speed_mps)
        ceiling.append(speed)
    return ceiling


def lead_corner_speeds(
    ceilings: list[float], lengths: list[float], lead_m: float
) -> list[float]:
    """Each waypoint's ceiling lowered to the lowest ceiling within lead_m ahead.

    A corner's speed thereby applies from lead_m before the corner, and the
    backward braking pass then starts its ramp in front of THAT point.
    """

    if lead_m <= 0.0:
        return list(ceilings)
    led = list(ceilings)
    for index in range(len(ceilings)):
        travelled = 0.0
        ahead = index
        while ahead + 1 < len(ceilings) and travelled < lead_m:
            travelled += lengths[ahead]
            ahead += 1
            led[index] = min(led[index], ceilings[ahead])
    return led


def release_corner_exits(
    ceilings: list[float],
    lengths: list[float],
    exit_lead_m: float,
    max_speed: float,
    curvatures: list[float],
) -> list[float]:
    """Raise the last exit_lead_m of every curve to max_speed.

    A curve is a run of waypoints below max_speed. Its last exit_lead_m - but
    nothing at or before its tightest waypoint - is released, so the forward
    acceleration pass ramps up from inside the corner and the vehicle exits at
    straight speed. A curve running into the route's end is left alone; the
    stop ramp owns that.

    "Tightest" is by CURVATURE, not by the ceiling: with min_speed equal to
    the curve cap every waypoint of a curve has the same ceiling, and picking
    the slowest would pick the curve's first waypoint and accelerate through
    the whole of a short corner.
    """

    if exit_lead_m <= 0.0:
        return list(ceilings)
    count = len(ceilings)
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    released = list(ceilings)
    index = 0
    while index < count:
        if ceilings[index] >= max_speed:
            index += 1
            continue
        start = index
        while index < count and ceilings[index] < max_speed:
            index += 1
        end = index  # first waypoint after the curve
        if end >= count:
            break
        tightest = max(range(start, end), key=lambda i: abs(curvatures[i]))
        for i in range(tightest + 1, end):
            if cumulative[end] - cumulative[i] <= exit_lead_m:
                released[i] = max_speed
    return released


def speed_profile(
    lengths: list[float],
    curvatures: list[float],
    limits: SpeedLimits,
    stop_at_end: bool,
) -> list[float]:
    """Per-waypoint target speed, feasible in both directions.

    Three passes. The curvature ceiling says how fast each corner may be taken;
    a backward pass makes sure the vehicle can still brake down to each of those
    ceilings in the distance available before it; a forward pass makes sure it
    could have accelerated to the speed being asked for.

    Order matters. Without the backward pass the profile is "correct" at every
    single waypoint and undriveable between them - it would ask for corner speed
    at the corner entry with no room to have shed the straight's speed first.
    """

    limits.validate()
    count = len(curvatures)
    if count == 0:
        return []
    if len(lengths) != count - 1:
        raise ValueError(
            f"expected {count - 1} segment lengths for {count} waypoints, "
            f"got {len(lengths)}"
        )

    speeds = curvature_speed_ceiling(curvatures, limits)
    speeds = [max(speed, limits.min_speed_mps) for speed in speeds]
    speeds = release_corner_exits(
        speeds, lengths, limits.curve_exit_lead_m, limits.max_speed_mps, curvatures
    )
    speeds = lead_corner_speeds(speeds, lengths, limits.curve_lead_m)

    # A goal the vehicle must stop at is a zero-speed constraint like any other,
    # and the backward pass below turns it into the braking ramp that reaches it.
    if stop_at_end:
        speeds[-1] = 0.0

    for index in range(count - 2, -1, -1):
        reachable = math.sqrt(
            speeds[index + 1] ** 2 + 2.0 * limits.max_decel_mps2 * lengths[index]
        )
        speeds[index] = min(speeds[index], reachable)

    for index in range(1, count):
        reachable = math.sqrt(
            speeds[index - 1] ** 2 + 2.0 * limits.max_accel_mps2 * lengths[index - 1]
        )
        speeds[index] = min(speeds[index], reachable)

    return speeds
