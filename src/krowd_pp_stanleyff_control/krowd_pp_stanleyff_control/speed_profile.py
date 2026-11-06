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
    # Graded corner speed (2026-10-03): curve_max_speed_mps at a radius of
    # tight_curve_radius_m or less, max_speed_mps at curve_radius_m, linear in
    # the radius between - a tight corner steps down a lot, a gentle one a
    # little. 0 = the flat cap (curve_max_speed_mps below curve_radius_m).
    tight_curve_radius_m: float = 0.0
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
        if self.curve_cap_enabled and self.curve_max_speed_mps > self.max_speed_mps:
            raise ValueError(
                f"curve_max_speed_mps {self.curve_max_speed_mps} is above "
                f"max_speed_mps {self.max_speed_mps}"
            )
        if self.tight_curve_radius_m < 0.0:
            raise ValueError(
                f"tight_curve_radius_m must not be negative: {self.tight_curve_radius_m}"
            )
        if (
            self.curve_cap_enabled
            and self.tight_curve_radius_m > 0.0
            and self.tight_curve_radius_m >= self.curve_radius_m
        ):
            raise ValueError(
                f"tight_curve_radius_m {self.tight_curve_radius_m} must be below "
                f"curve_radius_m {self.curve_radius_m}"
            )


def grip_speed(curvature: float, limits: SpeedLimits) -> float:
    """The lateral-grip ceiling sqrt(a_lat / |kappa|), at most max_speed."""

    magnitude = abs(curvature)
    if magnitude < 1e-6:
        return limits.max_speed_mps
    return min(limits.max_speed_mps, math.sqrt(limits.max_lateral_accel_mps2 / magnitude))


def corner_speed(curvature: float, limits: SpeedLimits) -> float:
    """The curve cap at this curvature: max_speed outside a curve, graded in
    the radius between tight_curve_radius_m and curve_radius_m when set."""

    magnitude = abs(curvature)
    if not limits.curve_cap_enabled or magnitude <= 1.0 / limits.curve_radius_m:
        return limits.max_speed_mps
    radius = 1.0 / magnitude
    tight = limits.tight_curve_radius_m
    if tight <= 0.0 or radius <= tight:
        return limits.curve_max_speed_mps
    fraction = (radius - tight) / (limits.curve_radius_m - tight)
    return limits.curve_max_speed_mps + fraction * (limits.max_speed_mps - limits.curve_max_speed_mps)


def curvature_speed_ceiling(
    curvatures: list[float], limits: SpeedLimits
) -> list[float]:
    """The fastest each waypoint may be taken: lateral grip, then the curve cap."""

    return [min(grip_speed(k, limits), corner_speed(k, limits)) for k in curvatures]


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


def drop_short_curves(
    ceilings: list[float], lengths: list[float], max_speed: float, min_length_m: float
) -> list[float]:
    """Raise every run of below-max ceilings shorter than min_length_m back to max.

    One or two waypoints whose smoothed curvature pokes past 1/curve_radius_m on
    a straight (waypoint noise, a gentle bend hovering at the threshold) would
    otherwise drop the target to the curve speed for a metre. 0 disables.
    """

    if min_length_m <= 0.0:
        return list(ceilings)
    out = list(ceilings)
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    index = 0
    while index < len(out):
        if out[index] >= max_speed:
            index += 1
            continue
        start = index
        while index < len(out) and out[index] < max_speed:
            index += 1
        end = min(index, len(out) - 1)  # run covers [start, end)
        if cumulative[end] - cumulative[start] < min_length_m:
            for k in range(start, index):
                out[k] = max_speed
    return out


def merge_close_corners(
    ceilings: list[float], lengths: list[float], max_speed: float, gap_m: float
) -> list[float]:
    """Fill a straight gap shorter than gap_m between two curve runs with the
    lower of the speeds on either side, so the two become ONE corner.

    Two corners a few metres apart (20261003_220932: 3 m between R 6.5/5.5 m
    and R 6.3/4.5 m) were two runs: the first one's exit was released and
    ramped up 11.5 -> 19.4 km/h, then stepped back to 7 for the second. The
    drive answered the moment of acceleration with 7 -> 12 km/h in 2 m and
    only coasted back to 8-9 km/h by the second, tighter corner, which it then
    ran wide of (2026-10-03 23:00 bag). Merged, the pair is held at the corner
    speed and the exit is released only after the second corner's tightest
    point. 0 disables.
    """

    if gap_m <= 0.0:
        return list(ceilings)
    out = list(ceilings)
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    count = len(out)
    index = 0
    while index < count:
        if out[index] < max_speed:
            index += 1
            continue
        start = index  # first straight waypoint of the gap
        while index < count and out[index] >= max_speed:
            index += 1
        if start == 0 or index >= count:
            continue  # the route's own start/end is no gap between two corners
        if cumulative[index] - cumulative[start - 1] < gap_m:
            fill = min(out[start - 1], out[index])
            for k in range(start, index):
                out[k] = fill
    return out


def one_speed_per_corner(ceilings: list[float], max_speed: float) -> list[float]:
    """Every run of below-max ceilings lowered to its own minimum.

    A corner whose radius tightens towards the apex would otherwise step down
    several times on the way in; the drive follows a creeping target lazily,
    so the corner's speed - the one its tightest point needs - is commanded
    whole from its start (2026-10-03).
    """

    out = list(ceilings)
    index = 0
    while index < len(out):
        if out[index] >= max_speed:
            index += 1
            continue
        start = index
        while index < len(out) and out[index] < max_speed:
            index += 1
        low = min(out[start:index])
        for k in range(start, index):
            out[k] = low
    return out


def curvature_only_profile(
    lengths: list[float],
    curvatures: list[float],
    limits: SpeedLimits,
    stop_at_end: bool,
    min_curve_length_m: float = 0.0,
    corner_merge_gap_m: float = 0.0,
) -> list[float]:
    """Corner speed decided by curvature; dropped AT ONCE before a corner,
    raised GRADUALLY out of it (2026-10-01, user decision).

    1. the curve cap from each waypoint's own curvature (graded in the radius
       when tight_curve_radius_m is set, straight speed outside curves);
    2. curve runs shorter than min_curve_length_m are ignored (noise);
    3. the lateral-grip ceiling, AFTER the noise filter: a gentle bend that is
       too fast for max_lateral_accel_mps2 is never filtered away (with a
       higher grip limit its slow run gets shorter than the filter and used
       to vanish - 40 km/h through an R 18 m bend, 2026-10-03);
    3b. corners less than corner_merge_gap_m apart become one corner
       (merge_close_corners);
    4. each corner at one speed, its slowest point's (one step down);
    5. the corner's exit is released after its tightest point
       (curve_exit_lead_m) so the climb starts inside the corner;
    6. the corner speed is applied curve_lead_m BEFORE the corner - a step,
       there is NO backward braking pass;
    7. a forward pass limits every rise to max_accel_mps2 - the gradual exit.

    Only the last waypoint is 0 when the route ends in a stop; the stop itself
    is the controller's braking model.
    """

    limits.validate()
    count = len(curvatures)
    if count == 0:
        return []
    if len(lengths) != count - 1:
        raise ValueError(f"expected {count - 1} segment lengths for {count} waypoints, got {len(lengths)}")
    speeds = [max(corner_speed(k, limits), limits.min_speed_mps) for k in curvatures]
    speeds = drop_short_curves(speeds, lengths, limits.max_speed_mps, min_curve_length_m)
    speeds = [max(min(v, grip_speed(k, limits)), limits.min_speed_mps) for v, k in zip(speeds, curvatures)]
    speeds = merge_close_corners(speeds, lengths, limits.max_speed_mps, corner_merge_gap_m)
    speeds = one_speed_per_corner(speeds, limits.max_speed_mps)
    speeds = release_corner_exits(speeds, lengths, limits.curve_exit_lead_m, limits.max_speed_mps, curvatures)
    speeds = lead_corner_speeds(speeds, lengths, limits.curve_lead_m)
    if stop_at_end:
        speeds[-1] = 0.0
    for index in range(1, count):
        reachable = math.sqrt(speeds[index - 1] ** 2 + 2.0 * limits.max_accel_mps2 * lengths[index - 1])
        speeds[index] = min(speeds[index], reachable)
    return speeds


def step_down_decelerations(speeds: list[float], stop_floor_mps: float) -> list[float]:
    """Turn every braking ramp of a profile into one step to its final speed.

    The drive decelerates best when the target drops at once (2026-10-01,
    user decision): a target that creeps down 0.5 m/s² at a time is followed
    lazily. The backward pass still decides WHERE the ramp would have started;
    from that point the profile commands the corner speed directly. A ramp
    into the route's final stop becomes a step to `stop_floor_mps` (the drive
    cannot hold less), with only the last waypoint at 0; the controller's
    braking model owns the stop itself. Acceleration ramps are left alone.
    """

    out = list(speeds)
    count = len(out)
    index = 0
    while index < count - 1:
        if out[index + 1] >= out[index]:
            index += 1
            continue
        end = index
        while end + 1 < count and out[end + 1] < out[end]:
            end += 1
        low = out[end]
        target = max(low, stop_floor_mps) if low <= 0.0 else low
        for k in range(index, end):
            out[k] = target
        index = end
    return out
