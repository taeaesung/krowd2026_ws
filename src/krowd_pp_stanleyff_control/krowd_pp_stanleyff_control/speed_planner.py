"""Comfort speed shaping on top of the safety envelope (pure functions).

The braking model (braking.py) is the SAFETY limit: it decides when the brake
must be engaged and the largest speed from which the vehicle can still stop.
These helpers only shape the approach below that limit so the vehicle slows
down early and smoothly instead of holding the cap and braking late:

  comfort_speed(d, v_target, a)   speed that reaches v_target after d metres at
                                  a constant deceleration a
  anticipatory_cap(s, [(s_b, v_b)], a)
                                  lowest of comfort_speed toward every upcoming
                                  cap change (mission approach zone, slower
                                  section, obstacle zone) ahead of the vehicle
  profile_drops(s, speed)         where the route speed profile steps down
  coast_aim_points(...)           where each corner's coast-down must end: just
                                  before the corner proper (radius <= tight)
  coast_step_cap(...)             those drops commanded from the point where
                                  COASTING from the vehicle's actual speed just
                                  reaches the corner speed: one step, or a ramp
                                  along the coast curve (ramp_below_mps)
  align_hold_cap(...)             no acceleration while the vehicle's heading
                                  is off the route's by more than a limit
"""

from __future__ import annotations

import math


def comfort_speed(distance_m: float, target_mps: float, decel_mps2: float) -> float:
    if decel_mps2 <= 0.0:
        return math.inf
    d = max(distance_m, 0.0)
    return math.sqrt(max(target_mps, 0.0) ** 2 + 2.0 * decel_mps2 * d)


def anticipatory_cap(s: float, boundaries, decel_mps2: float) -> float:
    cap = math.inf
    for s_b, v_b in boundaries:
        if s_b < s:
            continue
        cap = min(cap, comfort_speed(s_b - s, v_b, decel_mps2))
    return cap


def step_cap(ramp_mps: float, floor_mps: float, other_mps: float) -> float:
    """A comfort ramp as ONE step: once the ramp would bind (drops below the
    other caps), command its floor at once instead of creeping down to it.
    The drive decelerates best when the target drops in one go (2026-10-01)."""

    return floor_mps if ramp_mps < other_mps else math.inf


def anticipatory_step(s: float, boundaries, decel_mps2: float, other_mps: float) -> float:
    """anticipatory_cap as a step: the next zone's cap from the point where the
    comfort ramp toward it would have started binding."""

    cap = math.inf
    for s_b, v_b in boundaries:
        if s_b < s:
            continue
        if comfort_speed(s_b - s, v_b, decel_mps2) < other_mps:
            cap = min(cap, v_b)
    return cap


def profile_drops(s_m, speed_mps, min_drop_mps: float = 0.01) -> list[tuple[float, float]]:
    """(s_b, v_b) for every step down of a route speed profile.

    RouteModel.speed_at_s takes a drop between two waypoints whole at the
    first one, so the drop begins at s[i-1] and commands speed[i]. The last
    waypoint (0 when the route ends in a stop) is left out: the route-end stop
    belongs to the braking model, not to coasting.
    """

    drops = []
    for i in range(1, len(speed_mps) - 1):
        if float(speed_mps[i]) < float(speed_mps[i - 1]) - min_drop_mps:
            drops.append((float(s_m[i - 1]), float(speed_mps[i])))
    return drops


def coast_aim_points(
    drops: list[tuple[float, float]],
    s_m,
    curvature,
    tight_radius_m: float,
    before_m: float,
    max_shift_m: float = 20.0,
) -> list[tuple[float, float, float]]:
    """(s_b, v_b, s_aim): the coast-down ends at s_aim instead of the drop.

    The profile steps down curve_speed_lead_m before the corner RUN, which
    begins where the radius falls under curve_radius_m (30 m) - 5-9 m before
    the corner proper on 20261003_220932. On the 2026-10-03 23:00 bag the
    steering passed 10 deg only 1-2 m before the first point of radius
    <= 10 m (5 deg at 9-11 m, at 7 km/h), so the corner speed is needed there,
    not at the drop. s_aim = before_m ahead of the first point at or after the
    drop with radius <= tight_radius_m, never before the drop; none within
    max_shift_m (a gentle bend's drop) -> the drop. tight_radius_m <= 0 -> the
    drop (the old behaviour).
    """

    out = []
    for s_b, v_b in drops:
        aim = s_b
        if tight_radius_m > 0.0:
            k_tight = 1.0 / tight_radius_m
            for i in range(len(s_m)):
                si = float(s_m[i])
                if si < s_b:
                    continue
                if si - s_b > max_shift_m:
                    break
                if abs(float(curvature[i])) >= k_tight:
                    aim = max(s_b, si - before_m)
                    break
        out.append((s_b, v_b, aim))
    return out


def coast_step_cap(
    s: float,
    speed_mps: float,
    drops: list[tuple],
    decel_mps2: float,
    latched: set,
    horizon_m: float = 150.0,
    ramp_below_mps: float | None = None,
) -> tuple[float, set]:
    """Corner speed, commanded early enough to coast down to it (2026-10-03).

    The route profile drops to a corner's speed only curve_speed_lead_m before
    the corner, and the drive only coasts (no brake for corners, ~0.5 m/s^2
    measured on 2026-10-03): from 15 km/h it needs ~14 m to reach 7 km/h, so
    the vehicle entered the corners at 12-14 km/h. Here the drop's speed v_b is
    commanded at once from the point where coasting from the vehicle's ACTUAL
    speed at decel_mps2 just reaches v_b at the drop - the actual speed, not
    the profile's, because the profile ramps up faster (max_accel_mps2) than
    the drive accelerates and the vehicle often never reaches it.

    Once a drop has triggered it stays latched until the vehicle passes it:
    re-testing every cycle would raise the target again whenever the vehicle
    coasts a little faster than decel_mps2 and chatter the drive.

    ramp_below_mps None: v_b is commanded as one STEP from the trigger point
    (the 7 km/h shown on the straight 40 m before a corner, 2026-10-03 23:00).
    A number: the command FOLLOWS the coast curve sqrt(v_b^2 + 2 a d) down to
    the drop, ramp_below_mps under it (never under v_b) - the target stays a
    little below the coasting vehicle so the drive keeps its power off, and
    reads what the vehicle is about to do. The curve only falls as the vehicle
    advances, so the latched command never rises.

    drops are (s_b, v_b) or (s_b, v_b, s_aim) (coast_aim_points): the
    coast-down is planned to reach v_b at s_aim; s_b stays the drop's identity
    (latch, horizon).

    Returns (cap, latched drops still ahead); cap is inf when nothing binds.
    decel_mps2 <= 0 disables it.
    """

    cap = math.inf
    live = set()
    if decel_mps2 <= 0.0:
        return cap, live
    for drop in drops:
        s_b, v_b = drop[0], drop[1]
        s_aim = drop[2] if len(drop) > 2 else s_b
        if not (s < s_b <= s + horizon_m):
            continue
        curve = comfort_speed(s_aim - s, v_b, decel_mps2)
        if s_b in latched or curve < speed_mps:
            live.add(s_b)
            cap = min(cap, v_b if ramp_below_mps is None else max(v_b, curve - ramp_below_mps))
    return cap, live


def align_hold_cap(
    heading_error_rad: float,
    limit_rad: float,
    speed_mps: float,
    target_mps: float,
    floor_mps: float,
    hold_mps: float | None,
    steering_hold: bool = False,
) -> tuple[float, float | None]:
    """No acceleration until the vehicle is aligned with the route (2026-10-03).

    steering_hold (2026-10-10) is the second, ANDed condition: the measured
    road-wheel angle (/encoder) is above its limit or not fresh. Either one
    holds the speed the same way; both must clear to release it.

    On the 2026-10-03 drive the controller asked for 12-16 km/h out of corners
    the vehicle had run wide of, its heading 34-112 deg off the route; normal
    tracking stays within ~5 deg (1-5 deg on the vehicle, 2.2 deg in the sim).

    While |heading_error| > limit the speed is HELD: at the lower of the speed
    and the other caps when the hold began (the corner speed in a corner), never
    below floor_mps (a vehicle standing still may still crawl), and lowered with
    the target but never raised. Aligned again -> released.

    Returns (cap, hold); cap is inf and hold None when not holding.
    limit_rad <= 0 disables it.
    """

    heading_off = limit_rad > 0.0 and abs(heading_error_rad) > limit_rad
    if not (heading_off or steering_hold):
        return math.inf, None
    if hold_mps is None:
        hold_mps = max(min(speed_mps, target_mps), floor_mps)
    else:
        hold_mps = max(min(hold_mps, target_mps), floor_mps)
    return hold_mps, hold_mps
