"""Brake to a full stop once on entering an obstacle-avoidance zone (2026-10-10).

The user's rule: drive at speed up to the zone, then publish brake=true the
moment the vehicle enters it, keep it until the vehicle has STOOD STILL for
stop_hold_s, then go - no anticipatory deceleration. (The first version braked
for a fixed 0.5 s: entered at 20.8 km/h on the 2026-10-10 19:47 run it only
slowed the car; a stop from there needs ~1.5 s even at 3.95 m/s^2.) max_brake_s
ends a brake that never sees a standstill (no speed input, stuck vehicle). The zone is the map's
avoidance_zones.yaml, published as a Bool by localization_monitor_node
(/avoidance_space: true inside a zone).

Rising edge only, and only once per zone: the flag flickers at a zone's
boundary (localization jitter - true for 0.2 s then false was seen on
2026-09-28), so after a trigger it re-arms only once the vehicle has been
OUTSIDE for rearm_outside_s. A message older than stale_s counts as outside.

Where the brake starts (2026-10-10 later): the controller reads the map's
avoidance_zones.yaml itself (load_zone_ranges) and judges on its own monotonic
route position. From lead_waypoints before a zone's first waypoint it brakes
as soon as the remaining distance to the zone start is within the stopping
distance (stop_needed): the target is a stop AT the zone start - a slow
vehicle is not stopped early, a fast one may stop inside the zone (accepted by
the user, 2026-10-10). Without a readable zone file it falls back to the
/avoidance_space Bool (brake at the zone itself).

Plain numbers, no ROS, like start_ramp.py.
"""

import os

import yaml


def load_zone_ranges(path: str) -> list[tuple[int, int]]:
    """Inclusive (first, last) waypoint ranges of an avoidance_zones.yaml
    (localization_monitor_node.ros__parameters.avoidance_zones: ["a-b", ...])."""

    with open(os.path.expanduser(path), encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    params = (data.get("localization_monitor_node") or {}).get("ros__parameters") or {}
    raw = params.get("avoidance_zones")
    if raw is None:
        raise ValueError(f"{path}: no localization_monitor_node.ros__parameters.avoidance_zones")
    ranges = []
    for entry in raw:
        first, sep, last = str(entry).strip().partition("-")
        try:
            a, b = int(first), int(last if sep else first)
        except ValueError as error:
            raise ValueError(f"{path}: bad zone '{entry}' (want 'first-last')") from error
        if a < 0 or b < a:
            raise ValueError(f"{path}: bad zone '{entry}'")
        ranges.append((a, b))
    return ranges


def in_brake_region(ranges, waypoint: float, lead_waypoints: float) -> bool:
    """waypoint (fractional route index) inside [first - lead, last] of a zone."""

    return any(a - lead_waypoints <= waypoint <= b for a, b in ranges)


def in_lead_region(ranges, waypoint: float, lead_waypoints: float) -> bool:
    """waypoint inside [first - lead, first) of a zone: before the zone proper."""

    return any(a - lead_waypoints <= waypoint < a for a, b in ranges)


def zone_at(ranges, waypoint: float, lead_waypoints: float) -> tuple[int, int] | None:
    """The zone whose brake region [first - lead, last] holds waypoint, else None."""

    for a, b in ranges:
        if a - lead_waypoints <= waypoint <= b:
            return a, b
    return None


def stop_needed(distance_to_zone_m: float, stop_distance_m: float) -> bool:
    """Brake now? Yes once the zone start is no farther than the stopping
    distance (so a slow vehicle drives on to the zone and stops there, a fast
    one stops inside it), and always once the zone has been reached."""

    return distance_to_zone_m <= max(stop_distance_m, 0.0)


class ZoneEntryBrake:
    def __init__(self, stop_hold_s: float, rearm_outside_s: float, stale_s: float, max_brake_s: float = 5.0):
        if stop_hold_s < 0.0 or rearm_outside_s < 0.0 or stale_s <= 0.0 or max_brake_s <= 0.0:
            raise ValueError(
                f"zone brake needs stop_hold_s >= 0, rearm_outside_s >= 0, stale_s > 0, max_brake_s > 0: "
                f"{stop_hold_s}, {rearm_outside_s}, {stale_s}, {max_brake_s}"
            )
        self.stop_hold_ns = int(stop_hold_s * 1e9)
        self.max_brake_ns = int(max_brake_s * 1e9)
        self.rearm_ns = int(rearm_outside_s * 1e9)
        self.stale_ns = int(stale_s * 1e9)
        self.inside_msg: bool | None = None
        self.rx_ns: int | None = None
        self.armed = True
        self.started_ns: int | None = None      # brake running since (None = not braking)
        self._still_since_ns: int | None = None
        self._outside_since_ns: int | None = None
        self.triggers = 0
        self.timed_out = 0                       # brakes ended by max_brake_s, not by a standstill

    def observe(self, inside: bool, rx_ns: int) -> None:
        self.inside_msg = bool(inside)
        self.rx_ns = rx_ns

    def inside(self, now_ns: int) -> bool:
        return bool(
            self.inside_msg
            and self.rx_ns is not None
            and 0 <= now_ns - self.rx_ns <= self.stale_ns
        )

    def update(self, now_ns: int, standing_still: bool, can_trigger: bool = True) -> bool:
        """True while the entry brake is to be held.

        standing_still: the vehicle's measured speed is at a standstill now.
        can_trigger: False keeps a new brake from starting (the vehicle is on
        the avoidance splice, where the route index is not a waypoint index).
        """

        if self.inside(now_ns):
            self._outside_since_ns = None
            if self.armed and can_trigger:
                self.armed = False
                self.started_ns = now_ns
                self._still_since_ns = None
                self.triggers += 1
        else:
            if self._outside_since_ns is None:
                self._outside_since_ns = now_ns
            if not self.armed and now_ns - self._outside_since_ns >= self.rearm_ns:
                self.armed = True
        if self.started_ns is None:
            return False
        if standing_still:
            if self._still_since_ns is None:
                self._still_since_ns = now_ns
            if now_ns - self._still_since_ns >= self.stop_hold_ns:
                self.started_ns = None
                return False
        else:
            self._still_since_ns = None
        if now_ns - self.started_ns >= self.max_brake_ns:
            self.started_ns = None
            self.timed_out += 1
            return False
        return True

    def reset(self) -> None:
        """Tracking (re)started: forget a running brake; re-arm."""

        self.started_ns = None
        self._still_since_ns = None
        self.armed = True
        self._outside_since_ns = None
