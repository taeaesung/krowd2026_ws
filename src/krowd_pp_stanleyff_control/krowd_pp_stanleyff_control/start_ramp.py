"""A speed cap for the first metres after tracking starts or resumes.

The route's speed profile is a function of position only, so it has no ramp
from standstill: the moment tracking begins it asks for the waypoint's
cruising speed, 15 km/h on a straight. This caps the command at
sqrt(v0^2 + 2 a d) - what a vehicle accelerating at `accel_mps2` from `v0`
could be doing after `d` metres - where v0 is `start_speed_mps` or the speed
the vehicle already has, whichever is higher, so a resume at speed is not a
brake application. Plain numbers, no ROS, like stanley.py.
"""

import math


class StartRamp:
    def __init__(self, start_speed_mps: float, accel_mps2: float):
        if start_speed_mps < 0.0 or accel_mps2 < 0.0:
            raise ValueError(
                f"start ramp needs non-negative values: {start_speed_mps}, {accel_mps2}"
            )
        self.start_speed_mps = start_speed_mps
        self.accel_mps2 = accel_mps2
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.start_speed_mps > 0.0 and self.accel_mps2 > 0.0

    def reset(self) -> None:
        """Forget the current ramp; the next cap() starts a new one."""

        self._origin_speed: float | None = None
        self._last: tuple[float, float] | None = None
        self.distance_m = 0.0

    def cap(self, speed_now_mps: float, x: float, y: float) -> float:
        """The speed cap at (x, y); infinite when disabled."""

        if not self.enabled:
            return math.inf
        if self._origin_speed is None:
            self._origin_speed = max(self.start_speed_mps, speed_now_mps, 0.0)
            self.distance_m = 0.0
        else:
            self.distance_m += math.hypot(x - self._last[0], y - self._last[1])
        self._last = (x, y)
        return math.sqrt(
            self._origin_speed**2 + 2.0 * self.accel_mps2 * self.distance_m
        )
