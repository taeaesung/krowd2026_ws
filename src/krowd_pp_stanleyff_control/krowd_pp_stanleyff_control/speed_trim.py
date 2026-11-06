"""Raise the speed command while the vehicle runs slower than its target.

The drive MCU closes the speed loop itself (PI on its wheel sensors), and on
the first Stanley drive (2026-09-20) it did not reach a 5 km/h target: 108 m
took 103 s, 3.8 km/h on average, on a straight, with /target_speed constant.
Its gains are firmware; what this side can change is the number it is asked
for. So this integrates the shortfall into a trim ON TOP of the target:

    command = target + trim,   d(trim)/dt = gain * (target - measured)

An outer integrator around the MCU's loop, deliberately slow (the MCU loop must
settle inside it) and bounded:

* trim stays within [0, max_trim]: it only ever asks for MORE than the target,
  never more than max_trim more, and cannot run away when the vehicle is held
  back by something a higher command will not fix (brake dragging, a kerb).
* it integrates only while the vehicle MOVES (above min_speed), so waiting for
  the ASMS with a 5 km/h target does not wind it up.
* `active=False` FREEZES it: the trim is neither integrated nor applied and
  the bare target is returned. The controller passes that while a safety cap
  (stoppable distance, comfort stop, commit approach), the start ramp or a
  stop owns the target - a trim on top of a stopping-distance cap would defeat
  the cap, and one on top of the start ramp would defeat the ramp.
* a zero target (brake) also freezes it rather than clearing it, so a vehicle
  that has learnt its drive runs 1.7 km/h under target keeps that knowledge
  through every traffic-light stop. reset() - a tracking restart - clears it.

Plain numbers, no ROS, like start_ramp.py.
"""


class SpeedTrim:
    def __init__(self, gain_per_s: float, max_trim_mps: float, min_speed_mps: float):
        if gain_per_s < 0.0 or max_trim_mps < 0.0 or min_speed_mps < 0.0:
            raise ValueError(
                "speed trim needs non-negative values: "
                f"{gain_per_s}, {max_trim_mps}, {min_speed_mps}"
            )
        self.gain_per_s = gain_per_s
        self.max_trim_mps = max_trim_mps
        self.min_speed_mps = min_speed_mps
        self.trim_mps = 0.0

    @property
    def enabled(self) -> bool:
        return self.gain_per_s > 0.0 and self.max_trim_mps > 0.0

    def reset(self) -> None:
        self.trim_mps = 0.0

    def command(self, target_mps: float, measured_mps: float, dt_s: float, active: bool = True) -> float:
        """The speed to ask the MCU for so that the vehicle does `target_mps`."""

        target = max(target_mps, 0.0)
        if not self.enabled or target <= 0.0 or not active:
            return target
        if measured_mps >= self.min_speed_mps and dt_s > 0.0:
            self.trim_mps += self.gain_per_s * (target - measured_mps) * dt_s
            self.trim_mps = min(max(self.trim_mps, 0.0), self.max_trim_mps)
        return target + self.trim_mps
