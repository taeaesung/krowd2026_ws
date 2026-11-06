"""Small deterministic Ackermann bicycle model used by the ROS simulator."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BicycleState:
    x: float
    y: float
    yaw: float
    speed_mps: float
    steering_rad: float


def approach(current: float, target: float, maximum_step: float) -> float:
    if maximum_step < 0.0:
        raise ValueError("maximum_step must be non-negative")
    return current + max(-maximum_step, min(maximum_step, target - current))


def step_bicycle(
    state: BicycleState,
    target_speed_mps: float,
    target_steering_rad: float,
    dt: float,
    wheelbase_m: float,
    max_accel_mps2: float,
    max_decel_mps2: float,
    max_steering_rate_rad_s: float,
) -> tuple[BicycleState, float]:
    """Advance a rear-axle bicycle model and return state plus yaw rate."""

    if dt <= 0.0:
        return state, 0.0
    if wheelbase_m <= 0.0:
        raise ValueError("wheelbase_m must be positive")

    speed_limit = max_accel_mps2 if target_speed_mps >= state.speed_mps else max_decel_mps2
    speed = approach(state.speed_mps, max(target_speed_mps, 0.0), speed_limit * dt)
    steering = approach(
        state.steering_rad,
        target_steering_rad,
        max_steering_rate_rad_s * dt,
    )
    yaw_rate = speed * math.tan(steering) / wheelbase_m
    midpoint_yaw = state.yaw + 0.5 * yaw_rate * dt
    return (
        BicycleState(
            x=state.x + speed * math.cos(midpoint_yaw) * dt,
            y=state.y + speed * math.sin(midpoint_yaw) * dt,
            yaw=math.atan2(
                math.sin(state.yaw + yaw_rate * dt),
                math.cos(state.yaw + yaw_rate * dt),
            ),
            speed_mps=speed,
            steering_rad=steering,
        ),
        yaw_rate,
    )


class DelayedCommand:
    """A command that takes effect `delay_s` after it was sent.

    Models the steering actuator's measured dead time (0.24 s from /target_angle
    to /encoder, stanley_tuning_0922_235531): the value returned is the newest
    command at least delay_s old, `initial` before any is that old. delay_s <= 0
    passes every command straight through. Plain numbers, no ROS.
    """

    def __init__(self, delay_s: float, initial: float = 0.0):
        self.delay_s = max(float(delay_s), 0.0)
        self._queue: list[tuple[float, float]] = []
        self._current = float(initial)

    def push(self, now_s: float, value: float) -> None:
        if self.delay_s <= 0.0:
            self._current = float(value)
            return
        self._queue.append((now_s, float(value)))

    def value(self, now_s: float) -> float:
        while self._queue and now_s - self._queue[0][0] >= self.delay_s:
            self._current = self._queue.pop(0)[1]
        return self._current


def achieved_speed(target_mps: float, deficit_mps: float, full_deficit_above_mps: float = 1.4) -> float:
    """The speed the drive MCU actually holds for a target: `deficit_mps` short
    of it (1.7 km/h on the 2026-09-22 lap: 8.1 targeted, 6.5 driven), scaled
    down linearly below `full_deficit_above_mps` so a small target still moves
    the vehicle and zero stays zero."""

    if target_mps <= 0.0 or deficit_mps <= 0.0:
        return max(target_mps, 0.0)
    scale = min(1.0, target_mps / full_deficit_above_mps) if full_deficit_above_mps > 0 else 1.0
    return max(target_mps - deficit_mps * scale, 0.0)
