"""Road-wheel (bicycle) angle <-> steering actuator command (CAN 0x120, rad).

The steer Teensy takes and reports an ENCODER angle in radians (0.01 rad/LSB,
+ = LEFT, mechanical ends -2.71/+2.75 rad, command limit +/-2.5 rad). The
controller reasons in the equivalent single-track (bicycle) road-wheel angle
delta, with tan(delta) / wheelbase the path curvature.

Table (2026-09-23, krowd_stanley_control vehicle.yaml): four inner rows are
the bicycle angle measured on the stanley_tuning_0922_235531 lap as
atan(wheelbase * yaw_rate / speed) at +/-1.73 and +/-1.0 rad of encoder, split
into left/right tyre angles with Ackermann; the two ends are the 2026-09-19
hand measurement. The bicycle-equivalent angle of an Ackermann pair is the
average of the two cotangents: cot(delta) = (cot(left) + cot(right)) / 2.

The encoder column is CENTRED: encoder reading minus `zero_offset_rad`, the
reading at which the wheels are straight (-0.21 rad on 2026-09-23, the median
/encoder while driving straight). The Teensy zero (/steering_center) is set per
session, so this offset is per session too: re-measure it from a straight
drive after every re-zero. With it at 0.0 the old krowd_final_control table
commanded "straight" with the wheels ~1.7 deg left.

The table is data, not code: replace the points in control.yaml when the
5-10 point re-measurement inside +/-2.5 rad is done and set
`steering.table_verified: true`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import interpolate_table


def equivalent_bicycle_angle(left_deg: float, right_deg: float) -> float:
    """Bicycle-model steering angle (rad) from left/right road-wheel angles (deg)."""

    left = math.radians(left_deg)
    right = math.radians(right_deg)
    if abs(left) < 1e-9 and abs(right) < 1e-9:
        return 0.0
    if math.copysign(1.0, left) != math.copysign(1.0, right) and abs(left) > 1e-9 and abs(right) > 1e-9:
        raise ValueError(f"left/right tyre angles disagree in sign: {left_deg}, {right_deg}")
    cots = []
    for angle in (left, right):
        if abs(angle) < 1e-9:
            continue
        cots.append(1.0 / math.tan(angle))
    mean_cot = sum(cots) / len(cots)
    return math.atan(1.0 / mean_cot)


@dataclass(frozen=True)
class SteeringCalibration:
    encoder_rad: tuple[float, ...]  # CENTRED encoder (reading - zero_offset), strictly increasing, contains 0.0
    road_wheel_rad: tuple[float, ...]  # strictly increasing, 0.0 at centred encoder 0
    command_limit_rad: float  # +/- RAW actuator command limit (team decision 2.5)
    verified: bool
    sign: float = 1.0  # +1: encoder + is left (measured). -1 only after a wiring change.
    zero_offset_rad: float = 0.0  # RAW encoder reading with the wheels straight

    @classmethod
    def from_config(cls, cfg: dict) -> "SteeringCalibration":
        enc = [float(v) for v in cfg["table_encoder_rad"]]
        left = [float(v) for v in cfg["table_left_tire_deg"]]
        right = [float(v) for v in cfg["table_right_tire_deg"]]
        if not (len(enc) == len(left) == len(right) >= 3):
            raise ValueError("steering table needs >= 3 rows of encoder/left/right")
        rows = sorted(zip(enc, left, right))
        enc = [r[0] for r in rows]
        wheel = [equivalent_bicycle_angle(r[1], r[2]) for r in rows]
        calib = cls(
            encoder_rad=tuple(enc),
            road_wheel_rad=tuple(wheel),
            command_limit_rad=float(cfg["command_limit_rad"]),
            verified=bool(cfg.get("table_verified", False)),
            sign=float(cfg.get("sign", 1.0)),
            zero_offset_rad=float(cfg.get("zero_offset_rad", 0.0) or 0.0),
        )
        calib.validate()
        return calib

    def validate(self) -> None:
        enc = np.asarray(self.encoder_rad)
        wheel = np.asarray(self.road_wheel_rad)
        if np.any(np.diff(enc) <= 0) or np.any(np.diff(wheel) <= 0):
            raise ValueError("steering table must be strictly increasing in both columns")
        if 0.0 not in self.encoder_rad:
            raise ValueError("steering table must contain the straight-ahead point (centred encoder = 0)")
        zero = self.encoder_rad.index(0.0)
        if abs(self.road_wheel_rad[zero]) > 1e-9:
            raise ValueError("road-wheel angle at centred encoder 0 must be 0")
        if self.sign not in (1.0, -1.0):
            raise ValueError("steering sign must be +1 or -1")
        if not math.isfinite(self.zero_offset_rad):
            raise ValueError("zero_offset_rad must be finite")
        raw_lo = self.encoder_rad[0] + self.zero_offset_rad
        raw_hi = self.encoder_rad[-1] + self.zero_offset_rad
        if not 0.0 < self.command_limit_rad or self.command_limit_rad > raw_hi + 1e-9 or -self.command_limit_rad < raw_lo - 1e-9:
            raise ValueError(
                "command_limit_rad must lie inside the calibrated RAW encoder range "
                f"[{raw_lo:.3f}, {raw_hi:.3f}] (table ends + zero offset)"
            )

    # ---- conversions ------------------------------------------------------
    def road_wheel_from_encoder(self, encoder_rad: float) -> float:
        """Road-wheel angle for a RAW encoder reading (or command)."""

        centred = encoder_rad - self.zero_offset_rad
        return interpolate_table(self.encoder_rad, self.road_wheel_rad, self.sign * centred)

    def encoder_from_road_wheel(self, road_wheel_rad: float) -> float:
        """RAW encoder reading (or command) that produces a road-wheel angle."""

        centred = self.sign * interpolate_table(self.road_wheel_rad, self.encoder_rad, road_wheel_rad)
        return centred + self.zero_offset_rad

    @property
    def road_wheel_limits(self) -> tuple[float, float]:
        """(min, max) road-wheel angle expressible inside +/-command_limit_rad.

        Asymmetric with a non-zero offset: on 2026-09-23 the right had 2.29 rad
        of centred travel and the left 2.71.
        """

        ends = (self.road_wheel_from_encoder(-self.command_limit_rad), self.road_wheel_from_encoder(self.command_limit_rad))
        return (float(min(ends)), float(max(ends)))

    def clamp_road_wheel(self, road_wheel_rad: float) -> float:
        lo, hi = self.road_wheel_limits
        return min(max(road_wheel_rad, lo), hi)

    def command(self, road_wheel_rad: float) -> float:
        """Actuator command (rad on /target_angle) for a road-wheel angle, clamped."""

        value = self.encoder_from_road_wheel(self.clamp_road_wheel(road_wheel_rad))
        return float(min(max(value, -self.command_limit_rad), self.command_limit_rad))

    def describe(self) -> str:
        points = ", ".join(
            f"{e:+.2f}->{math.degrees(w):+.1f}deg"
            for e, w in zip(self.encoder_rad, self.road_wheel_rad)
        )
        lo, hi = self.road_wheel_limits
        return (
            f"steering table (centred) [{points}], zero offset {self.zero_offset_rad:+.3f} rad, "
            f"command limit +/-{self.command_limit_rad:.2f} rad "
            f"-> road wheel {math.degrees(lo):+.1f}..{math.degrees(hi):+.1f} deg, "
            f"{'VERIFIED' if self.verified else 'NOT verified (2026-09-23 bag fit + hand-measured ends)'}"
        )
