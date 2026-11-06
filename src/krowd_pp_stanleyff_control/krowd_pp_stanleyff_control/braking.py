"""Stopping distance model and approach-speed limit (section 8.1 D).

    d_model    = v * t_total_latency + v^2 / (2 * a_safe) + d_margin
    d_required = max(d_model, measured_envelope(v) + d_margin)

The measured envelope is the team's E-stop braking table (competition surface,
loaded vehicle, wheel-sensor speed). Between table speeds the UPPER bin's
distance is used - never an optimistic interpolation. Above the table the last
value is scaled by (v / v_max)^2 (kinetic energy), which is a warning-level
extrapolation; the controller caps its speed well inside the table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BrakingModel:
    safe_decel_mps2: float
    total_latency_s: float
    margin_m: float
    envelope_kmh_m: tuple[tuple[float, float], ...]  # sorted by speed

    @classmethod
    def from_config(cls, cfg: dict) -> "BrakingModel":
        env = sorted((float(k), float(d)) for k, d in cfg["measured_envelope_kmh_m"])
        model = cls(
            safe_decel_mps2=float(cfg["safe_deceleration_mps2"]),
            total_latency_s=float(cfg["total_latency_s"]),
            margin_m=float(cfg["distance_margin_m"]),
            envelope_kmh_m=tuple(env),
        )
        model.validate()
        return model

    def validate(self) -> None:
        if not self.safe_decel_mps2 > 0.0:
            raise ValueError("safe_deceleration_mps2 must be > 0")
        if self.total_latency_s < 0.0 or self.margin_m < 0.0:
            raise ValueError("latency and margin must be >= 0")
        if len(self.envelope_kmh_m) < 2:
            raise ValueError("measured braking envelope needs >= 2 points")
        speeds = [k for k, _ in self.envelope_kmh_m]
        if any(b <= a for a, b in zip(speeds, speeds[1:])):
            raise ValueError("braking envelope speeds must be strictly increasing")

    def envelope_distance(self, speed_mps: float) -> float:
        kmh = max(speed_mps, 0.0) * 3.6
        for table_kmh, dist in self.envelope_kmh_m:
            if kmh <= table_kmh + 1e-9:
                return dist
        top_kmh, top_dist = self.envelope_kmh_m[-1]
        return top_dist * (kmh / top_kmh) ** 2

    def model_distance(self, speed_mps: float) -> float:
        v = max(speed_mps, 0.0)
        return v * self.total_latency_s + v * v / (2.0 * self.safe_decel_mps2) + self.margin_m

    def required_distance(self, speed_mps: float) -> float:
        return max(self.model_distance(speed_mps), self.envelope_distance(speed_mps) + self.margin_m)

    def speed_limit_for_distance(self, distance_m: float) -> float:
        """Largest speed whose required stopping distance fits in `distance_m`.

        Solved on the analytic model, then reduced until the measured envelope
        also fits (the envelope is a step function, so a short bisection).
        """

        d = distance_m - self.margin_m
        if d <= 0.0:
            return 0.0
        a = self.safe_decel_mps2
        t = self.total_latency_s
        v = -a * t + math.sqrt(a * a * t * t + 2.0 * a * d)
        lo, hi = 0.0, max(v, 0.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if self.required_distance(mid) <= distance_m:
                lo = mid
            else:
                hi = mid
        return lo
