"""Live stop-line distance from /glane3d/stopline_markers (section 8.1 D).

The lane detector publishes a MarkerArray whose LINE_STRIP markers are stop
line curves in `base_link` (rear axle, +x forward). Their `x` values are
along-track distances from the control point. The TEXT marker on
/glane3d/stopline_distance_marker is never parsed here.

A candidate is kept when it is fresh, finite, ahead of the vehicle, overlaps
the ego lane laterally, is wide enough and roughly transverse. The nearest
positive candidate is the estimate. Between detections the distance is
carried forward by the arc length the vehicle has travelled, but only up to
`live_ttl_s` (a measured value; None disables the live source entirely).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class StoplineObservation:
    stamp_ns: int
    distance_m: float  # median x of the line, rear axle -> line
    y_min: float
    y_max: float
    marker_id: int
    route_s_at_stamp: float | None


@dataclass(frozen=True)
class StoplineEstimate:
    valid: bool
    distance_m: float | None
    age_s: float | None
    reason: str


class StoplineEstimator:
    def __init__(
        self,
        half_band_m: float,
        min_span_m: float,
        max_abs_slope: float,
        max_distance_m: float,
    ):
        self.half_band = float(half_band_m)
        self.min_span = float(min_span_m)
        self.max_slope = float(max_abs_slope)
        self.max_distance = float(max_distance_m)
        self.last: StoplineObservation | None = None
        self.rejected_last = ""

    def ingest(self, markers, stamp_ns: int, route_s: float | None) -> StoplineObservation | None:
        """`markers` is an iterable of visualization_msgs/Marker."""

        best: StoplineObservation | None = None
        reasons = []
        for marker in markers:
            if marker.action != 0 or marker.type != 4:  # ADD, LINE_STRIP
                continue
            if len(marker.points) < 2:
                reasons.append("too_few_points")
                continue
            xs = np.array([p.x for p in marker.points], dtype=float)
            ys = np.array([p.y for p in marker.points], dtype=float)
            if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                reasons.append("non_finite")
                continue
            distance = float(np.median(xs))
            if not 0.0 < distance <= self.max_distance:
                reasons.append(f"distance_{distance:.1f}")
                continue
            y_min, y_max = float(ys.min()), float(ys.max())
            if y_max < -self.half_band or y_min > self.half_band:
                reasons.append("no_lateral_overlap")
                continue
            if (y_max - y_min) < self.min_span:
                reasons.append("short_span")
                continue
            # Transverse check: x should vary little across y.
            span_y = max(y_max - y_min, 1e-6)
            slope = float((xs.max() - xs.min()) / span_y)
            if slope > self.max_slope:
                reasons.append(f"slope_{slope:.2f}")
                continue
            obs = StoplineObservation(stamp_ns, distance, y_min, y_max, int(marker.id), route_s)
            if best is None or obs.distance_m < best.distance_m:
                best = obs
        if best is not None:
            self.last = best
            self.rejected_last = ""
        else:
            self.rejected_last = ",".join(sorted(set(reasons))) if reasons else "no_candidates"
        return best

    def estimate(self, now_ns: int, ttl_s: float | None, route_s_now: float | None) -> StoplineEstimate:
        if ttl_s is None:
            return StoplineEstimate(False, None, None, "LIVE_DISABLED_TTL_UNSET")
        if self.last is None:
            return StoplineEstimate(False, None, None, "NO_OBSERVATION")
        age = (now_ns - self.last.stamp_ns) * 1e-9
        if age < -0.05:
            return StoplineEstimate(False, None, age, "FUTURE_STAMP")
        if age > ttl_s:
            return StoplineEstimate(False, None, age, "STALE")
        travelled = 0.0
        if route_s_now is not None and self.last.route_s_at_stamp is not None:
            travelled = max(route_s_now - self.last.route_s_at_stamp, 0.0)
        return StoplineEstimate(True, self.last.distance_m - travelled, age, "OK")
