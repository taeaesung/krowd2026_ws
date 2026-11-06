"""Per-section speed and steering parameters, keyed by waypoint ranges.

`config/segments_<map>.yaml` holds one row per section of the route:

    - name: "start straight"
      start_idx: 3
      end_idx: 50            # inclusive waypoint range
      speed_kmh: null        # null = inherit (global cruise / mission cap still apply)
      lookahead_straight_m: null
      lookahead_curve_m: null
      straight_radius_m: null
      curve_radius_m: null
      max_road_wheel_rate_deg_s: null

Every value is an OVERRIDE of the global pure_pursuit / steering / speed
settings; the mission caps (10 km/h signal approach, 7 km/h obstacle) and
the braking envelope always apply on top, so a section can only make the
vehicle slower than the safety logic allows, never faster. Rows must not
overlap. Gaps fall back to the global values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start_idx: int
    end_idx: int
    speed_kmh: Optional[float] = None
    lookahead_straight_m: Optional[float] = None
    lookahead_curve_m: Optional[float] = None
    straight_radius_m: Optional[float] = None
    curve_radius_m: Optional[float] = None
    max_road_wheel_rate_deg_s: Optional[float] = None
    notes: str = ""


@dataclass(frozen=True)
class SegmentParams:
    """Effective parameters at one route position (global defaults + overrides)."""

    name: str
    speed_cap_mps: Optional[float]
    lookahead_straight_m: float
    lookahead_curve_m: float
    straight_radius_m: float
    curve_radius_m: float
    max_road_wheel_rate_rad_s: float


def _opt(value, key, name) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"segment '{name}': {key} must be a positive number or null, got {value}")
    return number


def load_segments(cfg: dict) -> list[SegmentSpec]:
    rows = cfg.get("list", []) or []
    specs: list[SegmentSpec] = []
    for raw in rows:
        name = str(raw.get("name", f"segment{len(specs)}"))
        start = int(raw["start_idx"])
        end = int(raw["end_idx"])
        if end < start:
            raise ValueError(f"segment '{name}': end_idx {end} < start_idx {start}")
        specs.append(
            SegmentSpec(
                name=name,
                start_idx=start,
                end_idx=end,
                speed_kmh=_opt(raw.get("speed_kmh"), "speed_kmh", name),
                lookahead_straight_m=_opt(raw.get("lookahead_straight_m"), "lookahead_straight_m", name),
                lookahead_curve_m=_opt(raw.get("lookahead_curve_m"), "lookahead_curve_m", name),
                straight_radius_m=_opt(raw.get("straight_radius_m"), "straight_radius_m", name),
                curve_radius_m=_opt(raw.get("curve_radius_m"), "curve_radius_m", name),
                max_road_wheel_rate_deg_s=_opt(
                    raw.get("max_road_wheel_rate_deg_s"), "max_road_wheel_rate_deg_s", name
                ),
                notes=str(raw.get("notes", "")),
            )
        )
    specs.sort(key=lambda x: x.start_idx)
    for a, b in zip(specs, specs[1:]):
        if b.start_idx <= a.end_idx:
            raise ValueError(f"segments '{a.name}' and '{b.name}' overlap (waypoints {b.start_idx}..{a.end_idx})")
    for spec in specs:
        if (spec.lookahead_curve_m or 0) > (spec.lookahead_straight_m or math.inf):
            raise ValueError(f"segment '{spec.name}': lookahead_curve_m must be <= lookahead_straight_m")
        if (spec.curve_radius_m or 0) >= (spec.straight_radius_m or math.inf):
            raise ValueError(f"segment '{spec.name}': curve_radius_m must be < straight_radius_m")
    return specs


@dataclass(frozen=True)
class ResolvedSegment:
    spec: SegmentSpec
    start_s: float
    end_s: float  # arc length at end_idx + 1 (exclusive), or route end


def resolve_segments(specs: list[SegmentSpec], route) -> list[ResolvedSegment]:
    out = []
    for spec in specs:
        if not 0 <= spec.start_idx < route.count or not 0 <= spec.end_idx < route.count:
            raise ValueError(f"segment '{spec.name}': indices outside route 0..{route.count - 1}")
        end_s = route.s_at_index(spec.end_idx + 1) if spec.end_idx + 1 < route.count else route.length + 1e-6
        out.append(ResolvedSegment(spec, route.s_at_index(spec.start_idx), end_s))
    return out


class SegmentTable:
    def __init__(self, resolved: list[ResolvedSegment], defaults: SegmentParams):
        self.resolved = list(resolved)
        self.defaults = defaults

    def active(self, s: float) -> Optional[ResolvedSegment]:
        for seg in self.resolved:
            if seg.start_s <= s < seg.end_s:
                return seg
        return None

    def params_at(self, s: float) -> SegmentParams:
        seg = self.active(s)
        if seg is None:
            return self.defaults
        sp = seg.spec
        d = self.defaults
        lookahead_straight = sp.lookahead_straight_m or d.lookahead_straight_m
        lookahead_curve = sp.lookahead_curve_m or d.lookahead_curve_m
        return replace(
            d,
            name=sp.name,
            speed_cap_mps=None if sp.speed_kmh is None else sp.speed_kmh / 3.6,
            lookahead_straight_m=lookahead_straight,
            lookahead_curve_m=min(lookahead_curve, lookahead_straight),
            straight_radius_m=sp.straight_radius_m or d.straight_radius_m,
            curve_radius_m=sp.curve_radius_m or d.curve_radius_m,
            max_road_wheel_rate_rad_s=(
                math.radians(sp.max_road_wheel_rate_deg_s)
                if sp.max_road_wheel_rate_deg_s is not None
                else d.max_road_wheel_rate_rad_s
            ),
        )

    def upcoming_speed_caps(self, s: float, horizon_m: float) -> list[tuple[float, float]]:
        """(start_s, cap_mps) of segment speed caps that begin within horizon_m ahead."""

        out = []
        for seg in self.resolved:
            if seg.spec.speed_kmh is None:
                continue
            if s < seg.start_s <= s + horizon_m:
                out.append((seg.start_s, seg.spec.speed_kmh / 3.6))
        return out
