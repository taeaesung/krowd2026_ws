"""Virtual-wall supervisor: static lane boundaries, route corridor fallback.

Inputs: the fixed continuous left/right lane boundaries of a map profile
(map frame, e.g. 692 points each for 20260913_103732) when the profile has
them, and ALWAYS a fallback corridor built from the Global Path itself at
+/- `fallback_half_width_m` (3 m). Every point the supervisor checks is
measured against the lane walls where they cover it and against the route
corridor where they do not (a route start before the lane map begins, an
intersection the lane map skips, a test map with no lane map at all) - per
POINT of the predicted footprint, so a horizon that runs off the end of the
lane map switches walls mid-prediction instead of stopping the vehicle. The
two are never combined at one point: where the lane walls exist, only they
count.

The corridor is contracted inwards by

    inset = vehicle half width + localization error + map error
            + tracking error + physical margin

and every control cycle the nominal (blended) steering command is
forward-simulated with the kinematic bicycle model: the candidate angle is
applied AT ONCE for `candidate_hold_s` (as in krowd_final_control), then Pure
Pursuit on the Global Path takes over, rate limited at the measured road-wheel
rate. The rotated footprint's four corners must stay inside the contracted
corridor at every step; a point no wall covers (past the ends of both the lane
map and the route corridor) has no wall to hit. If the nominal command fails,
the candidates are tried nearest first (no time budget) and the first safe one
is taken. The supervisor never asks for a stop (2026-10-01 decision):

    NO_SAFE_STEERING_CANDIDATE  every candidate hits a wall: the one with the
                                most predicted clearance is steered and the node
                                commands speed 0 WITHOUT the brake - the crawl
                                when standing still (a vehicle that does not
                                move cannot steer back)
    LANE_OUTSIDE                all four corners are already beyond a wall:
                                candidates cannot be told apart from there, so
                                the nominal command steers back and the node
                                crawls (corridor.outside_lane_speed_kmh)

A footprint only partly over a line is still "inside" and is evaluated as usual.
Which stretch of a wall is "the vehicle's wall" is fixed when the route arrives
(StaticCorridor.bind_route), never searched by position while driving.

The lane boundaries include route-based interpolation in intersections and
undetected stretches (see the map's README): they are a supervisor for the
Global Path, not ground truth.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math

import numpy as np

from .geometry import bicycle_step, footprint_corners, nearest_on_segments, to_body_frame
from .pure_pursuit import curvature_lookahead, pure_pursuit_steering, rate_limited, time_floored_lookahead


@dataclass
class BoundarySet:
    left: np.ndarray  # (N, 2) map frame, along the direction of travel
    right: np.ndarray  # (M, 2)
    stoplines: list  # list of (K, 2) arrays, informational only
    source: str
    left_namespace: str
    right_namespace: str


def load_boundaries(
    csv_path,
    left_key: str = "leftBoundary",
    right_key: str = "rightBoundary",
) -> BoundarySet:
    segments: dict[int, list] = {}
    namespaces: dict[int, str] = {}
    with open(csv_path, newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"segment_id", "point_index", "namespace", "map_x", "map_y"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{csv_path}: expected columns {sorted(required)}")
        for row in reader:
            sid = int(row["segment_id"])
            segments.setdefault(sid, []).append(
                (int(row["point_index"]), float(row["map_x"]), float(row["map_y"]))
            )
            namespaces[sid] = row["namespace"]
    left_ids = [s for s, ns in namespaces.items() if left_key in ns]
    right_ids = [s for s, ns in namespaces.items() if right_key in ns]
    if len(left_ids) != 1 or len(right_ids) != 1:
        raise ValueError(
            f"{csv_path}: expected exactly one '{left_key}' and one '{right_key}' segment, "
            f"found {left_ids} / {right_ids}"
        )

    def as_array(sid: int) -> np.ndarray:
        pts = sorted(segments[sid])
        indices = [p[0] for p in pts]
        if indices != list(range(len(indices))):
            raise ValueError(f"{csv_path}: segment {sid} has missing point indices")
        arr = np.array([[p[1], p[2]] for p in pts], dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{csv_path}: segment {sid} has non-finite coordinates")
        return arr

    left = as_array(left_ids[0])
    right = as_array(right_ids[0])
    stoplines = [
        as_array(sid) for sid in sorted(segments) if sid not in (left_ids[0], right_ids[0])
    ]
    return BoundarySet(
        left=left,
        right=right,
        stoplines=stoplines,
        source=str(csv_path),
        left_namespace=namespaces[left_ids[0]],
        right_namespace=namespaces[right_ids[0]],
    )


def validate_boundaries(bounds: BoundarySet) -> dict:
    """Sanity checks: sizes, no crossing, width range, orientation."""

    left, right = bounds.left, bounds.right
    if len(left) < 2 or len(right) < 2:
        raise ValueError("boundaries need at least two points each")
    n = min(len(left), len(right))
    width = np.hypot(*(left[:n] - right[:n]).T)
    # Orientation: the right boundary must lie to the RIGHT of the directed
    # left polyline (negative cross) at the matching stations.
    d = np.diff(left[:n], axis=0)
    rel = right[: n - 1] - left[: n - 1]
    cross = d[:, 0] * rel[:, 1] - d[:, 1] * rel[:, 0]
    right_of_left = float(np.mean(cross < 0.0))
    if right_of_left < 0.99:
        raise ValueError(
            f"boundary orientation inconsistent: right lies right of left on only "
            f"{right_of_left * 100:.1f}% of stations"
        )
    if float(width.min()) <= 0.5:
        raise ValueError(f"boundaries touch or cross (min width {width.min():.2f} m)")
    return {
        "left_points": int(len(left)),
        "right_points": int(len(right)),
        "width_min_m": float(width.min()),
        "width_median_m": float(np.median(width)),
        "width_max_m": float(width.max()),
        "right_of_left_fraction": right_of_left,
    }


def route_boundaries(xy: np.ndarray, half_width_m: float) -> BoundarySet:
    """The fallback corridor: the route offset +/- half_width_m along its normal.

    The normal at each vertex comes from the central difference of its
    neighbours (one-sided at the ends). An offset polyline folds on itself
    where the route's radius drops below half_width_m; the tightest corner of
    the mapped courses (4.4-5.8 m) stays above the 3 m default.
    """

    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if len(xy) < 2:
        raise ValueError("a route corridor needs at least two route points")
    if half_width_m <= 0.0:
        raise ValueError("fallback half width must be positive")
    tangent = np.empty_like(xy)
    tangent[1:-1] = xy[2:] - xy[:-2]
    tangent[0] = xy[1] - xy[0]
    tangent[-1] = xy[-1] - xy[-2]
    norm = np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)
    tangent /= norm[:, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # +90 deg = left
    return BoundarySet(
        left=xy + half_width_m * normal,
        right=xy - half_width_m * normal,
        stoplines=[],
        source=f"route +/-{half_width_m:.1f} m",
        left_namespace="route_left",
        right_namespace="route_right",
    )


def _extend(poly: np.ndarray, length_m: float) -> np.ndarray:
    if length_m <= 0.0:
        return poly
    d0 = poly[0] - poly[1]
    d1 = poly[-1] - poly[-2]
    d0 = d0 / max(np.hypot(*d0), 1e-9) * length_m
    d1 = d1 / max(np.hypot(*d1), 1e-9) * length_m
    return np.vstack([poly[0] + d0, poly, poly[-1] + d1])


@dataclass
class Clearance:
    left_min: float  # metres inside the left wall (negative = outside)
    right_min: float
    covered: bool  # every point projected strictly inside the polyline extent

    @property
    def min(self) -> float:
        return min(self.left_min, self.right_min)


class StaticCorridor:
    """Windowed clearance queries against two boundary polylines."""

    def __init__(self, bounds: BoundarySet, end_extension_m: float = 0.0, window_points: int = 45):
        self.left = _extend(bounds.left, end_extension_m)
        self.right = _extend(bounds.right, end_extension_m)
        self.window = int(window_points)
        self._hint_left: int | None = None
        self._hint_right: int | None = None
        self._pinned = False
        self.bounds = bounds
        # Filled by bind_route(): wall segment per ROUTE point, left and right.
        self._route_left: np.ndarray | None = None
        self._route_right: np.ndarray | None = None

    def reset(self) -> None:
        self._hint_left = None
        self._hint_right = None
        self._pinned = False

    # ---- route binding -----------------------------------------------------
    @property
    def bound(self) -> bool:
        return self._route_left is not None

    def bind_route(self, route_xy, window: int = 15, tie_m: float = 1.0) -> dict:
        """Tie every route point to its wall segments ONCE, when the map is loaded.

        The walls never change, so which stretch of them is "the vehicle's wall"
        is decided here, by walking the route and both wall polylines FORWARD
        together: route point i may only use wall segments at or after the one
        point i-1 used (within `window` segments). Where the route crosses or
        runs beside itself (the 20260913_101241 X crossing at wp 60/369, the
        two-way road of 20260913_103732) a nearest-in-xy search would pick the
        other pass's wall; a forward walk cannot. Point 0 is searched over the
        whole wall and, among segments within `tie_m` of the closest, takes the
        EARLIEST one, so a route that starts beside its own end starts at the
        wall's beginning. During driving locate_route_index() only reads the
        table - no search, no global fallback.
        """

        xy = np.asarray(route_xy, dtype=float).reshape(-1, 2)
        if len(xy) < 2:
            raise ValueError("binding needs at least two route points")
        self._route_left, left_dist = self._walk(self.left, xy, window, tie_m)
        self._route_right, right_dist = self._walk(self.right, xy, window, tie_m)
        warnings = []
        for name, table, dist in (("left", self._route_left, left_dist), ("right", self._route_right, right_dist)):
            steps = np.diff(table)
            if len(steps) and int(steps.max()) >= window - 1:
                warnings.append(f"{name} wall: the walk jumped {int(steps.max())} segments at once (route kink or wall gap?)")
            if float(dist.max()) > 10.0:
                far = np.flatnonzero(dist > 10.0)
                warnings.append(f"{name} wall is >10 m from route points {far[:5].tolist()}{'...' if len(far) > 5 else ''}")
        summary = (
            f"{len(xy)} route points -> wall segments left {int(self._route_left[0])}..{int(self._route_left[-1])} "
            f"of {len(self.left) - 1}, right {int(self._route_right[0])}..{int(self._route_right[-1])} of {len(self.right) - 1}; "
            f"route-to-wall distance left {left_dist.min():.2f}-{left_dist.max():.2f} m, right {right_dist.min():.2f}-{right_dist.max():.2f} m"
        )
        return {"summary": summary, "warnings": warnings, "left_dist": left_dist, "right_dist": right_dist}

    @staticmethod
    def _walk(poly: np.ndarray, xy: np.ndarray, window: int, tie_m: float) -> tuple[np.ndarray, np.ndarray]:
        a, b = poly[:-1], poly[1:]
        table = np.zeros(len(xy), dtype=int)
        dist = np.zeros(len(xy))
        prev: int | None = None
        for i in range(len(xy)):
            if prev is None:
                d = nearest_on_segments(xy[i : i + 1], a, b)[0][0]
                k = int(np.flatnonzero(d <= d.min() + tie_m).min())
            else:
                lo, hi = prev, min(prev + window, len(a))
                d = nearest_on_segments(xy[i : i + 1], a[lo:hi], b[lo:hi])[0][0]
                k = lo + int(np.argmin(d))
            table[i] = k
            dist[i] = float(d.min())
            prev = k
        return table, dist

    def locate_route_index(self, route_index: int) -> None:
        """Pin the search windows on the wall segments bound to this route point."""

        if self._route_left is None:
            raise RuntimeError("bind_route() first")
        i = min(max(int(route_index), 0), len(self._route_left) - 1)
        self._hint_left = int(self._route_left[i])
        self._hint_right = int(self._route_right[i])
        self._pinned = True

    def _window(self, poly: np.ndarray, hint: int | None) -> tuple[int, int]:
        n = len(poly) - 1  # segments
        if hint is None:
            return 0, n
        lo = max(hint - self.window, 0)
        hi = min(hint + self.window, n)
        return lo, max(hi, lo + 1)

    def _side_clearance(
        self, poly: np.ndarray, hint: int | None, points: np.ndarray, inside_sign: float
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """(signed clearance per point, hint segment, covered per point)."""

        lo, hi = self._window(poly, hint)
        result = self._clearance_in(poly, lo, hi, points, inside_sign)
        full = (lo == 0 and hi == len(poly) - 1)
        if not full and result[3] and not self._pinned:
            # Nearest segment sits on the window edge: the vehicle left the
            # window (pose jump, reset) - search the whole polyline instead.
            # Never for a pinned window: there the window IS the truth.
            result = self._clearance_in(poly, 0, len(poly) - 1, points, inside_sign)
        return result[0], result[1], result[2]

    def _clearance_in(self, poly, lo, hi, points, inside_sign):
        a = poly[lo:hi]
        b = poly[lo + 1 : hi + 1]
        dist, t, t_raw, cross = nearest_on_segments(points, a, b)
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(points))
        signed = np.where(np.sign(cross[rows, best]) == inside_sign, dist[rows, best], -dist[rows, best])
        seg_global = best + lo
        at_start = (seg_global == 0) & (t_raw[rows, best] < 0.0)
        at_end = (seg_global == len(poly) - 2) & (t_raw[rows, best] > 1.0)
        covered = ~(at_start | at_end)
        on_edge = bool(np.any((best == 0) & (lo > 0)) or np.any((best == len(a) - 1) & (hi < len(poly) - 1)))
        return signed.astype(float), int(seg_global[0]), covered, on_edge

    def locate(self, x: float, y: float) -> None:
        """Nearest-in-xy search windows (global search if lost). NOT used while
        driving - see bind_route(): on the 20260913_103732 course the route's
        end runs 2 m from its start (wp 8-30) and this measured the start
        against the END's left wall, calling the whole start "outside"."""

        self._pinned = False
        p = np.array([[x, y]])
        _, self._hint_left, _ = self._side_clearance(self.left, self._hint_left, p, -1.0)
        _, self._hint_right, _ = self._side_clearance(self.right, self._hint_right, p, +1.0)

    def clearance_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per point: (signed clearance to the left wall, to the right wall, covered)."""

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        left, _, cov_l = self._side_clearance(self.left, self._hint_left, points, -1.0)
        right, _, cov_r = self._side_clearance(self.right, self._hint_right, points, +1.0)
        return left, right, cov_l & cov_r

    def clearance(self, points: np.ndarray) -> Clearance:
        left, right, covered = self.clearance_points(points)
        return Clearance(left_min=float(left.min()), right_min=float(right.min()), covered=bool(covered.all()))


@dataclass
class SupervisorParams:
    wheelbase_m: float
    front_m: float
    rear_m: float
    half_width_m: float
    inset_m: float  # full contraction incl. half width
    sim_dt_s: float = 0.1
    min_horizon_s: float = 1.0
    max_horizon_s: float = 4.0
    candidate_step_rad: float = math.radians(1.0)
    candidate_count: int = 15  # each side
    max_steering_rate_rad_s: float = math.radians(40.0)
    lookahead_straight_m: float = 7.0
    lookahead_curve_m: float = 3.0
    lookahead_straight_radius_m: float = 50.0
    lookahead_curve_radius_m: float = 15.0
    lookahead_time_s: float = 0.0  # the controller's lookahead floor (time_floored_lookahead); 0 = off
    lookahead_straight_time_s: float = 0.0  # the straight value grows to this many seconds of travel; 0 = off
    min_sim_speed_mps: float = 1.0
    candidate_hold_s: float = 0.5  # the candidate is held this long before Pure Pursuit resumes
    # Candidate offsets are limited to the road-wheel angle whose steady-state
    # lateral acceleration v^2 tan(delta)/L stays below this (0 = no limit).
    # 2026-10-17 1017_f_ppstanley7: at 40 km/h the +/-5..13 deg candidates (held
    # 0.5 s each) threw the vehicle wall to wall at 40-55 deg/s of yaw.
    candidate_max_lat_accel_mps2: float = 0.0


@dataclass
class Rollout:
    """One forward simulation: the steering it started with and where it went."""

    steering_rad: float
    ok: bool
    predicted_min_clearance: float  # min over the horizon, minus the corner margin
    poses: list  # (x, y, yaw) per prediction step


@dataclass
class CorridorDecision:
    ok: bool
    steering_rad: float
    reason: str
    current_clearance: float
    predicted_min_clearance: float
    covered: bool  # every corner of the current footprint has a wall to measure against
    candidate_offset_rad: float = 0.0
    predicted_poses: list = field(default_factory=list)
    fallback_fraction: float = 0.0  # share of checked points measured against the route corridor
    candidates_tried: int = 0
    rollouts: list = field(default_factory=list)  # every Rollout tried this cycle, nominal first
    outside: bool = False  # LANE_OUTSIDE: all four corners already beyond a wall


class CorridorSupervisor:
    def __init__(
        self,
        corridor: StaticCorridor | None,
        params: SupervisorParams,
        fallback: StaticCorridor | None = None,
    ):
        self.corridor = corridor  # lane boundaries; None = no lane map for this route
        self.fallback = fallback  # route +/- half width; set/replaced per trajectory
        self.p = params

    # The footprint corners relative to the contracted wall: an inset of
    # `inset_m` applied to the CORNERS is equivalent to contracting the corridor
    # by inset_m minus the half width, because the corners already sit at the
    # half width. So the corner margin is inset - half_width.
    @property
    def corner_margin(self) -> float:
        return self.p.inset_m - self.p.half_width_m

    @property
    def available(self) -> bool:
        return self.corridor is not None or self.fallback is not None

    def set_fallback(self, fallback: StaticCorridor | None) -> None:
        self.fallback = fallback

    def _footprint(self, x, y, yaw) -> np.ndarray:
        return footprint_corners(x, y, yaw, self.p.front_m, self.p.rear_m, self.p.half_width_m)

    def _locate(self, route, route_index: int) -> None:
        # By route index, never by xy: see StaticCorridor.bind_route. The node
        # binds when the trajectory arrives; this is the safety net.
        for corridor in (self.corridor, self.fallback):
            if corridor is not None:
                if not corridor.bound:
                    corridor.bind_route(route.xy)
                corridor.locate_route_index(route_index)

    def _signed_clearance(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Per point: (signed clearance, covered by some wall, fallback fraction).

        Lane walls first; points they do not cover are measured against the
        route corridor. A point neither covers has no wall to hit: its
        clearance is +inf.
        """

        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        signed = np.full(len(pts), math.inf)
        covered = np.zeros(len(pts), dtype=bool)
        if len(pts) == 0:
            return signed, covered, 0.0
        if self.corridor is not None:
            left, right, lane_cov = self.corridor.clearance_points(pts)
            signed[lane_cov] = np.minimum(left, right)[lane_cov]
            covered |= lane_cov
        need = np.flatnonzero(~covered)
        fallback_fraction = len(need) / len(pts)
        if len(need) and self.fallback is not None:
            fl, fr, fcov = self.fallback.clearance_points(pts[need])
            signed[need[fcov]] = np.minimum(fl, fr)[fcov]
            covered[need[fcov]] = True
        return signed, covered, fallback_fraction

    def _clearance(self, pts: np.ndarray) -> tuple[float, bool, float]:
        """(min signed clearance, every point covered by some wall, fallback fraction)."""

        signed, covered, fallback_fraction = self._signed_clearance(pts)
        if len(signed) == 0:
            return -math.inf, False, 0.0
        return float(signed.min()), bool(covered.all()), fallback_fraction

    def _simulate(self, route, x, y, yaw, speed, s, first_delta, delta_limits, horizon_s, dt):
        """Predicted footprints when `first_delta` is commanded for
        `candidate_hold_s` and Pure Pursuit on the Global Path takes over.

        The candidate is applied at once, as in krowd_final_control: the
        supervisor asks where each steering angle leads, not how fast the motor
        reaches it. The continuation is Pure Pursuit alone - the supervisor
        tests the candidate's first seconds, not the blend's tuning - rate
        limited at the measured road-wheel rate from the candidate angle.
        """

        pts = []
        poses = []
        v = max(speed, self.p.min_sim_speed_mps)
        cur_s = s
        steps = max(int(math.ceil(horizon_s / dt)), 1)
        hold_steps = max(int(round(self.p.candidate_hold_s / dt)), 1)
        delta = first_delta
        for k in range(steps):
            if k >= hold_steps:
                # Pure Pursuit continuation from the predicted pose.
                proj = route.project(x, y, yaw, cur_s, back_window_m=5.0, forward_window_m=15.0)
                if proj is None:
                    break
                cur_s = proj.s
                straight_ld = time_floored_lookahead(self.p.lookahead_straight_m, v, self.p.lookahead_straight_time_s)
                kmax = route.max_abs_curvature(cur_s, cur_s + straight_ld)
                ld = time_floored_lookahead(
                    curvature_lookahead(
                        kmax,
                        straight_ld,
                        self.p.lookahead_curve_m,
                        self.p.lookahead_straight_radius_m,
                        self.p.lookahead_curve_radius_m,
                    ),
                    v,
                    self.p.lookahead_time_s,
                )
                tx, ty, _, _ = route.point_ahead(cur_s, ld)
                bx, by = to_body_frame(x, y, yaw, tx, ty)
                desired = pure_pursuit_steering(bx, by, self.p.wheelbase_m)
                desired = min(max(desired, delta_limits[0]), delta_limits[1])
                delta = rate_limited(delta, desired, self.p.max_steering_rate_rad_s, dt)
            x, y, yaw = bicycle_step(x, y, yaw, v, delta, self.p.wheelbase_m, dt)
            poses.append((x, y, yaw))
            pts.append(self._footprint(x, y, yaw))
        if not pts:
            return np.zeros((0, 2)), poses
        return np.vstack(pts), poses

    def evaluate(
        self,
        route,
        x: float,
        y: float,
        yaw: float,
        speed_mps: float,
        route_s: float,
        nominal_delta: float,
        prev_delta: float,
        delta_limits: tuple[float, float],
        stopping_time_s: float,
    ) -> CorridorDecision:
        if not self.available:
            return CorridorDecision(False, prev_delta, "CORRIDOR_UNAVAILABLE", -math.inf, -math.inf, False)
        self._locate(route, int(route.index_at_s(route_s)))
        margin = self.corner_margin
        corners, corners_covered, now_fb = self._signed_clearance(self._footprint(x, y, yaw))
        now_clearance = float(corners.min()) - margin
        now_covered = bool(corners_covered.all())
        if bool(np.all(corners < margin)):
            return CorridorDecision(
                False, nominal_delta, "LANE_OUTSIDE", now_clearance, now_clearance, now_covered,
                fallback_fraction=now_fb, outside=True,
            )

        horizon = min(max(stopping_time_s, self.p.min_horizon_s), self.p.max_horizon_s)
        dt = self.p.sim_dt_s
        rollouts: list[Rollout] = []

        def check(delta: float) -> tuple[Rollout, float]:
            pts, poses = self._simulate(route, x, y, yaw, speed_mps, route_s, delta, delta_limits, horizon, dt)
            if len(pts) == 0:
                rollout, fb = Rollout(delta, False, -math.inf, poses), 0.0
            else:
                mn, _, fb = self._clearance(pts)
                rollout = Rollout(delta, mn >= margin, mn - margin, poses)
            rollouts.append(rollout)
            return rollout, fb

        nominal, fb = check(nominal_delta)
        if nominal.ok:
            return CorridorDecision(
                True, nominal_delta, "OK", now_clearance, nominal.predicted_min_clearance, now_covered,
                0.0, nominal.poses, fb, len(rollouts), rollouts,
            )

        # Candidate search: nearest to the nominal first, every candidate if need be.
        max_offset = math.inf
        if self.p.candidate_max_lat_accel_mps2 > 0.0 and speed_mps > 0.0:
            max_offset = math.atan(self.p.candidate_max_lat_accel_mps2 * self.p.wheelbase_m / speed_mps ** 2)
        best, best_fb = nominal, fb
        for k in range(1, self.p.candidate_count + 1):
            if k * self.p.candidate_step_rad > max_offset:
                break
            for sign in (+1.0, -1.0):
                cand = nominal_delta + sign * k * self.p.candidate_step_rad
                if not delta_limits[0] <= cand <= delta_limits[1]:
                    continue
                rollout, fb_c = check(cand)
                if rollout.ok:
                    return CorridorDecision(
                        True, cand, f"CANDIDATE_{sign * k * math.degrees(self.p.candidate_step_rad):+.0f}deg",
                        now_clearance, rollout.predicted_min_clearance, now_covered, cand - nominal_delta,
                        rollout.poses, fb_c, len(rollouts), rollouts,
                    )
                if rollout.predicted_min_clearance > best.predicted_min_clearance:
                    best, best_fb = rollout, fb_c
        return CorridorDecision(
            False, best.steering_rad, "NO_SAFE_STEERING_CANDIDATE", now_clearance, best.predicted_min_clearance,
            now_covered, best.steering_rad - nominal_delta, best.poses, best_fb, len(rollouts), rollouts,
        )
