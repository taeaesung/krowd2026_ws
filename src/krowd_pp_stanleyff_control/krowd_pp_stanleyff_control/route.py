"""Global Path model: the ONLY steering source of the controller.

Built from krowd_control_interfaces/Trajectory (/planning/trajectory). Pose i
of that message is waypoint i of waypoints.csv, which is what a map profile's
mission table indexes, so the controller refuses a trajectory whose pose count
does not match the profile (see control_node).

Projection is MONOTONIC along the route: after the first fix the search is
limited to a window around the previous arc position and the position may
only move backwards by `max_backtrack_m`. This is what keeps a near-loop
route (start and end 6.7 m apart) and localization noise from flipping the
mission progress backwards or to the other end of the course. The controller
keeps TWO such positions - the rear axle's for Pure Pursuit and the front
axle's for the Stanley terms - each with its own continuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import cumulative_length, nearest_on_segments, wrap_angle, yaw_from_quaternion


@dataclass(frozen=True)
class RouteProjection:
    s: float
    segment_index: int
    fraction: float
    lateral_offset: float  # metres, positive = vehicle LEFT of the route
    heading_error: float  # route yaw - vehicle yaw, radians
    foot_x: float
    foot_y: float
    distance: float


@dataclass
class RouteModel:
    frame_id: str
    trajectory_id: str
    valid_until_ns: int
    xy: np.ndarray  # (N, 2)
    yaw: np.ndarray  # (N,)
    curvature: np.ndarray  # (N,) 1/m, positive left
    speed: np.ndarray  # (N,) m/s
    s: np.ndarray  # (N,) cumulative arc length

    # ---- construction -----------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        xy,
        yaw,
        curvature,
        speed,
        frame_id: str = "map",
        trajectory_id: str = "",
        valid_until_ns: int = 0,
    ) -> "RouteModel":
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(xy) < 2:
            raise ValueError("a route needs at least two poses")
        yaw = np.asarray(yaw, dtype=float)
        curvature = np.asarray(curvature, dtype=float)
        speed = np.asarray(speed, dtype=float)
        if not (len(yaw) == len(curvature) == len(speed) == len(xy)):
            raise ValueError("route arrays must have one value per pose")
        if not np.all(np.isfinite(xy)):
            raise ValueError("route contains non-finite coordinates")
        return cls(
            frame_id=frame_id,
            trajectory_id=trajectory_id,
            valid_until_ns=int(valid_until_ns),
            xy=xy,
            yaw=yaw,
            curvature=curvature,
            speed=speed,
            s=cumulative_length(xy),
        )

    @classmethod
    def from_trajectory(cls, msg) -> "RouteModel":
        poses = msg.path.poses
        count = len(poses)
        if count < 2:
            raise ValueError(f"trajectory '{msg.trajectory_id}' has {count} poses")
        if len(msg.curvatures) != count or len(msg.target_speeds) != count:
            raise ValueError(
                f"trajectory '{msg.trajectory_id}': {count} poses but "
                f"{len(msg.curvatures)} curvatures / {len(msg.target_speeds)} speeds"
            )
        xy = np.empty((count, 2))
        yaw = np.empty(count)
        for i, pose in enumerate(poses):
            p = pose.pose.position
            q = pose.pose.orientation
            xy[i, 0] = p.x
            xy[i, 1] = p.y
            yaw[i] = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        return cls.from_arrays(
            xy,
            yaw,
            list(msg.curvatures),
            list(msg.target_speeds),
            frame_id=str(msg.path.header.frame_id),
            trajectory_id=str(msg.trajectory_id),
            valid_until_ns=int(msg.valid_until.sec) * 1_000_000_000
            + int(msg.valid_until.nanosec),
        )

    # ---- queries ----------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.xy)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def run_name(self) -> str:
        """The mapping run named by the trajectory id (`<run>:<N>wp[:loop]`)."""

        return self.trajectory_id.split(":", 1)[0]

    def s_at_index(self, index: int) -> float:
        if not 0 <= index < self.count:
            raise IndexError(f"waypoint index {index} outside 0..{self.count - 1}")
        return float(self.s[index])

    def index_at_s(self, s: float) -> float:
        """Fractional waypoint index at arc length s (clamped)."""

        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.s, s, side="right")) - 1
        i = min(max(i, 0), self.count - 2)
        seg = self.s[i + 1] - self.s[i]
        frac = (s - self.s[i]) / seg if seg > 0 else 0.0
        return i + frac

    def _locate(self, s: float) -> tuple[int, float]:
        s = min(max(s, 0.0), self.length)
        i = int(np.searchsorted(self.s, s, side="right")) - 1
        i = min(max(i, 0), self.count - 2)
        seg = self.s[i + 1] - self.s[i]
        frac = (s - self.s[i]) / seg if seg > 0 else 0.0
        return i, float(min(max(frac, 0.0), 1.0))

    def point_at_s(self, s: float) -> tuple[float, float]:
        i, f = self._locate(s)
        p = self.xy[i] + f * (self.xy[i + 1] - self.xy[i])
        return float(p[0]), float(p[1])

    def yaw_at_s(self, s: float) -> float:
        """Route direction from the geometry (not the recorded yaw column)."""

        i, _ = self._locate(s)
        d = self.xy[i + 1] - self.xy[i]
        return math.atan2(d[1], d[0])

    def heading_over_window(self, s: float, window_m: float) -> float:
        """Route direction at s as the chord over `window_m` of arc centred on it.

        Deliberately not the recorded waypoint yaw: that is the mapping
        drive's own heading estimate, noise included, and the Stanley heading
        term passes it to the steering at its gain. With ~1 m waypoint spacing a
        single segment's direction also steps at every waypoint; the window
        averages that out. window_m <= 0 (or a degenerate chord) uses the
        projected segment alone. Clamped to the route's ends.
        """

        if window_m > 0.0:
            x0, y0 = self.point_at_s(s - 0.5 * window_m)
            x1, y1 = self.point_at_s(s + 0.5 * window_m)
            if math.hypot(x1 - x0, y1 - y0) > 1e-6:
                return math.atan2(y1 - y0, x1 - x0)
        return self.yaw_at_s(s)

    def speed_at_s(self, s: float) -> float:
        """Profile speed at s. A drop between two waypoints is taken whole at the
        first one (the profile's decelerations are steps, speed_profile.py);
        a rise is interpolated."""

        i, f = self._locate(s)
        if self.speed[i + 1] < self.speed[i]:
            return float(self.speed[i + 1])
        return float(self.speed[i] + f * (self.speed[i + 1] - self.speed[i]))

    def curvature_at_s(self, s: float) -> float:
        i, f = self._locate(s)
        return float(self.curvature[i] + f * (self.curvature[i + 1] - self.curvature[i]))

    def max_abs_curvature(self, s_from: float, s_to: float) -> float:
        """Largest |curvature| on the waypoints in [s_from, s_to] (inclusive ends)."""

        i0, _ = self._locate(s_from)
        i1, _ = self._locate(s_to)
        i1 = min(i1 + 1, self.count - 1)
        return float(np.max(np.abs(self.curvature[i0 : i1 + 1])))

    def point_ahead(self, s: float, distance_m: float) -> tuple[float, float, float, bool]:
        """Point `distance_m` further along the route: (x, y, s_target, reached_end)."""

        target_s = s + max(distance_m, 0.0)
        reached_end = target_s >= self.length
        target_s = min(target_s, self.length)
        x, y = self.point_at_s(target_s)
        return x, y, target_s, reached_end

    def remaining(self, s: float) -> float:
        return max(self.length - s, 0.0)

    # ---- joining the route from off it ------------------------------------
    def join_waypoint(
        self,
        x: float,
        y: float,
        yaw: float | None = None,
        tie_distance_m: float = 1.0,
        pass_tie_distance_m: float = 8.0,
        heading_gate_rad: float = math.radians(90.0),
    ) -> tuple[int, float]:
        """Waypoint to drive to from OFF the route: (index, distance to it).

        The waypoint nearest to (x, y) - the caller passes the vehicle's
        lookahead position, so the choice looks where the vehicle is heading,
        not where it stands - with the LOWER index winning whenever "nearest"
        is not clear-cut:

        * along one pass of the route, every waypoint within `tie_distance_m`
          of that pass's nearest distance is equally near, and the lowest
          index of them is taken, so no waypoint still drivable is skipped;
        * where several passes run close together (this course's start and
          end are 6.7 m apart, #53 and #508 4.7 m), every pass within
          `pass_tie_distance_m` of the nearest one competes. Passes running
          against the vehicle's heading (more than `heading_gate_rad` off) are
          dropped first - joining one would mean a U-turn onto it - and the
          lowest-index pass of the rest wins.
        """

        dist = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
        nearest = float(np.min(dist))
        near = np.flatnonzero(dist <= nearest + max(pass_tie_distance_m, 0.0))
        passes: list[list[int]] = []
        for k in near.tolist():
            if passes and k - passes[-1][-1] <= 2:
                passes[-1].append(k)
            else:
                passes.append([k])

        candidates = []
        for members in passes:
            best = min(dist[k] for k in members)
            candidates.append(
                next(k for k in members if dist[k] <= best + max(tie_distance_m, 0.0))
            )
        if yaw is not None and len(candidates) > 1:
            agreeing = [
                k for k in candidates
                if abs(wrap_angle(self._direction_at(k) - yaw)) <= heading_gate_rad
            ]
            candidates = agreeing or candidates
        index = min(candidates)
        return index, float(dist[index])

    def _direction_at(self, index: int) -> float:
        """Route direction leaving waypoint `index` (arriving, for the last)."""

        first = min(index, self.count - 2)
        d = self.xy[first + 1] - self.xy[first]
        return math.atan2(d[1], d[0])

    # ---- projection -------------------------------------------------------
    def project(
        self,
        x: float,
        y: float,
        yaw: float | None,
        prev_s: float | None,
        *,
        back_window_m: float = 8.0,
        forward_window_m: float = 30.0,
        max_backtrack_m: float = 0.5,
        tie_distance_m: float = 1.0,
        heading_gate_rad: float = math.radians(90.0),
    ) -> RouteProjection | None:
        """Nearest route position, monotonic once `prev_s` is known.

        Without `prev_s` (first fix or after a reset) the whole route is
        searched; candidates within `tie_distance_m` of the closest distance are
        disambiguated first by heading agreement (the route direction within
        `heading_gate_rad` of the vehicle yaw) and then by the SMALLER arc
        position, so a vehicle at the start line of a near-loop course is not
        projected onto the route's end.
        """

        n = self.count
        if prev_s is None:
            lo, hi = 0, n - 1
        else:
            lo = int(np.searchsorted(self.s, prev_s - back_window_m, side="left")) - 1
            hi = int(np.searchsorted(self.s, prev_s + forward_window_m, side="right"))
            lo = max(lo, 0)
            hi = min(max(hi, lo + 1), n - 1)
        a = self.xy[lo:hi]
        b = self.xy[lo + 1 : hi + 1]
        if len(a) == 0:
            return None
        dist, t, t_raw, cross = nearest_on_segments(np.array([[x, y]]), a, b)
        dist = dist[0]
        t = t[0]
        cross = cross[0]

        best = int(np.argmin(dist))
        if prev_s is None:
            # Candidates within tie_distance_m of the closest distance, grouped
            # into CLUSTERS of adjacent segments (one cluster = one pass of the
            # route near the vehicle). Adjacent segments are never a tie between
            # themselves; the closest one in each cluster represents it.
            near = np.flatnonzero(dist <= dist[best] + tie_distance_m)
            clusters: list[list[int]] = []
            for k in near.tolist():
                if clusters and k - clusters[-1][-1] <= 2:
                    clusters[-1].append(k)
                else:
                    clusters.append([k])
            reps = [min(c, key=lambda k: dist[k]) for c in clusters]
            chosen = None
            if yaw is not None and len(reps) > 1:
                consistent = []
                for k in reps:
                    d = b[k] - a[k]
                    route_yaw = math.atan2(d[1], d[0])
                    if abs(wrap_angle(route_yaw - yaw)) <= heading_gate_rad:
                        consistent.append(int(k))
                if consistent:
                    chosen = min(consistent)  # smallest arc position among consistent passes
            if chosen is None:
                chosen = min(reps) if len(reps) > 1 else best
            best = int(chosen)
        else:
            # Monotonic: do not accept a candidate further back than allowed.
            seg_s = self.s[lo + best] + t[best] * (self.s[lo + best + 1] - self.s[lo + best])
            if seg_s < prev_s - max_backtrack_m:
                allowed = [
                    k
                    for k in range(len(a))
                    if self.s[lo + k] + t[k] * (self.s[lo + k + 1] - self.s[lo + k])
                    >= prev_s - max_backtrack_m
                ]
                if not allowed:
                    return None
                best = min(allowed, key=lambda k: dist[k])

        seg_index = lo + best
        frac = float(t[best])
        s_here = float(self.s[seg_index] + frac * (self.s[seg_index + 1] - self.s[seg_index]))
        d = b[best] - a[best]
        seg_len = math.hypot(d[0], d[1])
        route_yaw = math.atan2(d[1], d[0])
        lateral = float(cross[best] / seg_len) if seg_len > 0 else 0.0
        foot = a[best] + frac * d
        heading_error = wrap_angle(route_yaw - yaw) if yaw is not None else 0.0
        return RouteProjection(
            s=s_here,
            segment_index=int(seg_index),
            fraction=frac,
            lateral_offset=lateral,
            heading_error=float(heading_error),
            foot_x=float(foot[0]),
            foot_y=float(foot[1]),
            distance=float(dist[best]),
        )
