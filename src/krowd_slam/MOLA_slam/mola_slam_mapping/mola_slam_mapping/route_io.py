"""Validated route/corridor file loading and geometric projection helpers."""

import bisect
from dataclasses import dataclass
import csv
import math
from pathlib import Path

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMsg

from mola_slam_mapping.route_visualization import RoutePoint, make_route_point


def route_to_path(route: list[RoutePoint], frame_id: str, stamp) -> PathMsg:
    """The saved route as the standard nav_msgs/Path a follower consumes.

    The route is otherwise published only as a MarkerArray, whose coordinates
    live as float32 offsets from a per-marker anchor and carry no heading -
    a rendering format, and one free to change whenever the drawing does. This
    is the data interface: position AND the recorded `yaw_rad` as a quaternion,
    so a follower uses the route's own heading instead of differencing
    consecutive waypoints, which is noisy at metre spacing.

    Pose i is waypoint i - the same `idx` as in waypoints.csv and the same index
    the traffic-detect zones and the lookahead's route_position refer to.
    """

    path = PathMsg()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for point in route:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = point.x
        pose.pose.position.y = point.y
        pose.pose.position.z = point.z
        pose.pose.orientation.z = math.sin(point.yaw * 0.5)
        pose.pose.orientation.w = math.cos(point.yaw * 0.5)
        path.poses.append(pose)
    return path


@dataclass(frozen=True)
class LoadedRoute:
    points: list[RoutePoint]
    used_saved_corridor: bool
    utm_zone: str


@dataclass(frozen=True)
class RouteProjection:
    signed_offset: float
    right_limit: float
    left_limit: float
    within_longitudinal_extent: bool = True
    # Where along the route the projection landed: the segment (between
    # waypoints segment_index and segment_index + 1) and the clamped 0..1
    # fraction along it, so route_position is a waypoint-index coordinate.
    segment_index: int = 0
    fraction: float = 0.0
    # Distance from the vehicle to that segment (not the signed offset: past
    # either end of the route the two differ).
    distance_m: float = math.nan

    @property
    def route_position(self) -> float:
        return self.segment_index + self.fraction

    @property
    def on_track(self) -> bool:
        # Inputs transformed by tf2 may be NumPy scalar types. Chained
        # comparisons then produce numpy.bool_, which std_msgs/Bool rejects.
        return bool(
            self.within_longitudinal_extent
            and self.right_limit <= self.signed_offset <= self.left_limit
        )


@dataclass(frozen=True)
class LookaheadPoint:
    """The path-following target: a point on the route, ahead of the vehicle.

    Interpolated along the route rather than snapped to the nearest recorded
    waypoint. Snapping would make the target jump a whole waypoint spacing
    (~1 m) at a time and let its distance drift between ld and ld + spacing;
    interpolating keeps it at exactly ld and moving continuously, which is what
    a pure-pursuit controller wants to steer at.
    """

    x: float
    y: float
    z: float
    # Arc length actually walked. Less than requested only at the route's end,
    # where reached_route_end is set - a controller should notice that instead
    # of steering at a point that stopped moving away.
    distance_m: float
    segment_index: int
    reached_route_end: bool
    # Fractional waypoint index: 173.42 means "42% of the way from waypoint 173
    # to 174". Same coordinate RouteProjection.route_position uses, so it can be
    # compared directly against the traffic-detect zone ranges - i.e. "will I be
    # inside a zone 5 m from now?".
    route_position: float

    @property
    def nearest_waypoint_index(self) -> int:
        """The recorded waypoint this target is closest to, for display and for
        joining back to waypoints.csv / corridor.csv by `idx`."""

        return int(round(self.route_position))


def point_ahead_on_route(
    route: list[RoutePoint], projection: "RouteProjection | None", distance_m: float
) -> LookaheadPoint | None:
    """Walk `distance_m` along the route from where the vehicle projects onto it.

    Arc length along the polyline, not a straight-line radius: through a corner
    a Euclidean lookahead of the same size cuts the corner and sits off the
    route, while this stays on it. At the end of the route it clamps to the last
    waypoint rather than extrapolating past what was actually mapped, and says
    so via reached_route_end.
    """

    if projection is None or len(route) < 2 or not math.isfinite(distance_m):
        return None
    if distance_m < 0.0:
        raise ValueError(f"lookahead distance must not be negative: {distance_m}")

    remaining = float(distance_m)
    travelled = 0.0
    index = projection.segment_index
    fraction = projection.fraction
    while index < len(route) - 1:
        current, following = route[index], route[index + 1]
        dx = following.x - current.x
        dy = following.y - current.y
        dz = following.z - current.z
        # Planar length: the route is driven on the ground and z just follows.
        length = math.hypot(dx, dy)
        available = length * (1.0 - fraction)
        if available >= remaining:
            end = fraction + (remaining / length if length > 0.0 else 0.0)
            return LookaheadPoint(
                x=current.x + end * dx,
                y=current.y + end * dy,
                z=current.z + end * dz,
                distance_m=travelled + remaining,
                segment_index=index,
                reached_route_end=False,
                route_position=index + end,
            )
        remaining -= available
        travelled += available
        fraction = 0.0
        index += 1

    last = route[-1]
    return LookaheadPoint(
        x=last.x,
        y=last.y,
        z=last.z,
        distance_m=travelled,
        segment_index=len(route) - 2,
        reached_route_end=True,
        route_position=float(len(route) - 1),
    )


def off_track_reason(projection: "RouteProjection | None") -> str | None:
    """Why `on_track` is false, in the operator's terms; None when on track.

    Two very different things clear that flag and they used to report the same
    way: a real corridor violation, and simply not having reached the recorded
    route yet (or having driven past its end). The second is normal at the start
    of a run - "0.08 m off the centerline (outside the corridor boundaries)" on a
    3 m half-width corridor reads as a bug and is not one.
    """

    if projection is None:
        return "no saved route segment to compare against"
    if projection.on_track:
        return None
    if not projection.within_longitudinal_extent:
        before_start = projection.segment_index == 0 and projection.fraction <= 0.0
        end = "before the first" if before_start else "past the last"
        return (
            f"{end} waypoint of the saved route, "
            f"{abs(projection.signed_offset):.2f} m off its centerline "
            "(not a corridor violation)"
        )
    return (
        f"{abs(projection.signed_offset):.2f} m off the saved centerline, "
        f"outside the corridor ({projection.right_limit:.2f} .. "
        f"{projection.left_limit:.2f} m)"
    )


def _finite(row: dict, key: str, default=None) -> float:
    raw = row.get(key)
    if (raw is None or raw == "") and default is not None:
        value = float(default)
    else:
        value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"non-finite value for '{key}'")
    return value


def load_route_files(
    waypoints_path: Path,
    corridor_path: Path | None,
    fallback_half_width: float,
) -> LoadedRoute:
    """Load the globally referenced route and join corridor rows by idx."""

    points: list[RoutePoint] = []
    waypoint_indices: list[int] = []
    zones: set[str] = set()
    with open(waypoints_path, newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream)):
            index = int(row.get("idx") or row_number)
            if index in waypoint_indices:
                raise ValueError(f"duplicate waypoint idx {index}")
            waypoint_indices.append(index)
            zone = str(row.get("utm_zone") or "")
            if zone:
                zones.add(zone)
            points.append(
                make_route_point(
                    _finite(row, "utm_x"),
                    _finite(row, "utm_y"),
                    _finite(row, "utm_z", 0.0),
                    _finite(row, "yaw_rad"),
                    fallback_half_width,
                )
            )

    if len(points) < 2:
        raise ValueError(
            f"waypoints_csv '{waypoints_path}' has fewer than 2 points"
        )
    if len(zones) > 1:
        raise ValueError(f"waypoints_csv contains multiple UTM zones: {sorted(zones)}")
    utm_zone = next(iter(zones), "")

    if corridor_path is None or not corridor_path.exists():
        return LoadedRoute(points, False, utm_zone)

    corridor_by_index: dict[int, dict] = {}
    with open(corridor_path, newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream)):
            index = int(row.get("idx") or row_number)
            if index in corridor_by_index:
                raise ValueError(f"duplicate corridor idx {index}")
            zone = str(row.get("utm_zone") or "")
            if zone and utm_zone and zone != utm_zone:
                raise ValueError(
                    f"corridor idx {index} uses UTM zone '{zone}', expected '{utm_zone}'"
                )
            corridor_by_index[index] = row

    if set(corridor_by_index) != set(waypoint_indices):
        missing = sorted(set(waypoint_indices) - set(corridor_by_index))
        extra = sorted(set(corridor_by_index) - set(waypoint_indices))
        raise ValueError(
            f"corridor indices do not match waypoints (missing={missing}, extra={extra})"
        )

    combined = []
    for point, index in zip(points, waypoint_indices):
        row = corridor_by_index[index]
        # Validate optional z fields too, even though RoutePoint stores one center z.
        _finite(row, "left_z", point.z)
        _finite(row, "right_z", point.z)
        combined.append(
            RoutePoint(
                x=point.x,
                y=point.y,
                z=point.z,
                yaw=point.yaw,
                left_x=_finite(row, "left_x"),
                left_y=_finite(row, "left_y"),
                right_x=_finite(row, "right_x"),
                right_y=_finite(row, "right_y"),
            )
        )
    return LoadedRoute(combined, True, utm_zone)


def nearest_waypoint_id(projection: "RouteProjection | None") -> int:
    """waypoints.csv idx of the waypoint nearest the projected vehicle; -1 without one.

    The projection lands on the nearest route segment, between waypoints
    segment_index and segment_index + 1; rounding its waypoint-index position
    picks whichever end of that segment is closer along it.
    """

    if projection is None:
        return -1
    return int(round(projection.route_position))


def project_onto_route(
    route: list[RoutePoint],
    x: float,
    y: float,
    first_segment: int = 0,
    last_segment: int | None = None,
) -> RouteProjection | None:
    """Project onto the nearest center segment and interpolate saved boundaries.

    Only segments first_segment..last_segment (inclusive, waypoint indices of
    their start) are searched; by default the whole route. RouteTracker uses
    the range to keep a vehicle on the pass of the route it is driving.
    """

    best_distance_sq = math.inf
    best_projection = None
    last_segment_index = len(route) - 2
    first = max(int(first_segment), 0)
    last = last_segment_index if last_segment is None else min(int(last_segment), last_segment_index)
    for segment_index in range(first, last + 1):
        current = route[segment_index]
        following = route[segment_index + 1]
        dx = following.x - current.x
        dy = following.y - current.y
        segment_length_sq = dx * dx + dy * dy
        if segment_length_sq < 1e-9:
            continue

        raw_fraction = (
            (x - current.x) * dx + (y - current.y) * dy
        ) / segment_length_sq
        fraction = max(0.0, min(1.0, raw_fraction))
        center_x = current.x + fraction * dx
        center_y = current.y + fraction * dy
        distance_sq = (x - center_x) ** 2 + (y - center_y) ** 2
        if distance_sq >= best_distance_sq:
            continue

        segment_length = math.sqrt(segment_length_sq)
        normal_x = -dy / segment_length
        normal_y = dx / segment_length
        signed_offset = (
            (x - center_x) * normal_x + (y - center_y) * normal_y
        )
        left_x = current.left_x + fraction * (
            following.left_x - current.left_x
        )
        left_y = current.left_y + fraction * (
            following.left_y - current.left_y
        )
        right_x = current.right_x + fraction * (
            following.right_x - current.right_x
        )
        right_y = current.right_y + fraction * (
            following.right_y - current.right_y
        )
        boundary_offsets = (
            (left_x - center_x) * normal_x + (left_y - center_y) * normal_y,
            (right_x - center_x) * normal_x + (right_y - center_y) * normal_y,
        )
        best_distance_sq = distance_sq
        best_projection = RouteProjection(
            signed_offset=signed_offset,
            right_limit=min(boundary_offsets),
            left_limit=max(boundary_offsets),
            within_longitudinal_extent=not (
                (segment_index == 0 and raw_fraction < 0.0)
                or (
                    segment_index == last_segment_index
                    and raw_fraction > 1.0
                )
            ),
            segment_index=segment_index,
            fraction=float(fraction),
            distance_m=math.sqrt(distance_sq),
        )

    return best_projection


def _segment_distance(route: list[RoutePoint], segment_index: int, x: float, y: float) -> float:
    current = route[segment_index]
    following = route[segment_index + 1]
    dx = following.x - current.x
    dy = following.y - current.y
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-9:
        return math.inf
    fraction = max(0.0, min(1.0, ((x - current.x) * dx + (y - current.y) * dy) / length_sq))
    return math.hypot(x - current.x - fraction * dx, y - current.y - fraction * dy)


class RouteTracker:
    """The vehicle's route position with continuity from one pose to the next.

    A route can pass the same place twice: the preliminary course crosses
    itself (waypoints 44-45 x 253-254), and the final course's two-way road
    runs its out and back legs 4.6-4.9 m apart. A nearest-segment search over
    the whole route puts the dividing line exactly halfway between two passes,
    so a vehicle that leaves its own pass by more than that - a cone avoidance
    into the opposite lane, localization noise at a crossing - was reported on
    the OTHER pass, and every zone flipped with it (/avoidance_space false in
    the middle of an avoidance).

    Once a position is known, only the segments from back_m behind to
    forward_m ahead of it (arc length) are searched. The whole route is
    searched again on the first pose, after reset(), and whenever the nearest
    segment inside that window is farther than relocate_distance_m (a pose
    jump, a relocalization) or points more than heading_gate_rad away from the
    vehicle yaw (a correction onto the opposite leg of a two-way road, which is
    closer than relocate_distance_m). On a whole-route search, passes within tie_m of
    the closest one are told apart by heading: the pass whose direction is
    within heading_gate_rad of the vehicle yaw wins, the earliest such pass on
    a tie - the same rule as the PP/Stanley controllers' route projection.
    """

    def __init__(
        self,
        route: list[RoutePoint],
        back_m: float = 8.0,
        forward_m: float = 30.0,
        relocate_distance_m: float = 5.0,
        tie_m: float = 1.0,
        heading_gate_rad: float = math.radians(90.0),
    ):
        self.route = route
        self.back_m = float(back_m)
        self.forward_m = float(forward_m)
        self.relocate_distance_m = float(relocate_distance_m)
        self.tie_m = float(tie_m)
        self.heading_gate_rad = float(heading_gate_rad)
        self.arc = [0.0]
        for current, following in zip(route, route[1:]):
            self.arc.append(self.arc[-1] + math.hypot(following.x - current.x, following.y - current.y))
        self.projection: RouteProjection | None = None
        self.relocated = False  # the last project() searched the whole route after having a position

    def reset(self) -> None:
        self.projection = None

    def _arc_at(self, route_position: float) -> float:
        index = min(int(route_position), len(self.arc) - 2)
        fraction = route_position - index
        return self.arc[index] + fraction * (self.arc[index + 1] - self.arc[index])

    def _window(self) -> tuple[int, int]:
        s = self._arc_at(self.projection.route_position)
        first = bisect.bisect_left(self.arc, s - self.back_m) - 1
        last = bisect.bisect_right(self.arc, s + self.forward_m) - 1
        return max(first, 0), max(min(last, len(self.route) - 2), max(first, 0))

    def _heading_error(self, segment_index: int, yaw: float | None) -> float:
        """|segment direction - yaw|; 0 without a yaw."""

        if yaw is None or not math.isfinite(yaw):
            return 0.0
        current, following = self.route[segment_index], self.route[segment_index + 1]
        direction = math.atan2(following.y - current.y, following.x - current.x)
        return abs(math.atan2(math.sin(direction - yaw), math.cos(direction - yaw)))

    def _whole_route(self, x: float, y: float, yaw: float | None) -> RouteProjection | None:
        distances = [_segment_distance(self.route, k, x, y) for k in range(len(self.route) - 1)]
        if not distances or not math.isfinite(min(distances)):
            return None
        best = min(range(len(distances)), key=distances.__getitem__)
        if yaw is not None and math.isfinite(yaw):
            # One candidate per pass: runs of adjacent near segments are one pass.
            near = [k for k, d in enumerate(distances) if d <= distances[best] + self.tie_m]
            passes: list[list[int]] = []
            for k in near:
                if passes and k - passes[-1][-1] <= 2:
                    passes[-1].append(k)
                else:
                    passes.append([k])
            if len(passes) > 1:
                representatives = [min(run, key=distances.__getitem__) for run in passes]
                consistent = [
                    k for k in representatives
                    if self._heading_error(k, yaw) <= self.heading_gate_rad
                ]
                if consistent:
                    best = min(consistent)
        return project_onto_route(self.route, x, y, best, best)

    def project(self, x: float, y: float, yaw: float | None = None) -> RouteProjection | None:
        self.relocated = False
        projection = None
        if self.projection is not None:
            first, last = self._window()
            projection = project_onto_route(self.route, x, y, first, last)
            if (
                projection is None
                or projection.distance_m > self.relocate_distance_m
                or self._heading_error(projection.segment_index, yaw) > self.heading_gate_rad
            ):
                projection = None
                self.relocated = True
        if projection is None:
            projection = self._whole_route(x, y, yaw)
        self.projection = projection
        return projection
