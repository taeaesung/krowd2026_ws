"""Planar geometry of a polyline route: lengths, curvature, arc positions.

Takes and returns plain (x, y) tuples rather than RoutePoint so the maths stays
testable without ROS or the mapping package, and so the same functions work on
a Trajectory's poses as on a freshly loaded waypoints.csv.
"""

import math

Point = tuple[float, float]


def segment_lengths(points: list[Point]) -> list[float]:
    """Planar length of each segment; one shorter than `points`.

    Planar because the vehicle drives on the ground and the recorded z only
    follows the terrain: including it would inflate every distance on a slope
    and, through the speed profile, quietly raise the cornering speeds.
    """

    return [
        math.hypot(following[0] - current[0], following[1] - current[1])
        for current, following in zip(points, points[1:])
    ]


def path_curvatures(points: list[Point], smoothing_window: int = 5) -> list[float]:
    """Signed curvature (1/m) at each point, positive to the left.

    Menger curvature of each consecutive triple - the circle through three
    points, k = 2 * cross / (a b c). The endpoints copy their neighbour because
    a triple does not exist there.

    Smoothing is not optional in practice. These waypoints are a RECORDED drive
    sampled every metre, so they carry the driver's line and the localization's
    per-sample noise; three-point curvature differentiates twice and amplifies
    both. Raw, it produces speed profiles that brake for noise on a straight.
    """

    count = len(points)
    if count < 3:
        return [0.0] * count

    raw = [0.0] * count
    for index in range(1, count - 1):
        previous, current, following = (
            points[index - 1],
            points[index],
            points[index + 1],
        )
        first_x, first_y = current[0] - previous[0], current[1] - previous[1]
        second_x, second_y = following[0] - current[0], following[1] - current[1]
        cross = first_x * second_y - first_y * second_x
        a = math.hypot(first_x, first_y)
        b = math.hypot(second_x, second_y)
        c = math.hypot(following[0] - previous[0], following[1] - previous[1])
        denominator = a * b * c
        raw[index] = 2.0 * cross / denominator if denominator > 1e-12 else 0.0
    raw[0] = raw[1]
    raw[-1] = raw[-2]

    return _moving_average(raw, smoothing_window)


def _moving_average(values: list[float], window: int) -> list[float]:
    """Centred moving average, shrinking the window at the ends.

    Shrinking rather than padding: padding with the edge value would flatten the
    curvature of the first and last few metres, which is exactly where the route
    starts and stops and where the speed profile has the least room to recover.
    """

    if window <= 1:
        return list(values)
    half = window // 2
    count = len(values)
    smoothed = []
    for index in range(count):
        first = max(0, index - half)
        last = min(count, index + half + 1)
        smoothed.append(sum(values[first:last]) / (last - first))
    return smoothed


def forward_progress_indices(
    points: list[Point], max_reversal_deg: float = 100.0, min_step_m: float = 0.05
) -> list[int]:
    """Indices of the waypoints that keep making forward progress.

    A saved route is a RECORDED drive, and localization can jump backwards
    during one: waypoint_saver_node samples every metre of travel but does not
    check the DIRECTION of that metre, so a pose that jumps back and recovers
    is written as a waypoint pair that doubles back and returns. The real
    20260907_114123 route has exactly one such excursion (waypoints 224-225
    re-tread 222-223), and it alone produces a 0.56 m radius corner - against a
    vehicle whose minimum is 3.4 m.

    That is not a smoothing problem. Arc-length lookahead walks the polyline it
    is given, so a doubled-back route makes the target run backwards through the
    excursion, and projection onto it is ambiguous between the two passes.

    Rejects a candidate that turns more than `max_reversal_deg` away from the
    direction actually being travelled, and one that barely moves at all
    (`min_step_m`), which is the second half of the same excursion.
    """

    if len(points) < 2:
        return list(range(len(points)))

    kept = [0]
    heading: float | None = None
    for index in range(1, len(points)):
        last = points[kept[-1]]
        step_x = points[index][0] - last[0]
        step_y = points[index][1] - last[1]
        if math.hypot(step_x, step_y) < min_step_m:
            continue
        step_heading = math.atan2(step_y, step_x)
        if heading is not None and max_reversal_deg > 0.0:
            difference = math.atan2(
                math.sin(step_heading - heading), math.cos(step_heading - heading)
            )
            if abs(math.degrees(difference)) > max_reversal_deg:
                continue
        kept.append(index)
        heading = step_heading
    return kept


def loop_closure(
    points: list[Point],
    max_gap_m: float,
    search_m: float = 30.0,
    max_turn_deg: float = 60.0,
    min_gap_m: float = 0.2,
) -> tuple[int, int] | None:
    """Where a recorded loop can be joined back to itself, as (head, tail).

    points[head..tail] followed by a segment from tail back to head is the loop.
    None when the route is not one.

    Joining the last waypoint straight back to the first is wrong for these
    recordings: the mapping drive rarely stops where it started. It stops a few
    metres SHORT of the start (20260914_220126: 5.3 m) or drives on PAST it,
    alongside the first waypoints (20260913_101241 overlaps its start by ~17 m
    and a straight end->start join points backwards, 161 deg off the start's
    direction). So every head waypoint in the first `search_m` of the route is
    tried against every tail waypoint in its last `search_m`, and the shortest
    join that neither leaves the tail nor enters the head by more than
    `max_turn_deg` wins. Joins under 1 m are checked only on the tail and head
    directions: over that distance the join's own direction is the lateral
    offset between the two passes, not a turn.

    `min_gap_m` keeps a join from degenerating into a zero-length segment when
    a waypoint of one pass lands on one of the other.
    """

    count = len(points)
    if max_gap_m <= 0.0 or count < 4:
        return None
    cumulative = [0.0]
    for length in segment_lengths(points):
        cumulative.append(cumulative[-1] + length)
    total = cumulative[-1]
    if total <= 2.0 * search_m:
        # Head and tail windows would overlap: a short route folds onto itself.
        return None

    def heading(start: Point, end: Point) -> float:
        return math.atan2(end[1] - start[1], end[0] - start[0])

    def turn(first: float, second: float) -> float:
        difference = math.atan2(math.sin(second - first), math.cos(second - first))
        return abs(math.degrees(difference))

    heads = [index for index in range(count - 1) if cumulative[index] <= search_m]
    tails = [index for index in range(1, count) if total - cumulative[index] <= search_m]
    best: tuple[float, int, int] | None = None
    for tail in tails:
        tail_heading = heading(points[tail - 1], points[tail])
        for head in heads:
            gap = math.hypot(
                points[head][0] - points[tail][0], points[head][1] - points[tail][1]
            )
            if not min_gap_m <= gap <= max_gap_m:
                continue
            if best is not None and gap >= best[0]:
                continue
            head_heading = heading(points[head], points[head + 1])
            if gap >= 1.0:
                join = heading(points[tail], points[head])
                if (
                    turn(tail_heading, join) > max_turn_deg
                    or turn(join, head_heading) > max_turn_deg
                ):
                    continue
            elif turn(tail_heading, head_heading) > max_turn_deg:
                continue
            best = (gap, head, tail)
    return None if best is None else (best[1], best[2])


def unroll_loop(values: list, laps: int) -> list:
    """A closed loop's per-waypoint values (last == first) repeated `laps` times.

    Consecutive laps share their seam waypoint, so lap k starts at index
    k * (len(values) - 1). Walking forwards past the end of lap k simply
    continues into lap k + 1, which is how the lookahead, the heading window,
    the curvature smoothing and the speed profile all see across the seam
    without learning about wrap-around themselves.
    """

    if laps < 1:
        raise ValueError(f"laps must be at least 1: {laps}")
    return list(values) + list(values[1:]) * (laps - 1)


def max_abs_value_ahead(
    points: list[Point],
    values: list[float],
    segment_index: int,
    fraction: float,
    distance_m: float,
) -> float:
    """Largest |value| on the waypoints within distance_m ahead of a position.

    Includes the segment the position is on (both its ends), so a corner the
    vehicle is already in counts, and walks forwards by arc length from there.
    Used with the per-waypoint curvatures to see how tight the route gets
    within the lookahead before the target point is placed.
    """

    count = min(len(points), len(values))
    if count == 0:
        return 0.0
    index = min(max(segment_index, 0), count - 1)
    largest = abs(values[index])
    travelled = -max(0.0, min(1.0, fraction)) * (
        math.hypot(
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        )
        if index + 1 < count
        else 0.0
    )
    while index + 1 < count and travelled < distance_m:
        travelled += math.hypot(
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        )
        index += 1
        largest = max(largest, abs(values[index]))
    return largest


def remaining_arc_length(
    points: list[Point], segment_index: int, fraction: float
) -> float:
    """Distance still to drive along the route from a projected position."""

    total = 0.0
    index = max(0, segment_index)
    carried_fraction = min(max(fraction, 0.0), 1.0)
    while index < len(points) - 1:
        length = math.hypot(
            points[index + 1][0] - points[index][0],
            points[index + 1][1] - points[index][1],
        )
        total += length * (1.0 - carried_fraction)
        carried_fraction = 0.0
        index += 1
    return total


def value_at_route_position(
    values: list[float], segment_index: int, fraction: float
) -> float:
    """Linearly interpolate a per-waypoint quantity at a fractional index.

    Snapping to the nearest waypoint instead would make the commanded speed step
    by a whole profile increment every metre driven, which the drivetrain reads
    as a series of small kicks.
    """

    if not values:
        raise ValueError("no values to interpolate")
    if len(values) == 1:
        return values[0]
    index = min(max(segment_index, 0), len(values) - 2)
    clamped_fraction = min(max(fraction, 0.0), 1.0)
    return values[index] + clamped_fraction * (values[index + 1] - values[index])


def travelled_arc_length(
    points: list[Point], segment_index: int, fraction: float
) -> float:
    """Distance along the route from its first point to a projected position."""

    last_segment = len(points) - 2
    if last_segment < 0:
        return 0.0
    index = min(max(segment_index, 0), last_segment)
    total = 0.0
    for i in range(index):
        total += math.hypot(
            points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]
        )
    length = math.hypot(
        points[index + 1][0] - points[index][0],
        points[index + 1][1] - points[index][1],
    )
    return total + length * min(max(fraction, 0.0), 1.0)


def back_at_route_start(
    points: list[Point],
    segment_index: int,
    fraction: float,
    x: float,
    y: float,
    within_m: float,
    max_offset_m: float,
    clear_of_end_m: float,
) -> bool:
    """Whether a vehicle that stopped at the route's end is at its start again.

    Releases a goal latch: once STOP_AT_END has stopped the vehicle, tracking
    resumes only when it is back within `within_m` of the start along the route,
    no further than `max_offset_m` from the route there, and more than
    `clear_of_end_m` from the route's final point. The last condition keeps a
    closed loop whose end lies next to its start from restarting where it
    stopped. `within_m` has to exceed the driving gnss_imu_pose_node needs
    before its pose reappears (~5 m): a restarted drive is first seen that far
    along already. within_m <= 0 disables the release.
    """

    if within_m <= 0.0 or len(points) < 2:
        return False
    if travelled_arc_length(points, segment_index, fraction) > within_m:
        return False
    index = min(max(segment_index, 0), len(points) - 2)
    t = min(max(fraction, 0.0), 1.0)
    foot_x = points[index][0] + t * (points[index + 1][0] - points[index][0])
    foot_y = points[index][1] + t * (points[index + 1][1] - points[index][1])
    if math.hypot(x - foot_x, y - foot_y) > max_offset_m:
        return False
    end_x, end_y = points[-1]
    return math.hypot(x - end_x, y - end_y) > clear_of_end_m
