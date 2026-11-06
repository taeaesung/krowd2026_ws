import math

import pytest

from mola_slam_mapping.route_io import (
    RouteTracker,
    nearest_waypoint_id,
    project_onto_route,
)
from mola_slam_mapping.route_visualization import make_route_point


def _polyline(points):
    """Waypoints ~1 m apart along the given corner points."""

    route = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        steps = max(int(round(math.hypot(x1 - x0, y1 - y0))), 1)
        yaw = math.atan2(y1 - y0, x1 - x0)
        for k in range(steps):
            f = k / steps
            route.append(make_route_point(x0 + f * (x1 - x0), y0 + f * (y1 - y0), 0.0, yaw, 3.0))
    x, y = points[-1]
    route.append(make_route_point(x, y, 0.0, route[-1].yaw, 3.0))
    return route


def _two_way_road(gap=4.8):
    """Out along +x for 40 m, a U-turn, back along -x `gap` to the left."""

    return _polyline([(0.0, 0.0), (40.0, 0.0), (44.0, gap / 2), (40.0, gap), (0.0, gap)])


def _drive(tracker, route, start, end, lateral=0.0):
    """Positions along waypoints start..end, `lateral` metres to the left."""

    ids = []
    for i in range(start, end):
        a, b = route[i], route[i + 1]
        length = math.hypot(b.x - a.x, b.y - a.y)
        nx, ny = -(b.y - a.y) / length, (b.x - a.x) / length
        for f in (0.0, 0.5):
            x = a.x + f * (b.x - a.x) + nx * lateral
            y = a.y + f * (b.y - a.y) + ny * lateral
            ids.append(nearest_waypoint_id(tracker.project(x, y, a.yaw)))
    return ids


def test_a_vehicle_in_the_opposite_lane_stays_on_its_own_pass():
    route = _two_way_road(gap=4.8)
    back_start = next(i for i, p in enumerate(route) if p.x < 39.0 and p.y > 4.0)

    # The old whole-route nearest search hands it to the other leg past halfway.
    a = route[back_start + 10]
    assert nearest_waypoint_id(project_onto_route(route, a.x, a.y - 3.4)) < 40

    tracker = RouteTracker(route)
    _drive(tracker, route, back_start, back_start + 10)
    ids = _drive(tracker, route, back_start + 10, back_start + 25, lateral=3.4)
    assert min(ids) >= back_start + 10
    assert ids == sorted(ids)


def test_a_crossing_is_passed_on_the_pass_being_driven():
    # A figure that crosses itself at right angles at (10, 0).
    route = _polyline([(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0), (10.0, -10.0)])
    crossing_second = next(i for i, p in enumerate(route) if i > 30 and p.x == 10.0 and abs(p.y) < 0.5)

    tracker = RouteTracker(route)
    first = _drive(tracker, route, 0, 18, lateral=0.15)
    assert max(first) <= 18
    second = _drive(tracker, route, 18, len(route) - 1, lateral=-0.15)
    assert min(second) >= 18
    assert crossing_second in second


def test_the_first_pose_is_placed_on_the_pass_matching_its_heading():
    route = _two_way_road(gap=1.5)
    back_start = next(i for i, p in enumerate(route) if p.x < 39.0 and p.y > 1.0)
    target = route[back_start + 15]

    # Between the legs, nearer the outbound one, but heading -x like the return leg.
    tracker = RouteTracker(route)
    projection = tracker.project(target.x, 0.6, math.pi)
    assert nearest_waypoint_id(projection) >= back_start

    tracker = RouteTracker(route)
    assert nearest_waypoint_id(tracker.project(target.x, 0.6, 0.0)) < 40


def test_a_pose_jump_searches_the_whole_route_again():
    route = _two_way_road(gap=12.0)
    tracker = RouteTracker(route, relocate_distance_m=5.0)
    _drive(tracker, route, 0, 10)
    assert not tracker.relocated

    far = route[-5]
    assert nearest_waypoint_id(tracker.project(far.x, far.y, far.yaw)) >= len(route) - 6
    assert tracker.relocated
    tracker.project(far.x - 1.0, far.y, far.yaw)
    assert not tracker.relocated


def test_a_correction_onto_the_opposite_leg_is_followed_by_heading():
    # The legs are closer than relocate_distance_m, so only the heading tells.
    route = _two_way_road(gap=4.8)
    tracker = RouteTracker(route, relocate_distance_m=5.0)
    _drive(tracker, route, 0, 10)
    back = route[-5]
    assert nearest_waypoint_id(tracker.project(back.x, back.y, back.yaw)) >= len(route) - 6
    assert tracker.relocated

    # A swerve of 40 degrees on its own leg is not a correction.
    tracker = RouteTracker(route)
    _drive(tracker, route, 0, 10)
    assert nearest_waypoint_id(tracker.project(10.5, 1.0, math.radians(40.0))) <= 12
    assert not tracker.relocated


def test_segment_range_limits_the_search_and_reports_distance():
    route = _two_way_road(gap=4.8)
    whole = project_onto_route(route, 10.0, 4.0)
    assert whole.distance_m == pytest.approx(0.8)
    limited = project_onto_route(route, 10.0, 4.0, 0, 30)
    assert limited.segment_index <= 30
    assert limited.distance_m == pytest.approx(4.0)
