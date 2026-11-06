import math

import pytest

from krowd_path_tracking.route_geometry import (
    back_at_route_start,
    travelled_arc_length,
)


WITHIN = 10.0
CORRIDOR = 3.0
CLEAR = 2.0


def _on_straight(x):
    """Segment index and fraction of a point at x on a 1 m-spaced +x route."""

    index = max(int(math.floor(x)), 0)
    return index, min(max(x - index, 0.0), 1.0)


def test_travelled_arc_length_sums_segments_and_the_fraction(straight_route):
    route = straight_route(10)
    assert travelled_arc_length(route, 0, 0.0) == pytest.approx(0.0)
    assert travelled_arc_length(route, 3, 0.5) == pytest.approx(3.5)
    assert travelled_arc_length(route, 20, 1.0) == pytest.approx(9.0)  # clamped


def test_a_restarted_drive_first_seen_several_metres_in_resumes(straight_route):
    """The RTK pose reappears ~5 m into a drive; 7 m along still counts."""

    route = straight_route(60)
    assert back_at_route_start(
        route, *_on_straight(7.0), 7.0, 0.5, WITHIN, CORRIDOR, CLEAR
    )


def test_further_along_than_the_restart_distance_stays_stopped(straight_route):
    route = straight_route(60)
    assert not back_at_route_start(
        route, *_on_straight(12.0), 12.0, 0.0, WITHIN, CORRIDOR, CLEAR
    )


def test_beside_the_start_but_outside_the_corridor_stays_stopped(straight_route):
    route = straight_route(60)
    assert not back_at_route_start(
        route, *_on_straight(2.0), 2.0, 4.0, WITHIN, CORRIDOR, CLEAR
    )


def test_behind_the_start_stays_stopped(straight_route):
    """Projection clamps to the first point; the vehicle is 15 m away from it."""

    route = straight_route(60)
    assert not back_at_route_start(route, 0, 0.0, -15.0, 0.0, WITHIN, CORRIDOR, CLEAR)


def test_a_loop_whose_end_touches_its_start_does_not_restart_where_it_stopped(
    circular_route,
):
    route = circular_route(10.0, 62)  # 61 m of a 62.8 m circle
    assert math.dist(route[0], route[-1]) < CLEAR
    x, y = route[0]
    assert not back_at_route_start(route, 0, 0.0, x, y, WITHIN, CORRIDOR, CLEAR)


def test_the_same_loop_restarts_once_clear_of_its_end_point(circular_route):
    route = circular_route(10.0, 62)
    x, y = route[4]
    assert math.dist((x, y), route[-1]) > CLEAR
    assert back_at_route_start(route, 4, 0.0, x, y, WITHIN, CORRIDOR, CLEAR)


def test_restart_distance_zero_disables_the_release(straight_route):
    route = straight_route(60)
    assert not back_at_route_start(
        route, *_on_straight(1.0), 1.0, 0.0, 0.0, CORRIDOR, CLEAR
    )
