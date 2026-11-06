import math
import os

import numpy as np
import pytest

from krowd_pp_stanleyff_control.route import RouteModel
from helpers import arc_route, straight_route


def test_arc_length_and_index_roundtrip():
    r = straight_route(50.0)
    assert r.length == pytest.approx(50.0)
    assert r.s_at_index(10) == pytest.approx(10.0)
    assert r.index_at_s(12.5) == pytest.approx(12.5)
    x, y = r.point_at_s(12.5)
    assert (x, y) == pytest.approx((12.5, 0.0))


def test_projection_signed_lateral_and_heading():
    r = straight_route(50.0)
    p = r.project(10.3, 0.7, 0.0, None)
    assert p.s == pytest.approx(10.3)
    assert p.lateral_offset == pytest.approx(0.7)  # left of the route is positive
    p = r.project(10.3, -0.7, 0.2, None)
    assert p.lateral_offset == pytest.approx(-0.7)
    assert p.heading_error == pytest.approx(-0.2)


def test_projection_is_monotonic_with_window():
    r = straight_route(100.0)
    p1 = r.project(40.0, 0.0, 0.0, None)
    # A localization glitch 5 m backwards must not move the progress back.
    p2 = r.project(35.0, 0.0, 0.0, p1.s, max_backtrack_m=0.5)
    assert p2.s >= p1.s - 0.5
    # But forward progress is followed.
    p3 = r.project(41.0, 0.0, 0.0, p1.s)
    assert p3.s == pytest.approx(41.0)


def test_near_loop_tie_break_prefers_start_by_heading():
    # A route whose end comes back next to its start, pointing the other way.
    n = 40
    fwd = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    back = np.column_stack([np.arange(n, dtype=float)[::-1], np.full(n, 1.5)])
    xy = np.vstack([fwd, back])
    r = RouteModel.from_arrays(xy, np.zeros(2 * n), np.zeros(2 * n), np.ones(2 * n))
    # Vehicle between the two passes, heading +x: start pass wins.
    p = r.project(5.0, 0.75, 0.0, None, tie_distance_m=1.0)
    assert p.s < 10.0
    # Heading -x: the return pass wins.
    p = r.project(5.0, 0.75, math.pi, None, tie_distance_m=1.0)
    assert p.s > r.length / 2


def test_point_ahead_and_curvature_lookup():
    r = arc_route()
    x, y, s_t, end = r.point_ahead(10.0, 7.0)
    assert (x, y) == pytest.approx((17.0, 0.0))
    assert not end
    assert r.max_abs_curvature(10.0, 17.0) == 0.0
    assert r.max_abs_curvature(35.0, 45.0) == pytest.approx(1.0 / 20.0)
    _, _, _, end = r.point_ahead(r.length - 1.0, 7.0)
    assert end


REAL = "/home/krowd/krowd_maps/20260913_103732/waypoints.csv"


@pytest.mark.skipif(not os.path.isfile(REAL), reason="competition map not present")
def test_real_map_mission_arc_lengths_match_document():
    import csv

    rows = []
    with open(REAL, newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append((float(row["utm_x"]), float(row["utm_y"]), float(row["yaw_rad"])))
    xy = np.array([(a, b) for a, b, _ in rows])
    r = RouteModel.from_arrays(xy, [c for _, _, c in rows], np.zeros(len(xy)), np.zeros(len(xy)))
    assert r.count == 586
    # Values from 자율주행_차량_필요값_분석결과 8.1 B (planar arc length, cm tolerance).
    assert r.s_at_index(51) == pytest.approx(88.606, abs=0.02)
    assert r.s_at_index(242) == pytest.approx(302.476, abs=0.02)
    assert r.s_at_index(480) == pytest.approx(588.275, abs=0.05)


def test_join_waypoint_is_the_nearest_to_the_given_point():
    route = straight_route(length_m=50.0, spacing=1.0)
    index, distance = route.join_waypoint(20.2, 3.0, tie_distance_m=0.0, pass_tie_distance_m=0.0)
    assert index == 20
    assert distance == pytest.approx(math.hypot(0.2, 3.0))


def test_join_waypoint_prefers_the_lower_index_among_equally_near_ones():
    route = straight_route(length_m=50.0, spacing=1.0)
    # 10 m to the side every waypoint within ~4.5 m along the route is within
    # 1 m of the nearest distance; the lowest of them wins.
    index, _ = route.join_waypoint(20.0, 10.0, tie_distance_m=1.0)
    assert index < 20
    nearest = math.hypot(0.0, 10.0)
    assert math.hypot(20.0 - route.xy[index, 0], 10.0) <= nearest + 1.0
    assert math.hypot(20.0 - route.xy[index - 1, 0], 10.0) > nearest + 1.0


def test_join_waypoint_takes_the_earlier_pass_of_a_near_loop():
    # Out along y=0, back along y=3: waypoint k and its twin on the way back
    # are 3 m apart. Halfway between them both passes are equally near.
    out = [(float(x), 0.0) for x in range(0, 31)]
    back = [(float(x), 3.0) for x in range(30, -1, -1)]
    xy = np.array(out + back)
    n = len(xy)
    route = RouteModel.from_arrays(xy, np.zeros(n), np.zeros(n), np.ones(n))
    index, _ = route.join_waypoint(10.0, 1.5, tie_distance_m=1.0)
    assert index < len(out)
    # Closer to the way back (y=2.5), heading east like the way OUT: the way
    # back runs against the vehicle and is dropped, the earlier pass wins.
    index, _ = route.join_waypoint(10.0, 2.5, yaw=0.0)
    assert index < len(out)
    # Heading west like the way BACK: the way out is dropped instead.
    index, _ = route.join_waypoint(10.0, 0.5, yaw=math.pi)
    assert index >= len(out)
