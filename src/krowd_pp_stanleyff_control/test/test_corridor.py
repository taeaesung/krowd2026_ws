import math
from dataclasses import replace
import os

import numpy as np
import pytest

from krowd_pp_stanleyff_control.corridor import (
    BoundarySet,
    CorridorSupervisor,
    StaticCorridor,
    SupervisorParams,
    load_boundaries,
    route_boundaries,
    validate_boundaries,
)
from krowd_pp_stanleyff_control.route import RouteModel
from helpers import arc_route, straight_route


def straight_bounds(length=120.0, half=2.0, start=0.0):
    xs = np.arange(start, length + 1.0, 1.0)
    left = np.column_stack([xs, np.full_like(xs, half)])
    right = np.column_stack([xs, np.full_like(xs, -half)])
    return BoundarySet(left, right, [], "synthetic", "leftBoundary", "rightBoundary")


def params(inset=0.628 + 0.4):
    return SupervisorParams(
        wheelbase_m=1.34, front_m=2.534, rear_m=0.0, half_width_m=0.628, inset_m=inset,
        sim_dt_s=0.1, min_horizon_s=1.0, max_horizon_s=3.0,
    )


def test_validate_synthetic_and_clearance_sign():
    b = straight_bounds()
    info = validate_boundaries(b)
    assert info["width_min_m"] == pytest.approx(4.0)
    c = StaticCorridor(b)
    c.locate(10.0, 0.0)
    cl = c.clearance(np.array([[10.0, 0.5]]))
    assert cl.left_min == pytest.approx(1.5)
    assert cl.right_min == pytest.approx(2.5)
    assert cl.covered
    outside = c.clearance(np.array([[10.0, 2.5]]))
    assert outside.left_min < 0
    beyond = StaticCorridor(b)
    beyond.locate(-3.0, 0.0)
    assert not beyond.clearance(np.array([[-3.0, 0.0]])).covered
    # Per-point coverage: one point inside, one beyond the end.
    left, right, covered = beyond.clearance_points(np.array([[10.0, 0.0], [-3.0, 0.0]]))
    assert covered.tolist() == [True, False]


def test_supervisor_accepts_centered_straight_driving():
    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    d = sup.evaluate(route, 20.0, 0.0, 0.0, 3.0, 20.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.reason == "OK" and not d.outside
    assert d.predicted_min_clearance > 0
    assert d.fallback_fraction == 0.0 and d.candidates_tried == 1 == len(d.rollouts)


def test_supervisor_replaces_steering_that_hits_a_wall():
    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    # Nominal command steers hard left from 0.6 m left of centre.
    d = sup.evaluate(route, 20.0, 0.6, 0.0, 3.0, 20.0, math.radians(15.0), math.radians(15.0), (-0.35, 0.4), 2.0)
    assert d.ok
    assert d.steering_rad < math.radians(15.0)
    assert d.reason.startswith("CANDIDATE")
    # Every rollout tried is reported, the nominal first and the chosen one last.
    assert d.candidates_tried == len(d.rollouts) > 1
    assert d.rollouts[0].steering_rad == pytest.approx(math.radians(15.0)) and not d.rollouts[0].ok
    assert d.rollouts[-1].ok and d.rollouts[-1].steering_rad == pytest.approx(d.steering_rad)
    assert d.rollouts[-1].poses == d.predicted_poses


def test_partly_over_a_wall_is_inside_and_steers_away_at_speed_zero():
    """Two corners over the left wall: the footprint is still "inside" (not all
    four corners are over), so the candidates are evaluated. Every one fails
    on its first step - the corner is already over - so the verdict is
    NO_SAFE_STEERING_CANDIDATE with the candidate that gains the most
    clearance: the hardest steer AWAY from the wall. The node then commands
    speed 0 without the brake; nothing here asks for a stop."""

    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    d = sup.evaluate(route, 20.0, 1.6, 0.0, 3.0, 20.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert not d.ok and not d.outside and d.reason == "NO_SAFE_STEERING_CANDIDATE"
    assert d.steering_rad == pytest.approx(math.radians(-15.0))
    assert d.candidates_tried == len(d.rollouts) == 31  # nominal + 15 each side, no budget


def test_wholly_outside_the_lane_is_lane_outside_with_the_nominal_steering():
    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    d = sup.evaluate(route, 20.0, 3.0, 0.0, 3.0, 20.0, math.radians(-8.0), 0.0, (-0.35, 0.4), 1.5)
    assert not d.ok and d.outside and d.reason == "LANE_OUTSIDE"
    assert d.steering_rad == pytest.approx(math.radians(-8.0))
    assert d.current_clearance < 0 and d.rollouts == []


def test_without_a_fallback_leaving_the_lane_map_is_free():
    """Past the end of the lane map nothing covers the points: no wall, no hit."""

    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    d = sup.evaluate(route, 118.0, 0.0, 0.0, 3.0, 118.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.reason == "OK" and not d.covered


def test_route_corridor_takes_over_where_the_lane_map_ends():
    """Per-point fallback: the lane map covers x <= 120 m, the route corridor the rest."""

    route = straight_route(150.0)
    fallback = StaticCorridor(route_boundaries(route.xy, 3.0), 10.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params(), fallback=fallback)
    d = sup.evaluate(route, 118.0, 0.0, 0.0, 3.0, 118.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.reason == "OK" and d.covered
    assert 0.0 < d.fallback_fraction < 1.0
    # Wholly outside the lane map: everything comes from the route corridor.
    d = sup.evaluate(route, 130.0, 0.0, 0.0, 3.0, 130.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.fallback_fraction == pytest.approx(1.0)
    # The route corridor is +/-3 m: two corners 2.6 m off the line are over it.
    d = sup.evaluate(route, 130.0, 2.6, 0.0, 3.0, 130.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert not d.ok and not d.outside and d.reason == "NO_SAFE_STEERING_CANDIDATE"


def test_route_only_supervisor_uses_the_route_corridor_alone():
    route = straight_route(150.0)
    sup = CorridorSupervisor(None, params(), fallback=StaticCorridor(route_boundaries(route.xy, 3.0), 10.0))
    assert sup.available
    d = sup.evaluate(route, 20.0, 0.0, 0.0, 3.0, 20.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.fallback_fraction == pytest.approx(1.0)
    none = CorridorSupervisor(None, params())
    assert not none.available
    assert none.evaluate(route, 20.0, 0.0, 0.0, 3.0, 20.0, 0.0, 0.0, (-0.35, 0.4), 1.5).reason == "CORRIDOR_UNAVAILABLE"


def test_where_the_lane_map_exists_the_route_corridor_is_not_used():
    """Lane walls +/-2 m, route corridor +/-3 m: a corner 2.2 m off is over the lane wall."""

    route = straight_route(150.0)
    fallback = StaticCorridor(route_boundaries(route.xy, 3.0), 10.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params(), fallback=fallback)
    d = sup.evaluate(route, 20.0, 1.6, 0.0, 3.0, 20.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert not d.ok and d.reason == "NO_SAFE_STEERING_CANDIDATE"


def test_route_corridor_is_not_captured_by_the_route_passing_itself():
    """Out along +x at y=0, back along -x at y=5: the return's LEFT wall (y=2)
    lies inside the outbound corridor (y -3..3). A nearest-in-xy wall search
    from (50, 1.5) finds that wall 0.5 m away and calls the outbound "outside"
    (the 20260913_103732 start/end trap); the supervisor pins the fallback's
    window on the route index instead and sees the right wall."""

    out = [(float(i), 0.0) for i in range(0, 101)]
    turn = [(100.0 + 2.5 * math.sin(a), 2.5 - 2.5 * math.cos(a)) for a in np.linspace(0.3, math.pi - 0.3, 6)]
    back = [(float(i), 5.0) for i in range(100, -1, -1)]
    pts = np.array(out + turn + back)
    d = np.diff(pts, axis=0)
    yaw = np.append(np.arctan2(d[:, 1], d[:, 0]), math.pi)
    route = RouteModel.from_arrays(pts, yaw, np.zeros(len(pts)), np.full(len(pts), 3.0))
    fallback = StaticCorridor(route_boundaries(route.xy, 3.0), 10.0)
    fallback.locate(50.0, 1.5)
    left, _, _ = fallback.clearance_points(np.array([[50.0, 1.5]]))
    assert left[0] < 0.0  # the trap: measured against the return's wall
    sup = CorridorSupervisor(None, params(), fallback=fallback)
    d = sup.evaluate(route, 50.0, 1.5, 0.0, 3.0, 50.0, 0.0, 0.0, (-0.35, 0.4), 1.5)
    assert d.ok and d.reason == "OK" and not d.outside
    assert d.current_clearance == pytest.approx(3.0 - 1.5 - 0.628 - 0.4, abs=0.02)
    # The binding walked both walls forward with the route: wall vertex i+1 is
    # route point i (the 10 m extension prepends one vertex), never the return leg.
    assert fallback.bound
    table = fallback._route_left
    assert np.all(np.diff(table) >= 0) and abs(int(table[50]) - 51) <= 1 and abs(int(table[-1]) - len(pts)) <= 1
    # On the return leg (route point 150 is at x~57, y=5) the table points at the return's wall too.
    assert int(table[150]) > 110


def test_bind_route_ties_a_lane_map_to_the_route_point_by_point():
    """A lane map that covers the route with its own point spacing: the walk
    stays monotonic and lands on the wall beside each route point."""

    route = straight_route(150.0)
    xs = np.arange(0.0, 151.0, 0.4)  # 2.5 x denser than the route
    lane = StaticCorridor(BoundarySet(np.column_stack([xs, np.full_like(xs, 2.0)]),
                                      np.column_stack([xs, np.full_like(xs, -2.0)]), [], "x", "leftBoundary", "rightBoundary"))
    stats = lane.bind_route(route.xy)
    assert not stats["warnings"]
    assert np.all(np.diff(lane._route_left) >= 0) and np.all(np.diff(lane._route_right) >= 0)
    assert np.allclose(stats["left_dist"], 2.0) and np.allclose(stats["right_dist"], 2.0)
    lane.locate_route_index(100)
    assert abs(lane._hint_left - 100 / 0.4) <= 1


def test_route_boundaries_follow_the_route_normal():
    route = arc_route(radius=20.0)
    b = route_boundaries(route.xy, 3.0)
    info = validate_boundaries(b)
    assert info["width_min_m"] == pytest.approx(6.0, abs=0.05)
    assert info["right_of_left_fraction"] == pytest.approx(1.0)
    # Left wall of a +x straight is at y = +3.
    assert b.left[5, 1] == pytest.approx(3.0) and b.right[5, 1] == pytest.approx(-3.0)


def test_candidate_is_applied_at_once_then_pure_pursuit_resumes():
    """krowd_final_control semantics: the candidate angle acts from the first
    step and is held for candidate_hold_s; only the Pure Pursuit continuation
    is rate limited."""

    route = straight_route(150.0)
    sup = CorridorSupervisor(StaticCorridor(straight_bounds()), params())
    sup.p.max_steering_rate_rad_s = math.radians(40.0)
    pts, poses = sup._simulate(route, 20.0, 0.0, 0.0, 3.0, 20.0, math.radians(20.0), (-0.4, 0.4), 1.0, 0.1)
    step = 3.0 / 1.34 * math.tan(math.radians(20.0)) * 0.1
    assert poses[0][2] == pytest.approx(step)
    assert poses[4][2] == pytest.approx(5 * step)  # held for 0.5 s = 5 steps
    # Afterwards Pure Pursuit steers back toward the route, at most 4 deg of wheel per step.
    assert poses[5][2] < poses[4][2] + step
    assert poses[5][2] > poses[4][2] + 3.0 / 1.34 * math.tan(math.radians(16.0)) * 0.1 - 1e-9


def test_orientation_check_rejects_swapped_sides():
    b = straight_bounds()
    swapped = BoundarySet(b.right, b.left, [], "x", "leftBoundary", "rightBoundary")
    with pytest.raises(ValueError):
        validate_boundaries(swapped)


HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, "..", "config", "maps", "20260913_103732", "lane_boundaries.csv")


@pytest.mark.skipif(not os.path.isfile(REAL), reason="lane map not present")
def test_real_boundaries_load_and_validate():
    b = load_boundaries(REAL)
    info = validate_boundaries(b)
    assert info["left_points"] == 692 and info["right_points"] == 692
    assert 3.0 < info["width_min_m"] < 3.1
    assert len(b.stoplines) == 13


def test_candidate_offsets_are_capped_by_lateral_acceleration_at_speed():
    route = straight_route(150.0)
    base = params()
    hi = CorridorSupervisor(StaticCorridor(straight_bounds()), replace(base, candidate_max_lat_accel_mps2=2.0))
    lo = CorridorSupervisor(StaticCorridor(straight_bounds()), base)
    args = (route, 20.0, 0.6, 0.0, 11.0, 20.0, math.radians(15.0), math.radians(15.0), (-0.35, 0.4), 3.0)
    d_cap, d_free = hi.evaluate(*args), lo.evaluate(*args)
    # at 11 m/s and 2 m/s^2 the cap is atan(2 * L / 121) ~ 1.3 deg: one 1-deg step each side, then it stops searching
    assert abs(d_cap.candidate_offset_rad) <= math.radians(1.0) + 1e-9
    assert d_cap.candidates_tried <= 3 < d_free.candidates_tried
    # at crawl speed the cap is wide and the search is unchanged
    slow = (route, 20.0, 0.6, 0.0, 1.5, 20.0, math.radians(15.0), math.radians(15.0), (-0.35, 0.4), 2.0)
    assert hi.evaluate(*slow).reason == lo.evaluate(*slow).reason
