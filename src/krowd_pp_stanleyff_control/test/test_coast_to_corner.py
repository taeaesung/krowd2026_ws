"""coast_to_corner (2026-10-03): corner speed as one step, early enough to coast down to it."""

import math

from krowd_pp_stanleyff_control.speed_planner import coast_aim_points, coast_step_cap, profile_drops

V15 = 15 / 3.6
V7 = 7 / 3.6
V17 = 17.4 / 3.6


def test_profile_drops_finds_steps_down_but_not_the_route_end():
    s = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    speed = [V15, V15, V7, V7, V15, V15, 0.0]
    # The drop to 7 km/h is taken whole at the first waypoint of its segment (s=1);
    # the final 0 (route-end stop) belongs to the braking model.
    assert profile_drops(s, speed) == [(1.0, V7)]


def test_no_cap_while_the_vehicle_can_still_coast_down_later():
    # 15 -> 7 km/h at 0.5 m/s^2 needs (4.17^2 - 1.94^2) / 1.0 = 13.6 m.
    cap, latched = coast_step_cap(0.0, V15, [(20.0, V7)], 0.5, set())
    assert cap == math.inf and latched == set()


def test_corner_speed_commanded_whole_once_coasting_just_reaches_it():
    cap, latched = coast_step_cap(7.0, V15, [(20.0, V7)], 0.5, set())  # 13 m before the drop
    assert cap == V7 and latched == {20.0}


def test_actual_speed_decides_not_the_profile():
    # 19 m before the drop: 15 km/h still has room, 17.4 km/h (trim overshoot) does not.
    assert coast_step_cap(1.0, V15, [(20.0, V7)], 0.5, set())[0] == math.inf
    assert coast_step_cap(1.0, V17, [(20.0, V7)], 0.5, set())[0] == V7


def test_latched_until_the_drop_is_passed():
    # Slower than the coast curve now (it decelerated harder): stays at corner speed, no chatter.
    cap, latched = coast_step_cap(15.0, 2.5, [(20.0, V7)], 0.5, {20.0})
    assert cap == V7 and latched == {20.0}
    # Past the drop the route profile itself carries the corner speed; the latch goes.
    cap, latched = coast_step_cap(20.5, 2.0, [(20.0, V7)], 0.5, {20.0})
    assert cap == math.inf and latched == set()


def test_lowest_of_several_drops_and_horizon():
    drops = [(30.0, 12 / 3.6), (40.0, V7), (400.0, 1.0)]
    # s=25: the 12 km/h step binds (5 m), the 7 km/h one 15 m ahead still has room.
    cap, latched = coast_step_cap(25.0, V15, drops, 0.5, set())
    assert cap == 12 / 3.6 and latched == {30.0}
    # s=28: both bind; the lower wins. The 400 m one is beyond the 150 m horizon.
    cap, latched = coast_step_cap(28.0, V15, drops, 0.5, latched)
    assert cap == V7 and latched == {30.0, 40.0}


def test_zero_decel_disables():
    assert coast_step_cap(19.0, V15, [(20.0, V7)], 0.0, set()) == (math.inf, set())


BELOW = 2 / 3.6


def test_ramp_follows_the_coast_curve_below_the_vehicle():
    # 13 m before the drop at 15 km/h it triggers; the command is the coast curve
    # (sqrt(1.94^2 + 2*0.5*13) = 4.10 m/s) 2 km/h under it - not 7 km/h.
    cap, latched = coast_step_cap(7.0, V15, [(20.0, V7)], 0.5, set(), ramp_below_mps=BELOW)
    assert latched == {20.0}
    assert math.isclose(cap, math.sqrt(V7**2 + 2 * 0.5 * 13.0) - BELOW)
    assert V7 < cap < V15


def test_ramp_only_falls_and_ends_at_the_corner_speed():
    caps, latched = [], set()
    for s in (7.0, 10.0, 14.0, 18.0, 19.9):
        cap, latched = coast_step_cap(s, 2.0, [(20.0, V7)], 0.5, latched or {20.0}, ramp_below_mps=BELOW)
        caps.append(cap)
    assert all(b <= a for a, b in zip(caps, caps[1:]))
    assert caps[-1] == V7  # never under the corner speed


def test_ramp_trigger_is_the_same_as_the_step():
    assert coast_step_cap(1.0, V15, [(20.0, V7)], 0.5, set(), ramp_below_mps=BELOW)[0] == math.inf


def test_aim_is_just_before_the_corner_proper():
    s = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    k = [0.0, 0.0, 0.05, 0.08, 0.2, 0.2]  # R 20 m, 12.5 m, then 5 m from s=20
    assert coast_aim_points([(5.0, V7)], s, k, 10.0, 2.0) == [(5.0, V7, 18.0)]


def test_aim_never_before_the_drop_and_falls_back_to_it():
    s = [0.0, 5.0, 10.0, 15.0]
    assert coast_aim_points([(5.0, V7)], s, [0.0, 0.3, 0.3, 0.3], 10.0, 2.0) == [(5.0, V7, 5.0)]  # tight at the drop
    assert coast_aim_points([(5.0, V7)], s, [0.0, 0.05, 0.05, 0.05], 10.0, 2.0) == [(5.0, V7, 5.0)]  # gentle bend only
    assert coast_aim_points([(5.0, V7)], s, [0.0, 0.0, 0.3, 0.3], 0.0, 2.0) == [(5.0, V7, 5.0)]  # disabled


def test_later_aim_starts_the_coast_later():
    # 15 -> 7 km/h needs 13.6 m. Aimed at the drop (20 m) it binds 13 m before; aimed 6 m past it, not yet.
    assert coast_step_cap(7.0, V15, [(20.0, V7)], 0.5, set())[0] == V7
    assert coast_step_cap(7.0, V15, [(20.0, V7, 26.0)], 0.5, set())[0] == math.inf
    cap, latched = coast_step_cap(13.0, V15, [(20.0, V7, 26.0)], 0.5, set())
    assert cap == V7 and latched == {20.0}  # 13 m before the aim; identity stays the drop
