"""align_hold (2026-10-03): no acceleration until the heading is back on the route."""

import math

import pytest

from krowd_pp_stanleyff_control.speed_planner import align_hold_cap

LIM = math.radians(10.0)
V5, V7, V15 = 5 / 3.6, 7 / 3.6, 15 / 3.6


def test_aligned_vehicle_is_not_held():
    assert align_hold_cap(math.radians(4.0), LIM, V7, V15, V5, None) == (math.inf, None)


def test_misaligned_out_of_a_corner_holds_the_corner_speed():
    # Coming out of a 7 km/h corner 35 deg off the route, the profile already asks for 15.
    cap, hold = align_hold_cap(math.radians(35.0), LIM, V7, V15, V5, None)
    assert cap == V7 and hold == V7


def test_hold_never_raised_but_follows_the_target_down():
    cap, hold = align_hold_cap(math.radians(-30.0), LIM, 9 / 3.6, V15, V5, V7)
    assert cap == V7  # speed rose to 9 (coasting in): still held at 7
    cap, hold = align_hold_cap(math.radians(-30.0), LIM, V7, 6 / 3.6, V5, hold)
    assert cap == 6 / 3.6  # the target itself dropped


def test_standing_still_may_crawl():
    cap, _ = align_hold_cap(math.radians(20.0), LIM, 0.0, V15, V5, None)
    assert cap == V5


def test_released_when_aligned_again():
    assert align_hold_cap(math.radians(9.5), LIM, V7, V15, V5, V7) == (math.inf, None)


def test_zero_limit_disables():
    assert align_hold_cap(math.radians(80.0), 0.0, V7, V15, V5, None) == (math.inf, None)


def test_steering_hold_is_anded_with_the_heading_condition():
    # Heading aligned, wheels still turned (or no encoder): no acceleration.
    cap, hold = align_hold_cap(math.radians(2.0), LIM, V7, V15, V5, None, steering_hold=True)
    assert cap == pytest.approx(V7) and hold == pytest.approx(V7)
    # Both clear: released.
    assert align_hold_cap(math.radians(2.0), LIM, V7, V15, V5, hold, steering_hold=False) == (math.inf, None)
    # Heading still off with the wheels straight: still held.
    cap, _ = align_hold_cap(math.radians(25.0), LIM, V7, V15, V5, hold, steering_hold=False)
    assert cap == pytest.approx(V7)


def test_steering_hold_works_with_the_heading_condition_disabled():
    cap, _ = align_hold_cap(math.radians(80.0), 0.0, 0.0, V15, V5, None, steering_hold=True)
    assert cap == pytest.approx(V5)
