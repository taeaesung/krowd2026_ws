"""merge_close_corners (2026-10-03): corners a few metres apart become one corner."""

from krowd_pp_stanleyff_control.speed_profile import merge_close_corners

V, C = 25.0, 7.0  # straight / corner speed (units do not matter here)


def test_short_gap_between_two_corners_is_held_at_corner_speed():
    ceilings = [V, C, C, V, V, C, C, V]
    lengths = [1.0] * 7
    # gap: waypoints 3-4, 3 m from the last corner point (2) to the next (5) < 15 m
    assert merge_close_corners(ceilings, lengths, V, 15.0) == [V, C, C, C, C, C, C, V]


def test_long_gap_stays_a_straight():
    ceilings = [V, C, C] + [V] * 20 + [C, C, V]
    out = merge_close_corners(ceilings, [1.0] * (len(ceilings) - 1), V, 15.0)
    assert out == ceilings


def test_route_start_and_end_are_not_gaps():
    ceilings = [V, V, C, C, V, V]
    assert merge_close_corners(ceilings, [1.0] * 5, V, 15.0) == ceilings


def test_fill_takes_the_lower_side_and_zero_disables():
    ceilings = [V, 9.0, V, 7.0, V]
    assert merge_close_corners(ceilings, [1.0] * 4, V, 15.0) == [V, 9.0, 7.0, 7.0, V]
    assert merge_close_corners(ceilings, [1.0] * 4, V, 0.0) == ceilings
