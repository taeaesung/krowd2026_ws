import pytest

from krowd_pp_stanleyff_control.route_geometry import (
    forward_progress_indices,
    path_curvatures,
    remaining_arc_length,
    segment_lengths,
    value_at_route_position,
)


def test_segment_lengths_are_one_shorter_than_the_route(straight_route):
    lengths = segment_lengths(straight_route(5, 2.0))
    assert lengths == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_segment_lengths_ignore_a_single_point():
    assert segment_lengths([(0.0, 0.0)]) == []


def test_a_straight_route_has_no_curvature(straight_route):
    curvatures = path_curvatures(straight_route(10), smoothing_window=1)
    assert curvatures == pytest.approx([0.0] * 10, abs=1e-12)


def test_curvature_of_a_circle_is_one_over_its_radius(circular_route):
    radius = 20.0
    curvatures = path_curvatures(circular_route(radius, 30), smoothing_window=1)
    # Skip the copied endpoints and check the interior.
    for value in curvatures[2:-2]:
        assert value == pytest.approx(1.0 / radius, rel=1e-3)


def test_curvature_is_positive_turning_left(circular_route):
    left = path_curvatures(circular_route(20.0, 20), smoothing_window=1)
    right = path_curvatures(
        [(x, -y) for x, y in circular_route(20.0, 20)], smoothing_window=1
    )
    assert left[5] > 0.0
    assert right[5] == pytest.approx(-left[5])


def test_endpoints_copy_their_neighbours(circular_route):
    curvatures = path_curvatures(circular_route(15.0, 12), smoothing_window=1)
    assert curvatures[0] == pytest.approx(curvatures[1])
    assert curvatures[-1] == pytest.approx(curvatures[-2])


def test_curvature_needs_three_points():
    assert path_curvatures([(0.0, 0.0), (1.0, 0.0)]) == [0.0, 0.0]
    assert path_curvatures([]) == []


def test_smoothing_suppresses_an_isolated_kink(straight_route):
    route = straight_route(21)
    route[10] = (10.0, 0.25)

    raw = path_curvatures(route, smoothing_window=1)
    smoothed = path_curvatures(route, smoothing_window=5)

    assert abs(smoothed[10]) < abs(raw[10])


def test_smoothing_preserves_a_real_constant_curvature(circular_route):
    smoothed = path_curvatures(circular_route(25.0, 40), smoothing_window=5)
    for value in smoothed[4:-4]:
        assert value == pytest.approx(1.0 / 25.0, rel=5e-3)


def test_remaining_arc_length_from_the_start_is_the_whole_route(straight_route):
    assert remaining_arc_length(straight_route(11), 0, 0.0) == pytest.approx(10.0)


def test_remaining_arc_length_accounts_for_the_partial_segment(straight_route):
    assert remaining_arc_length(straight_route(11), 3, 0.25) == pytest.approx(6.75)


def test_remaining_arc_length_at_the_end_is_zero(straight_route):
    assert remaining_arc_length(straight_route(11), 9, 1.0) == pytest.approx(0.0)


def test_forward_progress_keeps_a_clean_route(straight_route):
    route = straight_route(20)
    assert forward_progress_indices(route) == list(range(20))


def test_forward_progress_keeps_a_real_corner(circular_route):
    # A 5 m radius arc at 1 m spacing turns ~11.5 deg per segment: well under
    # the reversal threshold, so no part of a genuine corner may be dropped.
    route = circular_route(5.0, 30)
    assert forward_progress_indices(route) == list(range(30))


def test_forward_progress_drops_a_recorded_localization_jump():
    """The 20260907_114123 excursion, in its own coordinates.

    Waypoints 2 and 3 here re-tread 0 and 1: the recorded pose jumped back a
    metre and drove it again. Both halves of the excursion have to go - dropping
    only the backward step leaves a near-duplicate of the point before it.
    """

    route = [
        (302581.54, 4123860.26),
        (302582.22, 4123859.37),
        (302581.76, 4123860.34),
        (302582.25, 4123859.44),
        (302582.94, 4123858.59),
        (302583.55, 4123857.46),
    ]

    assert forward_progress_indices(route) == [0, 1, 4, 5]


def test_forward_progress_drops_a_standstill_duplicate():
    route = [(0.0, 0.0), (1.0, 0.0), (1.001, 0.0), (2.0, 0.0)]
    assert forward_progress_indices(route) == [0, 1, 3]


def test_forward_progress_can_be_disabled():
    route = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    kept = forward_progress_indices(route, max_reversal_deg=0.0, min_step_m=0.0)
    assert kept == [0, 1, 2, 3]


def test_forward_progress_handles_short_routes():
    assert forward_progress_indices([]) == []
    assert forward_progress_indices([(0.0, 0.0)]) == [0]


def test_value_interpolates_between_waypoints():
    assert value_at_route_position([2.0, 4.0, 6.0], 1, 0.5) == pytest.approx(5.0)


def test_value_clamps_a_position_past_the_end():
    assert value_at_route_position([2.0, 4.0, 6.0], 9, 0.5) == pytest.approx(5.0)


def test_value_of_a_single_waypoint():
    assert value_at_route_position([3.0], 0, 0.5) == pytest.approx(3.0)


def test_value_rejects_an_empty_profile():
    with pytest.raises(ValueError):
        value_at_route_position([], 0, 0.0)
