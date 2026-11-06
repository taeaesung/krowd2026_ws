import math

import pytest

from krowd_path_tracking.stanley import (
    cross_track_steering,
    front_axle_position,
    heading_not_aligned,
    path_heading,
    steering_angle,
    wrap_angle,
)


WHEELBASE = 1.340
GAIN = 1.0
SOFTENING = 1.0


def test_on_the_route_and_aligned_commands_no_steering():
    assert steering_angle(0.0, 0.0, 3.0, GAIN, SOFTENING) == pytest.approx(0.0)


def test_left_of_the_route_steers_right_by_the_stanley_law():
    """signed_offset is positive LEFT of the route; REP-103 steering is positive
    LEFT. A vehicle left of the route must therefore steer right."""

    speed = 2.0
    left = steering_angle(0.0, 0.5, speed, GAIN, SOFTENING)
    right = steering_angle(0.0, -0.5, speed, GAIN, SOFTENING)

    assert left == pytest.approx(-math.atan(GAIN * 0.5 / (SOFTENING + speed)))
    assert left < 0.0
    assert right == pytest.approx(-left)


def test_heading_error_passes_through_at_the_heading_gain():
    error = math.radians(10.0)
    assert steering_angle(error, 0.0, 3.0, GAIN, SOFTENING) == pytest.approx(error)
    assert steering_angle(
        error, 0.0, 3.0, GAIN, SOFTENING, heading_gain=0.5
    ) == pytest.approx(0.5 * error)


def test_cross_track_term_is_finite_at_a_standstill():
    """Plain atan(k e / v) is undefined at v = 0; the softening speed is why this is not."""

    commanded = cross_track_steering(1.0, 0.0, GAIN, SOFTENING)
    assert commanded == pytest.approx(-math.atan(GAIN / SOFTENING))


def test_cross_track_term_weakens_as_speed_rises():
    slow = abs(cross_track_steering(1.0, 1.0, GAIN, SOFTENING))
    fast = abs(cross_track_steering(1.0, 5.0, GAIN, SOFTENING))
    assert fast < slow


def test_reverse_speed_is_treated_as_a_standstill():
    assert cross_track_steering(1.0, -3.0, GAIN, SOFTENING) == pytest.approx(
        cross_track_steering(1.0, 0.0, GAIN, SOFTENING)
    )


def test_softening_speed_must_be_positive():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="softening"):
            cross_track_steering(1.0, 1.0, GAIN, bad)


def test_heading_error_wraps_across_pi():
    # Route at +179 deg, vehicle at -179 deg: 2 deg apart, not 358.
    assert wrap_angle(math.radians(179.0) - math.radians(-179.0)) == pytest.approx(
        math.radians(-2.0)
    )
    assert steering_angle(
        math.radians(358.0), 0.0, 3.0, GAIN, SOFTENING
    ) == pytest.approx(math.radians(-2.0))


def test_front_axle_sits_a_wheelbase_ahead_of_the_rear_axle():
    x, y = front_axle_position(10.0, 5.0, math.pi / 2.0, 0.0, WHEELBASE)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(5.0 + WHEELBASE)
    # Rear axle 0.3 m behind base_link: the front axle is WHEELBASE - 0.3 ahead of it.
    x, _ = front_axle_position(0.0, 0.0, 0.0, -0.3, WHEELBASE)
    assert x == pytest.approx(WHEELBASE - 0.3)


def test_path_heading_on_a_straight_route(straight_route):
    assert path_heading(straight_route(10), 4, 0.5, 2.0) == pytest.approx(0.0)


def test_path_heading_follows_the_route_geometry_through_an_arc(circular_route):
    """The chord over a window centred on a vertex is the tangent there."""

    radius = 10.0
    route = circular_route(radius, 20)
    step = 1.0 / radius
    chord = 2.0 * radius * math.sin(0.5 * step)
    assert path_heading(route, 7, 0.0, 2.0 * chord) == pytest.approx(7 * step, abs=1e-9)


def test_path_heading_window_is_clamped_at_the_route_start(circular_route):
    radius = 10.0
    route = circular_route(radius, 20)
    step = 1.0 / radius
    chord = 2.0 * radius * math.sin(0.5 * step)
    # Nothing lies behind waypoint 0, so the chord runs from it to waypoint 1.
    assert path_heading(route, 0, 0.0, 2.0 * chord) == pytest.approx(
        0.5 * step, abs=1e-9
    )


def test_path_heading_without_a_window_uses_the_projected_segment(circular_route):
    radius = 10.0
    route = circular_route(radius, 20)
    step = 1.0 / radius
    # Segment 7 joins vertices 7 and 8; its direction is midway between their tangents.
    assert path_heading(route, 7, 0.3, 0.0) == pytest.approx(7.5 * step, abs=1e-9)


def test_heading_alignment_gate():
    limit = math.radians(60.0)
    assert not heading_not_aligned(math.radians(59.0), limit)
    assert heading_not_aligned(math.radians(-61.0), limit)
    assert not heading_not_aligned(math.radians(170.0), 0.0)  # 0 disables


def test_heading_alignment_gate_uses_the_wrapped_error():
    # A 359 deg raw difference is a 1 deg misalignment, not a reversed vehicle.
    assert not heading_not_aligned(math.radians(359.0), math.radians(60.0))
    # 180 deg is a vehicle facing the route backwards.
    assert heading_not_aligned(math.pi, math.radians(60.0))


def test_the_offset_sign_matches_route_io_projection():
    """Guard the one convention the law cannot check by itself."""

    route_io = pytest.importorskip("mola_slam_mapping.route_io")
    visualization = pytest.importorskip("mola_slam_mapping.route_visualization")
    route = [
        visualization.make_route_point(float(index), 0.0, 0.0, 0.0, 3.0)
        for index in range(5)
    ]

    projection = route_io.project_onto_route(route, 2.0, 0.5)  # left of a +x route

    assert projection.signed_offset == pytest.approx(0.5)
    assert steering_angle(0.0, projection.signed_offset, 2.0, GAIN, SOFTENING) < 0.0
