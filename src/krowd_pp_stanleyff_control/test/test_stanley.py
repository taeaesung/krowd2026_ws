import math

import numpy as np
import pytest

from krowd_pp_stanleyff_control.route import RouteModel
from krowd_pp_stanleyff_control.stanley import (
    cross_track_steering,
    front_axle_position,
    predict_front_axle_pose,
    steering_angle,
    yaw_rate_damping,
)
from krowd_pp_stanleyff_control.geometry import wrap_angle

WHEELBASE = 1.340
GAIN = 1.0
SOFTENING = 1.0


def test_on_the_route_and_aligned_commands_no_steering():
    assert steering_angle(0.0, 0.0, 3.0, GAIN, SOFTENING) == pytest.approx(0.0)


def test_left_of_the_route_steers_right_by_the_stanley_law():
    speed = 2.0
    left = steering_angle(0.0, 0.5, speed, GAIN, SOFTENING)
    right = steering_angle(0.0, -0.5, speed, GAIN, SOFTENING)
    assert left == pytest.approx(-math.atan(GAIN * 0.5 / (SOFTENING + speed)))
    assert left < 0.0
    assert right == pytest.approx(-left)


def test_heading_error_passes_through_at_the_heading_gain():
    error = math.radians(10.0)
    assert steering_angle(error, 0.0, 3.0, GAIN, SOFTENING) == pytest.approx(error)
    assert steering_angle(error, 0.0, 3.0, GAIN, SOFTENING, heading_gain=0.5) == pytest.approx(0.5 * error)


def test_cross_track_term_is_finite_at_a_standstill_and_weakens_with_speed():
    assert cross_track_steering(1.0, 0.0, GAIN, SOFTENING) == pytest.approx(-math.atan(GAIN / SOFTENING))
    assert abs(cross_track_steering(1.0, 5.0, GAIN, SOFTENING)) < abs(cross_track_steering(1.0, 1.0, GAIN, SOFTENING))
    assert cross_track_steering(1.0, -3.0, GAIN, SOFTENING) == pytest.approx(cross_track_steering(1.0, 0.0, GAIN, SOFTENING))
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="softening"):
            cross_track_steering(1.0, 1.0, GAIN, bad)


def test_heading_error_wraps_across_pi():
    assert wrap_angle(math.radians(179.0) - math.radians(-179.0)) == pytest.approx(math.radians(-2.0))
    assert steering_angle(math.radians(358.0), 0.0, 3.0, GAIN, SOFTENING) == pytest.approx(math.radians(-2.0))


def test_front_axle_sits_a_wheelbase_ahead_of_the_rear_axle():
    x, y = front_axle_position(10.0, 5.0, math.pi / 2.0, 0.0, WHEELBASE)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(5.0 + WHEELBASE)
    x, _ = front_axle_position(0.0, 0.0, 0.0, -0.3, WHEELBASE)
    assert x == pytest.approx(WHEELBASE - 0.3)


def test_0924_latency_prediction_advances_a_straight_front_axle():
    predicted = predict_front_axle_pose(
        WHEELBASE, 0.0, 0.0, 2.0, 0.0, WHEELBASE, 0.60
    )
    assert predicted.x == pytest.approx(WHEELBASE + 1.2)
    assert predicted.y == pytest.approx(0.0)
    assert predicted.yaw == pytest.approx(0.0)


def test_0924_latency_prediction_matches_the_original_bicycle_equation():
    speed = 5.0 / 3.6
    previous = math.radians(8.0)
    latency = 0.60
    predicted = predict_front_axle_pose(
        WHEELBASE, 0.0, 0.0, speed, previous, WHEELBASE, latency
    )

    yaw_rate = speed / WHEELBASE * math.tan(previous)
    yaw_mid = 0.5 * yaw_rate * latency
    rear_x = speed * math.cos(yaw_mid) * latency
    rear_y = speed * math.sin(yaw_mid) * latency
    yaw = yaw_rate * latency
    assert predicted.x == pytest.approx(rear_x + WHEELBASE * math.cos(yaw))
    assert predicted.y == pytest.approx(rear_y + WHEELBASE * math.sin(yaw))
    assert predicted.yaw == pytest.approx(yaw)
    assert predicted.y > 0.0


def test_0924_proven_gains_reproduce_the_original_raw_command():
    heading_error = math.radians(4.0)
    lateral_error = 0.35
    speed = 5.0 / 3.6
    actual = steering_angle(
        heading_error,
        lateral_error,
        speed,
        cross_track_gain=1.5,
        softening_mps=0.5,
        heading_gain=1.0,
    )
    expected = heading_error + math.atan2(-1.5 * lateral_error, 0.5 + abs(speed))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("wheelbase,latency", [(0.0, 0.6), (-1.0, 0.6), (1.34, -0.1)])
def test_latency_prediction_rejects_invalid_geometry_or_time(wheelbase, latency):
    with pytest.raises(ValueError):
        predict_front_axle_pose(1.34, 0.0, 0.0, 1.0, 0.0, wheelbase, latency)


def _circle(radius, count, spacing=1.0):
    step = spacing / radius
    return np.array([(radius * math.sin(i * step), radius * (1.0 - math.cos(i * step))) for i in range(count)])


def test_route_heading_window_follows_the_geometry_through_an_arc():
    """The chord over a window centred on a vertex is the tangent there."""

    radius = 10.0
    xy = _circle(radius, 20)
    n = len(xy)
    route = RouteModel.from_arrays(xy, np.zeros(n), np.full(n, 1.0 / radius), np.full(n, 3.0))
    s7 = route.s_at_index(7)
    assert route.heading_over_window(s7, 2.0) == pytest.approx(7.0 / radius, abs=2e-3)
    # Clamped at the route start: the chord runs from waypoint 0 forwards.
    assert route.heading_over_window(0.0, 2.0) > 0.0
    # No window: the projected segment's own direction.
    assert route.heading_over_window(s7 + 0.3, 0.0) == pytest.approx(route.yaw_at_s(s7 + 0.3))


def test_yaw_damping_opposes_a_rotation_the_straight_does_not_ask_for():
    correction = yaw_rate_damping(5.0 / 3.6, 0.0, math.radians(10.0), 0.15)
    assert math.degrees(correction) == pytest.approx(-1.5)
    correction = yaw_rate_damping(5.0 / 3.6, 0.0, math.radians(-10.0), 0.15)
    assert math.degrees(correction) == pytest.approx(+1.5)


def test_yaw_damping_is_zero_on_a_curve_driven_correctly():
    speed, curvature = 3.0, 1.0 / 20.0
    assert yaw_rate_damping(speed, curvature, speed * curvature, 0.15) == pytest.approx(0.0)
    assert yaw_rate_damping(speed, curvature, 0.0, 0.15) > 0.0
    assert yaw_rate_damping(speed, -curvature, 0.0, 0.15) < 0.0
    assert yaw_rate_damping(3.0, 0.05, 0.4, 0.0) == 0.0
    assert yaw_rate_damping(-2.0, 0.05, 0.1, 0.15) == pytest.approx(-0.015)
