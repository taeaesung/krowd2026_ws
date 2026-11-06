import math

import pytest

from krowd_path_tracking.pure_pursuit import (
    clamp_steering,
    lookahead_distance,
    rate_limited,
    rear_axle_position,
    steering_angle,
    to_body_frame,
)
from krowd_path_tracking.vehicle import (
    SteeringActuator,
    Vehicle,
    implausible_steering_limit_reason,
    speed_command_kmh,
    steering_limit_from_turning_radius,
)


WHEELBASE = 1.340


def test_speed_command_is_km_h_for_the_drive_mcu():
    # ros_can_bridge sends /target_speed x 100 on CAN 0x110 and the drive MCU
    # reads it as km/h x 100: a 15 km/h route speed (4.1667 m/s) must go out
    # as 15.0, not 4.17 (which the MCU would drive as 4.17 km/h).
    assert speed_command_kmh(15.0 / 3.6) == pytest.approx(15.0)
    assert speed_command_kmh(0.0) == 0.0
    assert speed_command_kmh(-1.0) == 0.0


def test_target_straight_ahead_commands_no_steering():
    assert steering_angle(5.0, 0.0, WHEELBASE) == pytest.approx(0.0)


def test_steering_follows_the_arc_through_the_target():
    # A target at lateral offset y and lookahead ld lies on an arc of radius
    # ld^2 / (2y); the bicycle model steers atan(L / R) to follow it.
    lookahead = 5.0
    lateral = 1.0
    forward = math.sqrt(lookahead**2 - lateral**2)
    radius = lookahead**2 / (2.0 * lateral)

    commanded = steering_angle(forward, lateral, WHEELBASE)

    assert commanded == pytest.approx(math.atan(WHEELBASE / radius), rel=1e-9)


def test_steering_sign_follows_the_target_side():
    left = steering_angle(4.0, 1.0, WHEELBASE)
    right = steering_angle(4.0, -1.0, WHEELBASE)
    assert left > 0.0
    assert right == pytest.approx(-left)


def test_target_on_the_rear_axle_commands_no_steering():
    # Degenerate rather than infinite: the squared-distance form has to stay
    # finite when the lookahead collapses onto the vehicle.
    assert steering_angle(0.0, 0.0, WHEELBASE) == 0.0


def test_heading_error_is_attenuated_by_the_lookahead():
    """The documented reason this controller was chosen over Stanley.

    A yaw error moves the commanded angle by about (2 L / ld) * error, so a
    longer lookahead attenuates it. Stanley would pass the same error through at
    unity gain.
    """

    lookahead = 6.0
    heading_error = math.radians(9.0)

    # The target sits straight ahead; only the vehicle's believed yaw is wrong.
    truth = to_body_frame(0.0, 0.0, 0.0, lookahead, 0.0)
    perturbed = to_body_frame(0.0, 0.0, heading_error, lookahead, 0.0)

    error_in_command = abs(
        steering_angle(*perturbed, WHEELBASE) - steering_angle(*truth, WHEELBASE)
    )
    predicted = (2.0 * WHEELBASE / lookahead) * heading_error

    assert error_in_command == pytest.approx(predicted, rel=0.05)
    assert error_in_command < heading_error


def test_rear_axle_offset_slides_along_the_body_axis():
    x, y = rear_axle_position(10.0, 5.0, math.pi / 2.0, -1.5)
    assert x == pytest.approx(10.0)
    assert y == pytest.approx(3.5)


def test_body_frame_puts_a_point_ahead_on_positive_x():
    # Vehicle at (4, 4) facing north; a point directly north of it is ahead.
    forward, lateral = to_body_frame(4.0, 4.0, math.pi / 2.0, 4.0, 9.0)
    assert forward == pytest.approx(5.0)
    assert lateral == pytest.approx(0.0)


def test_lookahead_scales_with_speed_between_its_bounds():
    assert lookahead_distance(9.0, 0.6, 3.0, 8.0) == pytest.approx(5.4)


def test_lookahead_clamps_at_both_ends():
    assert lookahead_distance(0.0, 0.6, 3.0, 8.0) == pytest.approx(3.0)
    assert lookahead_distance(50.0, 0.6, 3.0, 8.0) == pytest.approx(8.0)


def test_lookahead_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        lookahead_distance(5.0, 0.6, 9.0, 3.0)


def test_rate_limit_caps_a_step_command():
    limited = rate_limited(0.0, 1.0, math.radians(60.0), 0.05)
    assert limited == pytest.approx(math.radians(3.0))


def test_rate_limit_passes_a_small_change_through():
    assert rate_limited(0.1, 0.11, math.radians(60.0), 0.05) == pytest.approx(0.11)


def test_rate_limit_is_symmetric():
    assert rate_limited(0.0, -1.0, math.radians(60.0), 0.05) == pytest.approx(
        -math.radians(3.0)
    )


def test_clamp_steering_bounds_both_directions():
    assert clamp_steering(1.0, 0.4) == pytest.approx(0.4)
    assert clamp_steering(-1.0, 0.4) == pytest.approx(-0.4)
    assert clamp_steering(0.2, 0.4) == pytest.approx(0.2)


def test_steering_limit_matches_the_measured_turning_circle():
    # wheelbase 1.340 m, 5.15 m turning-circle diameter by hand (2026-09-19).
    limit = steering_limit_from_turning_radius(1.340, 2.575)
    assert math.degrees(limit) == pytest.approx(27.5, abs=0.1)


def test_steering_command_is_the_road_wheel_angle_times_the_measured_gain():
    actuator = SteeringActuator(gain=8.0, offset_rad=0.0, max_command_rad=math.radians(99.0))
    actuator.validate()
    assert actuator.command(math.radians(5.0)) == pytest.approx(math.radians(40.0))
    assert actuator.command(math.radians(5.0), sign=-1.0) == pytest.approx(-math.radians(40.0))
    # Beyond the MCU's range the command is clamped, never sent as-is.
    assert actuator.command(math.radians(20.0)) == pytest.approx(math.radians(99.0))
    assert math.degrees(actuator.road_wheel_limit_rad) == pytest.approx(12.375)
    offset = SteeringActuator(gain=8.0, offset_rad=0.08, max_command_rad=math.radians(99.0))
    assert offset.command(0.0) == pytest.approx(0.08)
    with pytest.raises(ValueError):
        SteeringActuator(gain=0.0, offset_rad=0.0, max_command_rad=1.7).validate()


def test_steering_limit_rejects_nonsense_geometry():
    with pytest.raises(ValueError):
        steering_limit_from_turning_radius(0.0, 3.4)
    with pytest.raises(ValueError):
        steering_limit_from_turning_radius(1.34, 0.0)


def _vehicle(max_steering_deg: float) -> Vehicle:
    return Vehicle(
        wheelbase_m=1.340,
        track_width_m=1.080,
        min_turning_radius_m=3.4,
        max_steering_angle_rad=math.radians(max_steering_deg),
        max_steering_rate_rad_s=math.radians(60.0),
        rear_axle_offset_m=0.0,
    )


def test_spec_sheet_steering_limit_is_not_flagged():
    assert implausible_steering_limit_reason(_vehicle(21.5)) is None


def test_steering_wheel_angle_mistaken_for_road_wheel_is_flagged():
    # 75 deg of road-wheel angle implies a 0.36 m radius, inside the vehicle's
    # own half track width - the signature of a steering-wheel figure.
    reason = implausible_steering_limit_reason(_vehicle(75.0))
    assert reason is not None
    assert "road-WHEEL" in reason


def test_max_curvature_is_the_inverse_of_the_implied_radius():
    vehicle = _vehicle(21.5)
    assert vehicle.max_curvature == pytest.approx(
        1.0 / vehicle.implied_turning_radius_m
    )


def test_curvature_lookahead_is_4_m_on_straights_and_1_m_in_curves():
    from krowd_path_tracking.pure_pursuit import curvature_lookahead

    def ld(radius):
        return curvature_lookahead(1.0 / radius, 4.0, 1.0, 50.0, 15.0)

    assert curvature_lookahead(0.0, 4.0, 1.0, 50.0, 15.0) == pytest.approx(4.0)
    assert ld(200.0) == pytest.approx(4.0)
    assert ld(50.0) == pytest.approx(4.0)
    assert ld(15.0) == pytest.approx(1.0)
    assert ld(5.0) == pytest.approx(1.0)
    # Halfway in curvature between 1/50 and 1/15.
    halfway = 2.0 / (1.0 / 50.0 + 1.0 / 15.0)
    assert ld(halfway) == pytest.approx(2.5)
    # A right-hand bend shortens it the same.
    assert curvature_lookahead(-1.0 / 15.0, 4.0, 1.0, 50.0, 15.0) == pytest.approx(1.0)


def test_curvature_ahead_sees_a_corner_within_the_window_only():
    from krowd_path_tracking.route_geometry import max_abs_value_ahead

    points = [(float(x), 0.0) for x in range(20)]
    curvatures = [0.0] * 20
    curvatures[10] = -0.2  # a right-hand corner at x = 10

    # From x = 4.5: 6 m ahead reaches x = 10.5, 4 m only x = 8.5.
    assert max_abs_value_ahead(points, curvatures, 4, 0.5, 6.0) == pytest.approx(0.2)
    assert max_abs_value_ahead(points, curvatures, 4, 0.5, 4.0) == pytest.approx(0.0)
    # Already in the corner.
    assert max_abs_value_ahead(points, curvatures, 10, 0.0, 1.0) == pytest.approx(0.2)
