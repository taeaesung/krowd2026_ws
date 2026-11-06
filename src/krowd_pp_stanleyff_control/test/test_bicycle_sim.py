import math

import pytest

from krowd_pp_stanleyff_control.bicycle_sim import BicycleState, step_bicycle
from krowd_pp_stanleyff_control.stanley import steering_angle


def test_straight_motion_accelerates_and_advances():
    state, yaw_rate = step_bicycle(
        BicycleState(0.0, 0.0, 0.0, 0.0, 0.0),
        target_speed_mps=2.0,
        target_steering_rad=0.0,
        dt=1.0,
        wheelbase_m=1.34,
        max_accel_mps2=1.0,
        max_decel_mps2=2.0,
        max_steering_rate_rad_s=math.radians(60.0),
    )
    assert state.speed_mps == pytest.approx(1.0)
    assert state.x == pytest.approx(1.0)
    assert state.y == pytest.approx(0.0)
    assert yaw_rate == pytest.approx(0.0)


def test_left_steering_produces_left_turn():
    state, yaw_rate = step_bicycle(
        BicycleState(0.0, 0.0, 0.0, 2.0, 0.0),
        target_speed_mps=2.0,
        target_steering_rad=math.radians(10.0),
        dt=0.1,
        wheelbase_m=1.34,
        max_accel_mps2=1.0,
        max_decel_mps2=2.0,
        max_steering_rate_rad_s=math.radians(100.0),
    )
    assert state.yaw > 0.0
    assert state.y > 0.0
    assert yaw_rate > 0.0


def test_stanley_closed_loop_converges_to_straight_route():
    state = BicycleState(0.0, 1.0, 0.0, 2.0, 0.0)
    for _ in range(200):
        command = steering_angle(
            heading_error_rad=-state.yaw,
            signed_offset_m=state.y,
            speed_mps=state.speed_mps,
            cross_track_gain=1.0,
            softening_mps=1.0,
        )
        state, _ = step_bicycle(
            state,
            target_speed_mps=2.0,
            target_steering_rad=command,
            dt=0.05,
            wheelbase_m=1.34,
            max_accel_mps2=1.0,
            max_decel_mps2=2.0,
            max_steering_rate_rad_s=math.radians(60.0),
        )
    assert abs(state.y) < 0.08
    assert abs(state.yaw) < math.radians(3.0)


def test_a_delayed_command_arrives_after_the_delay():
    from krowd_pp_stanleyff_control.bicycle_sim import DelayedCommand

    d = DelayedCommand(0.24)
    d.push(0.00, 0.1)
    d.push(0.10, 0.2)
    assert d.value(0.20) == 0.0  # nothing old enough yet
    assert d.value(0.24) == 0.1
    assert d.value(0.33) == 0.1
    assert d.value(0.34) == 0.2
    passthrough = DelayedCommand(0.0)
    passthrough.push(1.0, 0.3)
    assert passthrough.value(1.0) == 0.3


def test_the_achieved_speed_falls_short_but_never_below_zero():
    from krowd_pp_stanleyff_control.bicycle_sim import achieved_speed

    deficit = 1.7 / 3.6
    assert achieved_speed(10.0 / 3.6, deficit) == pytest.approx(8.3 / 3.6)
    assert 0.0 < achieved_speed(0.5, deficit) < 0.5  # scaled down at a crawl
    assert achieved_speed(0.0, deficit) == 0.0
    assert achieved_speed(10.0 / 3.6, 0.0) == pytest.approx(10.0 / 3.6)
