import math

import pytest

from krowd_pp_stanleyff_control.braking import BrakingModel
from krowd_pp_stanleyff_control.steering_calibration import SteeringCalibration, equivalent_bicycle_angle

# The 2026-09-23 table (control.yaml): centred encoder column, zero offset -0.21.
TABLE = {
    "command_limit_rad": 2.5,
    "sign": 1.0,
    "table_verified": False,
    "zero_offset_rad": -0.21,
    "table_encoder_rad": [-2.50, -1.52, -0.78, 0.0, 1.21, 1.94, 2.96],
    "table_right_tire_deg": [-22.0, -12.7, -6.1, 0.0, 9.1, 13.8, 22.0],
    "table_left_tire_deg": [-21.0, -10.8, -5.6, 0.0, 10.4, 17.0, 29.0],
}


def test_equivalent_angle_between_inner_and_outer():
    d = math.degrees(equivalent_bicycle_angle(29.0, 22.0))
    assert 22.0 < d < 29.0
    assert equivalent_bicycle_angle(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        equivalent_bicycle_angle(10.0, -10.0)


def test_zero_offset_moves_the_straight_ahead_command():
    c = SteeringCalibration.from_config(TABLE)
    # Straight wheels are at RAW encoder -0.21, so that is what "0 deg" commands.
    assert c.command(0.0) == pytest.approx(-0.21)
    assert c.road_wheel_from_encoder(-0.21) == pytest.approx(0.0)
    # The old raw-zero table would read RAW 0.0 as straight; it is ~1.7 deg left.
    assert 1.5 < math.degrees(c.road_wheel_from_encoder(0.0)) < 2.0
    # Round trip inside the range.
    rw = math.radians(8.0)
    assert c.road_wheel_from_encoder(c.encoder_from_road_wheel(rw)) == pytest.approx(rw, abs=1e-9)


def test_limits_are_asymmetric_with_the_offset_and_commands_are_clamped():
    c = SteeringCalibration.from_config(TABLE)
    lo, hi = c.road_wheel_limits
    # +2.5 raw = 2.71 centred (left): between 17.0/13.8 and 29/22 rows;
    # -2.5 raw = -2.29 centred (right): between -12.7/-10.8 and -22/-21 rows.
    assert math.degrees(hi) == pytest.approx(22.6, abs=0.6)
    assert math.degrees(lo) == pytest.approx(-19.4, abs=0.6)
    assert c.command(1.0) == pytest.approx(2.5)
    assert c.command(-1.0) == pytest.approx(-2.5)
    assert c.command(math.radians(10.0)) > 0


def test_table_rejects_command_limit_outside_range():
    bad = dict(TABLE, command_limit_rad=3.0)
    with pytest.raises(ValueError):
        SteeringCalibration.from_config(bad)
    # Without the offset the raw range is [-2.50, 2.96] and 2.5 still fits.
    SteeringCalibration.from_config(dict(TABLE, zero_offset_rad=0.0))


def test_braking_envelope_uses_upper_bin_and_margin():
    b = BrakingModel.from_config({
        "safe_deceleration_mps2": 2.0,
        "total_latency_s": 0.3,
        "distance_margin_m": 0.5,
        "measured_envelope_kmh_m": [[5, 1.5], [10, 1.9], [15, 2.2], [20, 4.17], [30, 7.3]],
    })
    assert b.envelope_distance(7.0 / 3.6) == 1.9  # between 5 and 10 -> 10 km/h value
    assert b.envelope_distance(20.0 / 3.6) == 4.17
    assert b.required_distance(0.0) == pytest.approx(0.5 + 1.5)  # margin + smallest envelope
    v = 20.0 / 3.6
    assert b.required_distance(v) >= 4.17 + 0.5
    # speed_limit_for_distance is the inverse of required_distance.
    for d in (3.0, 6.0, 12.0):
        v_lim = b.speed_limit_for_distance(d)
        assert v_lim > 0.0
        assert b.required_distance(v_lim) <= d + 1e-6
        assert b.required_distance(v_lim + 0.05) > d
    # Below the smallest envelope bin + margin nothing may move.
    assert b.speed_limit_for_distance(1.0) == 0.0
    assert b.speed_limit_for_distance(0.2) == 0.0
