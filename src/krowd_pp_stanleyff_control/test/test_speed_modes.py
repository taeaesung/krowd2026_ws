"""Speed modes (config/speed_modes), the speed-dependent lookahead floor and the
yaw-damping loop-gain cap (2026-10-03)."""

from pathlib import Path

import pytest
import yaml

from krowd_pp_stanleyff_control.blended_steering import BlendParams, compute_steering, damping_gain
from krowd_pp_stanleyff_control.config import load_control_config, load_speed_mode
from krowd_pp_stanleyff_control.pure_pursuit import time_floored_lookahead
from krowd_pp_stanleyff_control.speed_profile import SpeedLimits

CONFIG = Path(__file__).resolve().parents[1] / "config"
MODES = {"low": (10.0, 15.0), "normal": (10.0, 25.0), "high": (10.0, 40.0)}  # corner 10 in every mode (2026-10-10)
KMH = 1.0 / 3.6


def base_profile() -> dict:
    return yaml.safe_load((CONFIG / "speed_profile.yaml").read_text())["route_publisher_node"]["ros__parameters"]


@pytest.mark.parametrize("mode", sorted(MODES))
def test_each_mode_sets_known_profile_parameters_and_validates(mode):
    data = load_speed_mode(CONFIG / "speed_modes" / f"{mode}.yaml")
    base = base_profile()
    params = data["route_publisher"]
    assert set(params) <= set(base), f"unknown speed_profile parameter in {mode}: {set(params) - set(base)}"
    corner, straight = MODES[mode]
    assert params["curve_max_speed_kmph"] == corner
    assert params["max_speed_kmph"] == straight
    merged = {**base, **params}
    SpeedLimits(
        max_speed_mps=merged["max_speed_kmph"] * KMH,
        max_lateral_accel_mps2=merged["max_lateral_accel_mps2"],
        max_accel_mps2=merged["max_accel_mps2"],
        max_decel_mps2=merged["max_decel_mps2"],
        min_speed_mps=merged["min_speed_kmph"] * KMH,
        curve_max_speed_mps=merged["curve_max_speed_kmph"] * KMH,
        curve_radius_m=merged["curve_radius_m"],
        tight_curve_radius_m=merged["tight_curve_radius_m"],
        curve_lead_m=merged["curve_speed_lead_m"],
        curve_exit_lead_m=merged["curve_exit_lead_m"],
    ).validate()


@pytest.mark.parametrize("mode", sorted(MODES))
def test_each_mode_lets_the_controller_reach_its_straight_speed(mode):
    cfg = load_control_config(CONFIG / "control.yaml", None, CONFIG / "speed_modes" / f"{mode}.yaml")
    straight = MODES[mode][1]
    speed = cfg["speed"]
    assert speed["cruise_kmh"] == straight
    # the speed trim may add trim_max_kmh on top of the target
    assert speed["max_kmh"] >= straight + speed["trim_max_kmh"]
    if not cfg["steering"]["table_verified"]:
        assert speed["cap_until_steering_verified_kmh"] >= straight


def test_the_overrides_file_is_merged_over_the_speed_mode():
    cfg = load_control_config(CONFIG / "control.yaml", CONFIG / "sim_overrides.yaml", CONFIG / "speed_modes" / "high.yaml")
    assert cfg["speed"]["cruise_kmh"] == 40.0  # from the mode
    assert cfg["speed"]["cap_until_steering_verified_kmh"] == 50.0  # sim_overrides.yaml wins


def test_lookahead_floor_only_lengthens():
    assert time_floored_lookahead(7.0, 40.0 * KMH, 0.6) == pytest.approx(7.0)
    assert time_floored_lookahead(3.0, 40.0 * KMH, 0.6) == pytest.approx(0.6 * 40.0 * KMH)
    assert time_floored_lookahead(3.0, 40.0 * KMH, 0.0) == pytest.approx(3.0)
    assert time_floored_lookahead(3.0, -2.0, 0.6) == pytest.approx(3.0)


def test_yaw_damping_gain_is_unchanged_to_25_kmh_and_capped_above():
    p = BlendParams(wheelbase_m=1.34, yaw_rate_damping_gain_s=0.15, yaw_rate_damping_max_loop_gain=0.78)
    assert damping_gain(p, 0.0) == pytest.approx(0.15)
    assert damping_gain(p, 25.0 * KMH) == pytest.approx(0.15)
    assert damping_gain(p, 40.0 * KMH) == pytest.approx(0.78 * 1.34 / (40.0 * KMH))
    assert damping_gain(p, 40.0 * KMH) * (40.0 * KMH) / 1.34 == pytest.approx(0.78)
    uncapped = BlendParams(wheelbase_m=1.34, yaw_rate_damping_gain_s=0.15)
    assert damping_gain(uncapped, 40.0 * KMH) == pytest.approx(0.15)


def test_compute_steering_applies_the_capped_damping():
    p = BlendParams(wheelbase_m=1.34, yaw_rate_damping_gain_s=0.15, yaw_rate_damping_max_loop_gain=0.78)
    v = 40.0 * KMH
    bd = compute_steering(0.0, 0.0, 0.0, v, 0.0, 0.1, p)  # straight, yawing left at 0.1 rad/s
    assert bd.yaw_damping == pytest.approx(-damping_gain(p, v) * 0.1)
