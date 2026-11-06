import math

import pytest

from krowd_pp_stanleyff_control.segments import SegmentParams, SegmentTable, load_segments, resolve_segments
from krowd_pp_stanleyff_control.speed_planner import anticipatory_cap, anticipatory_step, comfort_speed, step_cap


def test_deceleration_caps_are_steps_not_ramps():
    """Where a comfort ramp would bind, the floor is commanded whole (2026-10-01)."""

    v_cruise = 25 / 3.6
    # 100 m from a 10 km/h zone the ramp (sqrt(2.78^2 + 2*0.5*100) = 10.4 m/s) is above cruise: no cap.
    assert anticipatory_step(0.0, [(100.0, 10 / 3.6)], 0.5, v_cruise) == math.inf
    # 20 m before it the ramp (5.4 m/s) binds: the zone speed itself, at once.
    assert anticipatory_step(0.0, [(20.0, 10 / 3.6)], 0.5, v_cruise) == pytest.approx(10 / 3.6)
    assert step_cap(5.0, 5 / 3.6, v_cruise) == pytest.approx(5 / 3.6)
    assert step_cap(8.0, 5 / 3.6, v_cruise) == math.inf
from helpers import straight_route

DEFAULTS = SegmentParams("global", None, 7.0, 3.0, 50.0, 15.0, math.radians(60.0))


def test_segments_merge_and_lookup():
    specs = load_segments({"list": [
        {"name": "slow", "start_idx": 10, "end_idx": 19, "speed_kmh": 8.0, "lookahead_curve_m": 2.0},
        {"name": "fast", "start_idx": 30, "end_idx": 40, "max_road_wheel_rate_deg_s": 30.0},
    ]})
    route = straight_route(100.0)
    table = SegmentTable(resolve_segments(specs, route), DEFAULTS)
    assert table.params_at(5.0) is DEFAULTS
    p = table.params_at(12.0)
    assert p.name == "slow" and p.speed_cap_mps == pytest.approx(8.0 / 3.6)
    assert p.lookahead_curve_m == 2.0 and p.lookahead_straight_m == 7.0 and p.straight_radius_m == 50.0
    assert table.params_at(19.9).name == "slow" and table.params_at(20.0).name == "global"
    assert table.params_at(35.0).max_road_wheel_rate_rad_s == pytest.approx(math.radians(30.0))
    assert table.upcoming_speed_caps(5.0, 100.0) == [(10.0, pytest.approx(8.0 / 3.6))]
    assert table.upcoming_speed_caps(15.0, 100.0) == []


def test_segments_reject_overlap_and_bad_values():
    with pytest.raises(ValueError):
        load_segments({"list": [{"name": "a", "start_idx": 0, "end_idx": 10}, {"name": "b", "start_idx": 10, "end_idx": 20}]})
    with pytest.raises(ValueError):
        load_segments({"list": [{"name": "a", "start_idx": 0, "end_idx": 10, "speed_kmh": -1}]})
    with pytest.raises(ValueError):
        resolve_segments(load_segments({"list": [{"name": "a", "start_idx": 0, "end_idx": 500}]}), straight_route(100.0))


def test_comfort_and_anticipatory_caps():
    assert comfort_speed(0.0, 1.0, 0.5) == pytest.approx(1.0)
    assert comfort_speed(4.0, 0.0, 0.5) == pytest.approx(2.0)
    # 20 km/h zone ends in an approach zone (10 km/h) 30 m ahead: still 20 km/h far away,
    # then a sqrt profile down to 10 km/h at the boundary.
    v_far = anticipatory_cap(0.0, [(200.0, 10 / 3.6)], 0.5)
    assert v_far > 20 / 3.6
    v_near = anticipatory_cap(0.0, [(2.0, 10 / 3.6)], 0.5)
    assert 10 / 3.6 < v_near < 11.5 / 3.6
    assert anticipatory_cap(50.0, [(10.0, 1.0)], 0.5) == math.inf  # boundaries behind are ignored
