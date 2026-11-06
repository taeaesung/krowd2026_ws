from types import SimpleNamespace

import pytest

from krowd_pp_stanleyff_control import missions as M
from krowd_pp_stanleyff_control.config import deep_merge, front_offset_m, readiness
from helpers import straight_route


def table():
    return M.load_missions({"list": [
        {"id": "M1", "kind": "signal", "direction": "straight", "detect_idx": 50, "decision_idx": 64, "stop_idx": 68},
        {"id": "M2", "kind": "signal", "direction": "left", "detect_idx": 100, "decision_idx": 110, "stop_idx": 116,
         "pass_end_idx": 130, "require_vehicle_signal_roi": True},
        {"id": "M6", "kind": "turn", "direction": "left", "start_idx": 150, "end_idx": 160},
        {"id": "M8", "kind": "obstacle", "direction": None, "start_idx": None, "end_idx": None},
    ]})


def permit_factory(valid):
    def permit(direction, roi, roi_required):
        ok = valid and not (roi_required and roi is None)
        return SimpleNamespace(valid=ok, reason="GO" if ok else "NOT_CONFIRMED")
    return permit


def make_manager(route):
    resolved, warnings = M.resolve_missions(table(), route, accept_candidates=False)
    assert any("M8" in w for w in warnings)
    assert resolved[0].end_source == "fallback_next_mission" and resolved[0].end_s == pytest.approx(100.0)
    return M.MissionManager(resolved, cruise_mps=20 / 3.6, approach_mps=10 / 3.6, obstacle_mps=7 / 3.6,
                            turn_mps=None, stop_buffer_m=0.5, hold_distance_m=0.6)


def step(mgr, s, valid, stopped=False, front=2.5, latched=None):
    return mgr.step(M.MissionInputs(0, s, 2.0, front, stopped, permit_factory(valid), latched))


def test_stopped_short_of_line_while_braking_is_waiting():
    route = straight_route(200.0)
    mgr = make_manager(route)
    step(mgr, 55.0, False)
    out = step(mgr, 62.0, False, stopped=True, latched="M1")  # 3 m short of the stop point
    assert out.state == M.WAIT_SIGNAL and out.hold_brake
    out = step(mgr, 62.0, False, stopped=True, latched=None)  # stopped for another reason: keep approaching
    assert out.state == M.SIGNAL_DETECT


def test_signal_mission_lifecycle():
    route = straight_route(200.0)
    mgr = make_manager(route)
    assert step(mgr, 10.0, False).state == M.CRUISE_GLOBAL
    out = step(mgr, 55.0, False)
    assert out.state == M.SIGNAL_DETECT and out.stop_s == pytest.approx(68.0 - 2.5 - 0.5)
    out = step(mgr, 65.0, False)
    assert out.state == M.BRAKE_TO_STOP
    out = step(mgr, 64.6, False, stopped=True)  # at the line (d <= hold 0.6)
    assert out.state == M.WAIT_SIGNAL and out.hold_brake
    out = step(mgr, 64.6, True, stopped=True)
    assert out.state == M.RELEASE_TO_LINE and out.stop_s is None
    out = step(mgr, 65.6, True)  # front bumper 68.1 >= anchor 68 -> committed
    assert out.state == M.COMMITTED_CROSSING
    out = step(mgr, 70.0, False)  # signal lost after commit: no stop
    assert out.state == M.COMMITTED_CROSSING and out.stop_s is None
    out = step(mgr, 101.0, False)  # M1 done at fallback end (100), M2 active
    assert out.mission_id == "M2" and out.state == M.SIGNAL_DETECT
    assert "M1" in mgr.done


def test_overrun_without_permit_stops_and_roi_gate():
    route = straight_route(200.0)
    mgr = make_manager(route)
    step(mgr, 55.0, False)
    out = step(mgr, 66.0, False)  # bumper 68.5 past the anchor with no permit
    assert out.state == M.WAIT_SIGNAL and out.hold_brake and "OVERRUN" in out.reason
    for s in (101.0, 112.0):
        out = step(mgr, s, True)
    assert out.state == M.BRAKE_TO_STOP and out.stop_s is not None


def test_turn_and_ordering_errors():
    route = straight_route(200.0)
    mgr = make_manager(route)
    out = step(mgr, 155.0, False)
    assert out.state == M.TURN_GLOBAL and out.speed_cap_mps == pytest.approx(20 / 3.6)
    bad = M.load_missions({"list": [{"id": "X", "kind": "signal", "direction": "straight",
                                     "detect_idx": 70, "decision_idx": 64, "stop_idx": 68}]})
    with pytest.raises(ValueError):
        M.resolve_missions(bad, route, False)


def test_utm_cross_check_catches_shifted_indices():
    route = straight_route(200.0)  # waypoint i at x = i
    specs = M.load_missions({"list": [{
        "id": "M1", "kind": "signal", "direction": "straight",
        "detect_idx": 50, "decision_idx": 64, "stop_idx": 68, "stop_utm": [68.4, 0.1],
    }]})
    errors, checked = M.cross_check_missions(specs, route, lambda x, y: (x, y), 2.0)
    assert errors == [] and checked == 1
    shifted = M.load_missions({"list": [{
        "id": "M1", "kind": "signal", "direction": "straight",
        "detect_idx": 50, "decision_idx": 64, "stop_idx": 68, "stop_utm": [80.0, 0.0],
    }]})
    errors, _ = M.cross_check_missions(shifted, route, lambda x, y: (x, y), 2.0)
    assert len(errors) == 1 and "12.0 m apart" in errors[0]
    off_route = M.load_missions({"list": [{
        "id": "M1", "kind": "signal", "direction": "straight",
        "detect_idx": 50, "decision_idx": 64, "stop_idx": 68, "stop_utm": [68.0, 9.0],
    }]})
    errors, _ = M.cross_check_missions(off_route, route, lambda x, y: (x, y), 2.0)
    assert len(errors) == 1 and "off the route" in errors[0]
    with pytest.raises(ValueError):
        M.load_missions({"list": [{"id": "M1", "kind": "turn", "direction": None, "start_idx": 1, "end_idx": 2, "start_utm": [1.0]}]})


BASE = {
    "vehicle": {"wheelbase_m": 1.34, "length_m": 2.464, "width_m": 1.256, "front_axle_to_front_bumper_m": 1.194,
                "control_point_to_front_bumper_m": None, "control_point_to_rear_bumper_m": None},
    "braking": {"safe_deceleration_mps2": None, "total_latency_s": None, "distance_margin_m": None},
    "stopline": {"stop_buffer_m": None, "live_ttl_s": None, "map_live_association_tolerance_m": None},
    "corridor": {"localization_error_m": None, "map_error_m": None, "tracking_error_m": None, "physical_margin_m": None,
                 "fallback_half_width_m": 3.0},
    "signal": {"timeout_departure_s": None, "commit_mode": "last_stoppable_point"},
    "steering": {"table_verified": False, "max_road_wheel_rate_deg_s": 40.0, "zero_offset_rad": -0.21},
    "speed": {"cap_until_steering_verified_kmh": 10.0, "comfort_decel_mps2": 0.5,
              "start_ramp_accel_mps2": 1.0, "brake_release_hold_s": 0.8},
    "blend": {"max_stanley_weight": 0.5, "curve_radius_m": 5.0, "straight_radius_m": 50.0},
}
MISSION = {"route": {"frame_id": "map"}, "lane_boundaries": None, "missions": {"list": []}}


def test_readiness_blocks_on_nulls_and_conservative_front_offset():
    r = readiness(BASE, MISSION, check_files=False)
    assert not r.ok and len(r.blocking) == 8
    front, source = front_offset_m(BASE["vehicle"])
    assert front == pytest.approx(2.534) and source.startswith("conservative")
    filled = {**BASE,
              "braking": {"safe_deceleration_mps2": 2.0, "total_latency_s": 0.3, "distance_margin_m": 0.5},
              "stopline": {"stop_buffer_m": 0.3, "live_ttl_s": None, "map_live_association_tolerance_m": None},
              "corridor": {"localization_error_m": 0.1, "map_error_m": 0.2, "tracking_error_m": 0.2, "physical_margin_m": 0.1,
                           "fallback_half_width_m": 3.0}}
    assert readiness(filled, MISSION, check_files=False).ok
    with_timeout = {**filled, "signal": {"timeout_departure_s": 30.0, "commit_mode": "last_stoppable_point"}}
    assert readiness(with_timeout, MISSION, check_files=False).ok
    bad_timeout = {**filled, "signal": {"timeout_departure_s": -3.0}}
    assert not readiness(bad_timeout, MISSION, check_files=False).ok
    bad_blend = {**filled, "blend": {"max_stanley_weight": 1.5}}
    assert any("blend" in b for b in readiness(bad_blend, MISSION, check_files=False).blocking)
    avoidance = deep_merge(filled, {"avoidance_stanley": {
        "enabled": True, "mode_topic": "/integration_control/avoiding",
        "signal_timeout_s": 0.5, "k_heading": 1.0, "k_stanley": 1.5,
        "k_soft_mps": 0.5, "latency_compensation_s": 0.6,
        "heading_window_m": 0.6, "max_road_wheel_deg": 21.0,
        "max_road_wheel_rate_deg_s": 17.0,
    }})
    assert readiness(avoidance, MISSION, check_files=False).ok
    bad_avoidance = deep_merge(avoidance, {"avoidance_stanley": {"k_soft_mps": 0.0}})
    assert any(
        "avoidance_stanley.k_soft_mps" in b
        for b in readiness(bad_avoidance, MISSION, check_files=False).blocking
    )
    bad_segments = {"segments": {"list": [{"name": "a", "start_idx": 0, "end_idx": 5}, {"name": "b", "start_idx": 3, "end_idx": 9}]}}
    assert not readiness(filled, MISSION, bad_segments, check_files=False).ok
    # A route_only mission config (no lane map) is fine with file checks on.
    assert readiness(filled, MISSION, check_files=True).ok


def test_deep_merge_overrides_nested_keys_only():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 5}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 5}, "b": 3, "c": 4}


def make_manager_v2(route, **kw):
    resolved, _ = M.resolve_missions(table(), route, accept_candidates=False)
    args = dict(cruise_mps=20 / 3.6, approach_mps=10 / 3.6, obstacle_mps=7 / 3.6, turn_mps=None,
                stop_buffer_m=0.5, hold_distance_m=2.0, commit_distance_m=2.0)
    args.update(kw)
    return M.MissionManager(resolved, **args)


def step_v2(mgr, t_s, s, valid, stopped=False, latched=None, alive=True, red=False):
    return mgr.step(M.MissionInputs(int(t_s * 1e9), s, 2.0, 2.5, stopped, permit_factory(valid), latched, alive, red))


def test_permitted_approach_stays_stoppable_until_commit_point():
    mgr = make_manager_v2(straight_route(200.0))
    out = step_v2(mgr, 0, 55.0, True)
    assert out.state == M.SIGNAL_DETECT and out.stop_s is None
    assert out.approach_stop_s == pytest.approx(65.0) and out.commit_s == pytest.approx(63.0)
    out = step_v2(mgr, 1, 62.5, True)
    assert out.state == M.SIGNAL_DETECT and out.stop_s is None and out.approach_stop_s == pytest.approx(65.0)
    out = step_v2(mgr, 2, 62.8, False)
    assert out.stop_s == pytest.approx(65.0) and out.approach_stop_s is None
    out = step_v2(mgr, 3, 63.5, True)
    assert out.state == M.COMMITTED_CROSSING and out.permit_source == "detector"
    out = step_v2(mgr, 4, 65.0, False)
    assert out.state == M.COMMITTED_CROSSING and out.stop_s is None


def test_timeout_departure_counts_only_without_red_and_with_detector_alive():
    mgr = make_manager_v2(straight_route(200.0), timeout_departure_s=30.0)
    step_v2(mgr, 0, 55.0, False)
    out = step_v2(mgr, 10, 64.5, False, stopped=True, latched="M1")
    assert out.state == M.WAIT_SIGNAL and out.timeout_wait_s == pytest.approx(0.0) and out.timeout_limit_s == 30.0
    out = step_v2(mgr, 25, 64.5, False, stopped=True, latched="M1")
    assert out.timeout_wait_s == pytest.approx(15.0)
    out = step_v2(mgr, 26, 64.5, False, stopped=True, latched="M1", red=True)
    assert out.timeout_wait_s is None and "red light" in out.reason
    out = step_v2(mgr, 27, 64.5, False, stopped=True, latched="M1", alive=False)
    assert out.timeout_wait_s is None and "detector offline" in out.reason
    for t in range(28, 58):
        out = step_v2(mgr, t, 64.5, False, stopped=True, latched="M1")
    out = step_v2(mgr, 58, 64.5, False, stopped=True, latched="M1")
    assert out.state == M.RELEASE_TO_LINE and out.permit_source == "timeout" and out.stop_s is None


def test_initialize_at_skips_missions_behind_and_carry_over():
    route = straight_route(200.0)
    mgr = make_manager_v2(route)
    skipped = mgr.initialize_at(80.0, 2.5)
    assert skipped == ["M1"]
    out = step_v2(mgr, 0, 80.0, False)
    assert out.state == M.CRUISE_GLOBAL and out.mission_id is None
    mgr.committed.add("M2")
    rebuilt = make_manager_v2(route)
    rebuilt.carry_over(mgr)
    assert "M1" in rebuilt.done and "M2" in rebuilt.committed


def make_pause_manager(route, window_m=15.0, **kw):
    table = M.load_missions({"list": [
        {"id": "RT2", "kind": "pause", "direction": "right", "start_idx": 110, "stop_idx": 114},
        {"id": "P4", "kind": "turn", "direction": "right", "start_idx": 112, "end_idx": 128},
        {"id": "RT3", "kind": "pause", "direction": "right", "start_idx": 169, "stop_idx": 175, "hold_s": 5.0},
    ]})
    resolved, warnings = M.resolve_missions(table, route, accept_candidates=False, pause_window_m=window_m)
    args = dict(cruise_mps=20 / 3.6, approach_mps=10 / 3.6, obstacle_mps=7 / 3.6, turn_mps=None,
                stop_buffer_m=2.0, hold_distance_m=0.6, pause_hold_s=3.0, pause_stop_gap_m=0.5,
                pause_max_brake_s=5.0)
    args.update(kw)
    return resolved, warnings, M.MissionManager(resolved, **args)


def step_p(mgr, t_s, s, still=False, stop_m=4.0, v=2.0):
    """Pause input: standing still NOW and the stopping distance from the node."""
    return mgr.step(M.MissionInputs(int(t_s * 1e9), s, v, 2.5, still, permit_factory(False), None, True, False,
                                    still, stop_m))


def test_pause_resolves_with_window_and_no_overlap_warning():
    resolved, warnings, _ = make_pause_manager(straight_route(300.0))
    rt2 = resolved[0]
    assert rt2.mission_id == "RT2" and rt2.zone_start_s == pytest.approx(110.0)
    assert rt2.start_s == pytest.approx(95.0) and rt2.stop_anchor_s == pytest.approx(114.0)
    assert rt2.end_s == pytest.approx(114.0) and rt2.end_source == "pause_stop"
    assert [rm.mission_id for rm in resolved] == ["RT2", "P4", "RT3"]
    assert not any("overlaps" in w for w in warnings)
    with pytest.raises(ValueError):
        M.resolve_missions(M.load_missions({"list": [
            {"id": "X", "kind": "pause", "start_idx": 114, "stop_idx": 110}]}), straight_route(300.0), False)


def test_pause_keeps_speed_then_brakes_at_stopping_distance_holds_and_goes():
    _, _, mgr = make_pause_manager(straight_route(300.0))
    assert step_p(mgr, 0, 90.0).state == M.CRUISE_GLOBAL
    # front bumper (2.5 m) 0.5 m before #114 -> control point 111.0; no cap, no planner target
    out = step_p(mgr, 1, 100.0, stop_m=4.0)
    assert out.state == M.PAUSE_APPROACH and not out.hold_brake and out.stop_s is None
    assert out.speed_cap_mps == pytest.approx(20 / 3.6) and out.stop_control_s == pytest.approx(111.0)
    assert step_p(mgr, 2, 106.9, stop_m=4.0).state == M.PAUSE_APPROACH   # 4.1 m left > 4.0
    out = step_p(mgr, 3, 107.0, stop_m=4.0)                               # 4.0 m left: brake
    assert out.state == M.PAUSE_BRAKE and out.hold_brake
    out = step_p(mgr, 4, 109.0, stop_m=0.0)                               # braking holds while rolling
    assert out.state == M.PAUSE_BRAKE and out.hold_brake
    out = step_p(mgr, 5, 109.3, still=True)                               # standing still: count
    assert out.state == M.PAUSE_WAIT and out.hold_brake and out.timeout_wait_s == pytest.approx(0.0)
    assert step_p(mgr, 7.9, 109.3, still=True).state == M.PAUSE_WAIT
    out = step_p(mgr, 8.0, 109.3, still=True)                             # 3.0 s
    assert out.state == M.PAUSE_GO and not out.hold_brake and out.stop_control_s is None
    assert step_p(mgr, 9, 112.0).state == M.PAUSE_GO                      # stays released
    out = step_p(mgr, 10, 114.5)
    assert out.mission_id == "P4" and "RT2" in mgr.done


def test_pause_overrun_still_ends_in_a_stop_and_per_mission_hold():
    _, _, mgr = make_pause_manager(straight_route(300.0))
    mgr.done.update({"RT2", "P4"})
    assert step_p(mgr, 0, 160.0).mission_id == "RT3"
    out = step_p(mgr, 1, 171.5, stop_m=1.0)          # control point 172.0: 0.5 m left <= 1.0 -> brake
    assert out.state == M.PAUSE_BRAKE and out.hold_brake
    out = step_p(mgr, 2, 177.0, stop_m=0.0)          # rolled past #175 (end_s) while braking: still RT3
    assert out.mission_id == "RT3" and out.hold_brake
    step_p(mgr, 3, 177.2, still=True)
    assert step_p(mgr, 7.9, 177.2, still=True).state == M.PAUSE_WAIT    # RT3 hold_s 5.0
    assert step_p(mgr, 8.0, 177.2, still=True).state == M.PAUSE_GO
    assert step_p(mgr, 9, 178.0).mission_id is None and "RT3" in mgr.done


def test_pause_brake_without_standstill_times_out():
    _, _, mgr = make_pause_manager(straight_route(300.0))
    step_p(mgr, 0, 100.0)
    assert step_p(mgr, 1, 108.0, stop_m=4.0).state == M.PAUSE_BRAKE
    assert step_p(mgr, 5.9, 108.5, stop_m=0.0).state == M.PAUSE_BRAKE
    out = step_p(mgr, 6.0, 108.6, stop_m=0.0)
    assert out.state == M.PAUSE_GO and not out.hold_brake and mgr.pause_timeouts == 1


def test_pause_skipped_when_behind_and_carry_over():
    route = straight_route(300.0)
    _, _, mgr = make_pause_manager(route)
    assert "RT2" in mgr.initialize_at(112.0, 2.5)                  # front 114.5 is past #114
    _, _, mgr = make_pause_manager(route)
    mgr.paused.add("RT2")
    _, _, rebuilt = make_pause_manager(route)
    rebuilt.carry_over(mgr)
    assert "RT2" in rebuilt.paused


def test_signal_activates_early_when_stop_line_within_braking_distance():
    mgr = make_manager_v2(straight_route(200.0))
    # M1 detect at 50, stop control point 68 - 2.5 - 0.5 = 65. At s=30 the line is 35 m away.
    inp = lambda s, brake_m, valid: M.MissionInputs(0, s, 11.0, 2.5, False, permit_factory(valid), None, True, False, None, 0.0, brake_m)
    assert mgr.step(inp(30.0, 0.0, False)).state == M.CRUISE_GLOBAL          # feature off
    assert mgr.step(inp(30.0, 20.0, False)).state == M.CRUISE_GLOBAL         # 35 m > 20 m
    out = mgr.step(inp(40.0, 26.0, False))                                   # 25 m <= 26 m: red -> stop target now
    assert out.state == M.SIGNAL_DETECT and out.mission_id == "M1" and out.stop_s == pytest.approx(65.0)
    out = mgr.step(inp(40.0, 26.0, True))                                    # green -> no stop target
    assert out.state == M.SIGNAL_DETECT and out.stop_s is None
