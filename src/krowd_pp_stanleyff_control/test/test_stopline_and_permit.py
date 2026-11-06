import json
from types import SimpleNamespace

import pytest

from krowd_pp_stanleyff_control.signal_permit import SignalPermitTracker
from krowd_pp_stanleyff_control.stopline import StoplineEstimator


def marker(points, mid=0, action=0, mtype=4):
    return SimpleNamespace(
        action=action, type=mtype, id=mid,
        points=[SimpleNamespace(x=x, y=y, z=0.0) for x, y in points],
    )


def test_stopline_candidate_selection_and_ttl():
    est = StoplineEstimator(half_band_m=0.628, min_span_m=1.0, max_abs_slope=0.5, max_distance_m=30.0)
    markers = [
        SimpleNamespace(action=3, type=4, id=0, points=[]),  # DELETEALL
        marker([(12.0, -1.5), (12.1, 0.0), (12.0, 1.5)], mid=1),  # good, 12 m
        marker([(6.0, 3.0), (6.0, 4.5)], mid=2),  # no lateral overlap
        marker([(4.0, -1.0), (7.0, 1.0)], mid=3),  # too steep (lane line)
        marker([(-2.0, -1.0), (-2.0, 1.0)], mid=4),  # behind
    ]
    obs = est.ingest(markers, stamp_ns=1_000_000_000, route_s=100.0)
    assert obs.marker_id == 1
    assert obs.distance_m == pytest.approx(12.0)
    e = est.estimate(1_100_000_000, ttl_s=0.3, route_s_now=100.8)
    assert e.valid and e.distance_m == pytest.approx(11.2)
    e = est.estimate(1_500_000_000, ttl_s=0.3, route_s_now=101.0)
    assert not e.valid and e.reason == "STALE"
    assert not est.estimate(1_100_000_000, None, 100.0).valid


def payload(stamp_s, go_straight_confirmed=False, go_straight=False, selected_x=500.0, carried=False, held=False,
            state="green", go_left=False, go_left_confirmed=False):
    det = {"selected": True, "center_xy": [selected_x, 120.0], "size_wh": [20.0, 8.0], "confidence": 0.8,
           "signal": {"state": state}}
    if carried:
        det["carried"] = True
    return json.dumps({
        "stamp": {"sec": int(stamp_s), "nanosec": int((stamp_s % 1) * 1e9)},
        "state": state, "selection": "largest", "held": held, "pass": "full",
        "go_straight": go_straight, "go_left": go_left,
        "go_straight_confirmed": go_straight_confirmed, "go_left_confirmed": go_left_confirmed,
        "detections": [det],
    })


def ns(seconds):
    return int(seconds * 1e9)


def test_permit_requires_confirmed_fresh_go_within_ttl():
    t = SignalPermitTracker(ttl_s=0.25, target_jump_px=60.0)
    assert not t.decision("straight", ns(10.0)).valid
    t.ingest_json(payload(10.00, False, True), ns(10.01))
    assert t.decision("straight", ns(10.02)).reason == "NOT_CONFIRMED"
    t.ingest_json(payload(10.05, True, True), ns(10.06))
    d = t.decision("straight", ns(10.10))
    assert d.valid and d.reason == "GO"
    # Only carried frames afterwards: the fresh stamp ages out even though JSON keeps coming.
    t.ingest_json(payload(10.20, True, True, carried=True), ns(10.21))
    assert t.decision("straight", ns(10.25)).valid
    t.ingest_json(payload(10.32, True, True, carried=True), ns(10.33))
    d = t.decision("straight", ns(10.33))
    assert not d.valid and d.reason.startswith("FRESH_GO_EXPIRED")
    # Stale JSON alone invalidates.
    t.ingest_json(payload(10.40, True, True), ns(10.41))
    assert t.decision("straight", ns(10.45)).valid
    assert t.decision("straight", ns(10.70)).reason.startswith("JSON_STALE")


def test_permit_target_identity_and_bool_veto_and_roi():
    t = SignalPermitTracker(ttl_s=0.25, target_jump_px=60.0)
    t.ingest_json(payload(1.00, True, True, selected_x=500.0), ns(1.01))
    assert t.decision("straight", ns(1.05)).valid
    t.ingest_json(payload(1.05, True, True, selected_x=650.0), ns(1.06))
    assert t.decision("straight", ns(1.08)).reason == "TARGET_CHANGED"
    # Re-confirmation after a drop resets the lock.
    t.ingest_json(payload(1.10, False, False, selected_x=650.0), ns(1.11))
    t.ingest_json(payload(1.15, True, True, selected_x=650.0), ns(1.16))
    assert t.decision("straight", ns(1.18)).valid
    t.ingest_confirmed_bool("straight", False, ns(1.19))
    assert t.decision("straight", ns(1.20)).reason == "BOOL_VETO_FALSE"
    # Left turn with a required ROI.
    t.ingest_json(payload(2.00, go_left=True, go_left_confirmed=True, selected_x=620.0), ns(2.01))
    assert t.decision("left", ns(2.05), None, True).reason == "VEHICLE_SIGNAL_ROI_UNSET"
    assert t.decision("left", ns(2.05), (300, 0, 500, 300), True).reason == "TARGET_OUTSIDE_VEHICLE_SIGNAL_ROI"
    assert t.decision("left", ns(2.05), (550, 0, 700, 300), True).valid
    # A straight green is never a left permit.
    t.ingest_json(payload(2.10, True, True, selected_x=620.0), ns(2.11))
    assert not t.decision("left", ns(2.12)).valid
