"""Traffic-light departure permit with an original-image-stamp TTL (8.1 G).

The detector publishes header-less Bool topics and a JSON String
(/traffic_light/detections_json) that carries the ORIGINAL image stamp, the
selected target, whether the frame was carried over from the other pass or
held, and the confirmed GO flags. This tracker rebuilds the permit from the
JSON and uses the Bool topics only as a veto.

A direction's permit is valid only when ALL hold:
  * the latest JSON says go_<dir>_confirmed == true and the latest Bool did
    not say false afterwards;
  * the last FRESH GO recognition (selected box detected this frame, not
    carried, not held, raw go_<dir> true) is younger than ttl_s, measured
    from its image stamp;
  * the latest JSON itself is younger than ttl_s;
  * the selected target has not jumped since the permit was confirmed
    (same locked target);
  * when a vehicle-signal ROI is required (M3 bus-lane exception), the locked
    target centre lies inside it - a missing ROI means no permit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json


DIRECTIONS = ("straight", "left")


@dataclass
class _DirectionState:
    confirmed: bool = False
    confirmed_since_ns: int | None = None
    last_fresh_go_ns: int | None = None
    lock_x: float | None = None
    lock_w: float | None = None
    identity_broken: bool = False
    bool_false_ns: int | None = None
    bool_true_ns: int | None = None


@dataclass(frozen=True)
class PermitDecision:
    valid: bool
    reason: str
    fresh_age_s: float | None
    json_age_s: float | None
    confirmed: bool
    target_x: float | None
    target_y: float | None
    target_w: float | None
    selection: str
    state: str


@dataclass(frozen=True)
class SignalObservation:
    """What the detector currently reports, independent of any direction."""

    alive: bool          # a detections_json message arrived within alive_timeout_s
    sees_red: bool       # the selected light reads red (fresh, carried or held)
    state: str
    fresh: bool
    json_age_s: float | None


@dataclass
class SignalPermitTracker:
    ttl_s: float
    target_jump_px: float
    dirs: dict = field(default_factory=lambda: {d: _DirectionState() for d in DIRECTIONS})
    last_json_stamp_ns: int | None = None
    last_json_recv_ns: int | None = None
    last_state: str = "unknown"
    last_selection: str = "none"
    last_target: tuple | None = None  # (x, y, w)
    last_fresh: bool = False
    parse_errors: int = 0

    # ---- ingestion --------------------------------------------------------
    def ingest_json(self, text: str, recv_ns: int) -> bool:
        try:
            payload = json.loads(text)
            stamp = payload["stamp"]
            stamp_ns = int(stamp["sec"]) * 1_000_000_000 + int(stamp["nanosec"])
        except (ValueError, KeyError, TypeError):
            self.parse_errors += 1
            return False
        self.last_json_stamp_ns = stamp_ns
        self.last_json_recv_ns = recv_ns
        self.last_state = str(payload.get("state", "unknown"))
        self.last_selection = str(payload.get("selection", "none"))
        held = bool(payload.get("held", False))
        selected = None
        for det in payload.get("detections", []) or []:
            if det.get("selected"):
                selected = det
                break
        fresh = selected is not None and not selected.get("carried", False) and not held
        self.last_fresh = fresh
        if selected is not None:
            cx, cy = selected.get("center_xy", [None, None])[:2]
            w = selected.get("size_wh", [None, None])[0]
            self.last_target = (cx, cy, w)
        else:
            self.last_target = None

        for direction in DIRECTIONS:
            st = self.dirs[direction]
            raw_go = bool(payload.get(f"go_{direction}", False))
            confirmed = bool(payload.get(f"go_{direction}_confirmed", False))
            if confirmed and not st.confirmed:
                st.confirmed_since_ns = stamp_ns
                st.identity_broken = False
                st.lock_x = self.last_target[0] if self.last_target else None
                st.lock_w = self.last_target[2] if self.last_target else None
            if not confirmed:
                st.lock_x = None
                st.lock_w = None
                st.identity_broken = False
                st.confirmed_since_ns = None
            st.confirmed = confirmed
            if confirmed and fresh and raw_go:
                st.last_fresh_go_ns = stamp_ns
            # Identity: a fresh selected box that jumped away from the lock.
            if confirmed and fresh and self.last_target and st.lock_x is not None:
                cx, _, w = self.last_target
                gate = max(self.target_jump_px, 2.0 * float(w or 0.0))
                if cx is not None and abs(float(cx) - st.lock_x) > gate:
                    st.identity_broken = True
        return True

    def ingest_confirmed_bool(self, direction: str, value: bool, recv_ns: int) -> None:
        st = self.dirs[direction]
        if value:
            st.bool_true_ns = recv_ns
        else:
            st.bool_false_ns = recv_ns

    def observation(self, now_ns: int, alive_timeout_s: float) -> SignalObservation:
        if self.last_json_recv_ns is None:
            return SignalObservation(False, False, "none", False, None)
        age = (now_ns - self.last_json_recv_ns) * 1e-9
        alive = age <= alive_timeout_s
        return SignalObservation(alive, alive and self.last_state == "red", self.last_state, self.last_fresh, age)

    # ---- decision ---------------------------------------------------------
    def decision(self, direction: str, now_ns: int, roi=None, roi_required: bool = False) -> PermitDecision:
        st = self.dirs[direction]
        tx, ty, tw = (self.last_target or (None, None, None))

        def out(valid, reason, fresh_age=None, json_age=None):
            return PermitDecision(
                valid, reason, fresh_age, json_age, st.confirmed, tx, ty, tw,
                self.last_selection, self.last_state,
            )

        if self.last_json_stamp_ns is None:
            return out(False, "NO_DETECTIONS_JSON")
        json_age = (now_ns - self.last_json_stamp_ns) * 1e-9
        if json_age > self.ttl_s:
            return out(False, f"JSON_STALE {json_age:.2f}s", None, json_age)
        if not st.confirmed:
            return out(False, "NOT_CONFIRMED", None, json_age)
        if st.bool_false_ns is not None and self.last_json_recv_ns is not None and st.bool_false_ns > self.last_json_recv_ns:
            return out(False, "BOOL_VETO_FALSE", None, json_age)
        if st.last_fresh_go_ns is None:
            return out(False, "NO_FRESH_GO", None, json_age)
        fresh_age = (now_ns - st.last_fresh_go_ns) * 1e-9
        if fresh_age > self.ttl_s:
            return out(False, f"FRESH_GO_EXPIRED {fresh_age:.2f}s", fresh_age, json_age)
        if st.identity_broken:
            return out(False, "TARGET_CHANGED", fresh_age, json_age)
        if roi_required:
            if roi is None:
                return out(False, "VEHICLE_SIGNAL_ROI_UNSET", fresh_age, json_age)
            if st.lock_x is None or tx is None or ty is None:
                return out(False, "NO_TARGET_FOR_ROI", fresh_age, json_age)
            x1, y1, x2, y2 = roi
            if not (x1 <= float(tx) <= x2 and y1 <= float(ty) <= y2):
                return out(False, "TARGET_OUTSIDE_VEHICLE_SIGNAL_ROI", fresh_age, json_age)
        return out(True, "GO", fresh_age, json_age)
