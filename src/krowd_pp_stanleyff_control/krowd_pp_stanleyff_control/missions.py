"""Waypoint mission table and the mission state machine (section 8.1 B/C/G).

Missions are defined by waypoint INDICES of the profile's route; they are
resolved to arc length `s` from the live Trajectory, so gating uses the
monotonic route position and never a Euclidean radius.

Every `<point>_idx` may carry a sibling `<point>_utm: [x, y]` (UTM zone of
the map). The node transforms it into the route frame and checks that it
projects within `utm_cross_check_tolerance_m` of the index's arc position -
a plain-language diagnosis ("M1.stop is 12.4 m from where #68 is") when a
re-saved waypoints.csv has shifted the indices. The index stays the truth;
the coordinate only checks it.

State machine (per signal mission):

    CRUISE_GLOBAL -> SIGNAL_DETECT (s >= detect) -> SIGNAL_DECIDE (s >= decision)
      -> BRAKE_TO_STOP (no valid permit)  -> WAIT_SIGNAL (stopped at the line)
      -> RELEASE_TO_LINE (valid permit; still re-checked every cycle)
      -> COMMITTED_CROSSING (front bumper past the stop anchor WITH a valid
         permit; latched, signal loss is no longer a stop)
      -> MISSION_DONE at pass_end (latched, never re-activates)

`pass_end` that the user has not confirmed is NOT invented: the mission keeps
its approach speed cap until the next mission's detect point (or the route
end). Candidates are used only with accept_pass_end_candidates.

Pause mission (kind: pause, 2026-10-17 - stop before a right turn): a stop
zone [start_idx, stop_idx], stopped the way the avoidance zone brake stops
(zone_brake.py, user decision): no slow-down before it. From `brake_window_m`
before the zone the vehicle keeps its speed until the stop target (front bumper
`stop_gap_m` before stop_idx) is within its stopping distance, then brakes to a
standstill, holds it for hold_s and goes. Once per run, no signal involved.

    PAUSE_APPROACH -> PAUSE_BRAKE -> PAUSE_WAIT (standing still, counting hold_s)
      -> PAUSE_GO -> MISSION_DONE when the vehicle reaches stop_idx
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

SIGNAL = "signal"
TURN = "turn"
OBSTACLE = "obstacle"
PAUSE = "pause"
KINDS = (SIGNAL, TURN, OBSTACLE, PAUSE)

CRUISE_GLOBAL = "CRUISE_GLOBAL"
SIGNAL_DETECT = "SIGNAL_DETECT"
SIGNAL_DECIDE = "SIGNAL_DECIDE"
BRAKE_TO_STOP = "BRAKE_TO_STOP"
WAIT_SIGNAL = "WAIT_SIGNAL"
RELEASE_TO_LINE = "RELEASE_TO_LINE"
COMMITTED_CROSSING = "COMMITTED_CROSSING"
TURN_GLOBAL = "TURN_GLOBAL"
OBSTACLE_RESERVED = "OBSTACLE_RESERVED"
PAUSE_APPROACH = "PAUSE_APPROACH"
PAUSE_BRAKE = "PAUSE_BRAKE"
PAUSE_WAIT = "PAUSE_WAIT"
PAUSE_GO = "PAUSE_GO"
MISSION_DONE = "MISSION_DONE"

POINT_KEYS = ("detect", "decision", "stop", "signal_lost", "pass_end", "pass_end_candidate", "start", "end")


@dataclass(frozen=True)
class Mission:
    mission_id: str
    kind: str
    direction: Optional[str]
    detect_idx: Optional[int] = None
    decision_idx: Optional[int] = None
    stop_idx: Optional[int] = None
    signal_lost_idx: Optional[int] = None
    pass_end_idx: Optional[int] = None
    pass_end_candidate_idx: Optional[int] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    require_vehicle_signal_roi: bool = False
    vehicle_signal_roi: Optional[tuple] = None
    # Timeout departure: seconds of "signal not detected" while waiting at the
    # line before departing (None = inherit the global signal.timeout_departure_s).
    timeout_departure_s: Optional[float] = None
    allow_timeout_departure: bool = True
    # Pause missions: seconds to stand still (None = the global pause.hold_s).
    hold_s: Optional[float] = None
    notes: str = ""
    # Optional UTM coordinates of the index points, for the cross-check.
    utm_points: tuple = ()  # ((key, x, y), ...)

    @property
    def enabled(self) -> bool:
        if self.kind == SIGNAL:
            return None not in (self.detect_idx, self.decision_idx, self.stop_idx)
        if self.kind == PAUSE:
            return None not in (self.start_idx, self.stop_idx)
        return None not in (self.start_idx, self.end_idx)


def _opt_int(value) -> Optional[int]:
    return None if value is None else int(value)


def _utm_points(raw: dict, mission_id) -> tuple:
    points = []
    for key in POINT_KEYS:
        value = raw.get(f"{key}_utm")
        if value is None:
            continue
        try:
            x, y = (float(v) for v in value)
        except (TypeError, ValueError):
            raise ValueError(f"mission {mission_id}: {key}_utm must be [x, y]")
        points.append((key, x, y))
    return tuple(points)


def load_missions(cfg: dict) -> list[Mission]:
    missions = []
    for raw in cfg.get("list", []) or []:
        kind = str(raw["kind"])
        if kind not in KINDS:
            raise ValueError(f"mission {raw.get('id')}: unknown kind '{kind}'")
        direction = raw.get("direction")
        direction = None if direction is None else str(direction)
        if kind == SIGNAL and direction not in ("straight", "left"):
            raise ValueError(f"mission {raw.get('id')}: signal direction must be straight/left")
        roi = raw.get("vehicle_signal_roi")
        if roi is not None:
            roi = tuple(float(v) for v in roi)
            if len(roi) != 4 or roi[0] >= roi[2] or roi[1] >= roi[3]:
                raise ValueError(f"mission {raw.get('id')}: vehicle_signal_roi must be [x1,y1,x2,y2]")
        missions.append(
            Mission(
                mission_id=str(raw["id"]),
                kind=kind,
                direction=direction,
                detect_idx=_opt_int(raw.get("detect_idx")),
                decision_idx=_opt_int(raw.get("decision_idx")),
                stop_idx=_opt_int(raw.get("stop_idx")),
                signal_lost_idx=_opt_int(raw.get("signal_lost_idx")),
                pass_end_idx=_opt_int(raw.get("pass_end_idx")),
                pass_end_candidate_idx=_opt_int(raw.get("pass_end_candidate_idx")),
                start_idx=_opt_int(raw.get("start_idx")),
                end_idx=_opt_int(raw.get("end_idx")),
                require_vehicle_signal_roi=bool(raw.get("require_vehicle_signal_roi", False)),
                vehicle_signal_roi=roi,
                timeout_departure_s=(
                    None if raw.get("timeout_departure_s") is None else float(raw["timeout_departure_s"])
                ),
                allow_timeout_departure=bool(raw.get("allow_timeout_departure", True)),
                hold_s=None if raw.get("hold_s") is None else float(raw["hold_s"]),
                notes=str(raw.get("notes", "")),
                utm_points=_utm_points(raw, raw.get("id")),
            )
        )
    ids = [m.mission_id for m in missions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate mission ids: {ids}")
    return missions


def cross_check_missions(
    missions: list[Mission],
    route,
    to_route_frame: Callable[[float, float], tuple[float, float]],
    tolerance_m: float,
) -> tuple[list[str], int]:
    """Compare every `<key>_utm` with its `<key>_idx` on `route`: (errors, points checked)."""

    errors: list[str] = []
    checked = 0
    for m in missions:
        for key, ux, uy in m.utm_points:
            idx = getattr(m, f"{key}_idx")
            if idx is None or not 0 <= idx < route.count:
                continue
            checked += 1
            x, y = to_route_frame(ux, uy)
            proj = route.project(x, y, None, None)
            s_idx = route.s_at_index(idx)
            if proj is None or proj.distance > tolerance_m:
                errors.append(
                    f"{m.mission_id}.{key}_utm ({ux:.1f}, {uy:.1f}) lies "
                    f"{'nowhere near' if proj is None else f'{proj.distance:.1f} m off'} the route"
                )
                continue
            if abs(proj.s - s_idx) > tolerance_m:
                errors.append(
                    f"{m.mission_id}.{key}: #{idx} is at s={s_idx:.1f} m but {key}_utm projects to "
                    f"s={proj.s:.1f} m ({abs(proj.s - s_idx):.1f} m apart) - the waypoint indices do not "
                    "match this waypoints.csv"
                )
    return errors, checked


@dataclass(frozen=True)
class ResolvedMission:
    mission: Mission
    start_s: float  # detect_s for signals, start_s for turn/obstacle, brake-window start for pause
    decision_s: Optional[float]
    stop_anchor_s: Optional[float]
    end_s: float  # pass_end / end (pause: the stop anchor)
    end_source: str  # confirmed | candidate | fallback_next_mission | fallback_route_end | pause_stop
    zone_start_s: Optional[float] = None  # pause: start_idx (the zone itself; start_s is brake_window_m earlier)

    @property
    def mission_id(self) -> str:
        return self.mission.mission_id

    @property
    def kind(self) -> str:
        return self.mission.kind


def resolve_missions(
    missions: list[Mission], route, accept_candidates: bool, pause_window_m: float = 0.0
) -> tuple[list[ResolvedMission], list[str]]:
    """Turn indices into arc lengths on `route` (RouteModel). Returns (resolved, warnings).

    A pause mission becomes active `pause_window_m` before its zone (never
    before the previous mission's start): from there the brake check runs.
    """

    warnings: list[str] = []
    enabled = [m for m in missions if m.enabled]
    for m in missions:
        if not m.enabled:
            warnings.append(f"{m.mission_id}: waypoint indices incomplete -> mission DISABLED")
    for m in enabled:
        for name in ("detect_idx", "decision_idx", "stop_idx", "pass_end_idx", "start_idx", "end_idx", "signal_lost_idx"):
            value = getattr(m, name)
            if value is not None and not 0 <= value < route.count:
                raise ValueError(f"{m.mission_id}.{name}={value} outside route 0..{route.count - 1}")
        if m.kind == SIGNAL:
            if not m.detect_idx < m.decision_idx < m.stop_idx:
                raise ValueError(f"{m.mission_id}: need detect < decision < stop indices")
            if m.pass_end_idx is not None and m.pass_end_idx <= m.stop_idx:
                raise ValueError(f"{m.mission_id}: pass_end must lie after the stop anchor")
        elif m.kind == PAUSE:
            if m.start_idx >= m.stop_idx:
                raise ValueError(f"{m.mission_id}: pause needs start_idx < stop_idx")
            if m.hold_s is not None and m.hold_s < 0.0:
                raise ValueError(f"{m.mission_id}: hold_s must be >= 0")
        elif m.start_idx >= m.end_idx:
            raise ValueError(f"{m.mission_id}: start_idx must be < end_idx")

    ordered = sorted(enabled, key=lambda m: m.detect_idx if m.kind == SIGNAL else m.start_idx)
    resolved: list[ResolvedMission] = []
    for i, m in enumerate(ordered):
        next_start_idx = None
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            next_start_idx = nxt.detect_idx if nxt.kind == SIGNAL else nxt.start_idx
        if m.kind == SIGNAL:
            start_s = route.s_at_index(m.detect_idx)
            decision_s = route.s_at_index(m.decision_idx)
            stop_s = route.s_at_index(m.stop_idx)
            if m.pass_end_idx is not None:
                end_s, source = route.s_at_index(m.pass_end_idx), "confirmed"
            elif accept_candidates and m.pass_end_candidate_idx is not None:
                end_s, source = route.s_at_index(m.pass_end_candidate_idx), "candidate"
                warnings.append(f"{m.mission_id}: using UNCONFIRMED pass_end candidate #{m.pass_end_candidate_idx}")
            elif next_start_idx is not None:
                end_s, source = route.s_at_index(next_start_idx), "fallback_next_mission"
                warnings.append(
                    f"{m.mission_id}: pass_end unconfirmed -> approach speed cap held until next mission "
                    f"start #{next_start_idx}"
                )
            else:
                end_s, source = route.length, "fallback_route_end"
                warnings.append(f"{m.mission_id}: pass_end unconfirmed -> approach speed cap held to route end")
            if end_s <= stop_s:
                raise ValueError(f"{m.mission_id}: resolved end {end_s:.1f} m not after stop anchor {stop_s:.1f} m")
        elif m.kind == PAUSE:
            zone_start_s = route.s_at_index(m.start_idx)
            floor = resolved[-1].start_s if resolved else 0.0
            start_s = max(zone_start_s - max(pause_window_m, 0.0), floor, 0.0)
            decision_s = None
            stop_s = route.s_at_index(m.stop_idx)
            end_s, source = stop_s, "pause_stop"
            resolved.append(ResolvedMission(m, start_s, decision_s, stop_s, end_s, source, zone_start_s))
            continue
        else:
            start_s = route.s_at_index(m.start_idx)
            decision_s = None
            stop_s = None
            end_s, source = route.s_at_index(m.end_idx), "confirmed"
        if m.kind == SIGNAL and m.require_vehicle_signal_roi and m.vehicle_signal_roi is None:
            warnings.append(
                f"{m.mission_id}: vehicle-signal ROI required but unset -> autonomous departure DISABLED (STOP default)"
            )
        resolved.append(ResolvedMission(m, start_s, decision_s, stop_s, end_s, source))

    for prev, nxt in zip(resolved, resolved[1:]):
        if nxt.start_s < prev.start_s:
            raise ValueError("missions are not in increasing route order")
        if nxt.start_s < prev.end_s - 1e-6 and prev.end_source == "confirmed":
            warnings.append(f"{prev.mission_id} end overlaps {nxt.mission_id} start")
    return resolved, warnings


@dataclass
class MissionInputs:
    now_ns: int
    route_s: float
    speed_mps: float
    front_offset_m: float
    stopped: bool
    permit: Callable  # permit(direction: str, roi, roi_required: bool) -> PermitDecision
    # Mission id the node's brake latch is currently holding for (None if not braking
    # for a mission stop). A vehicle that came to rest while braking for THIS mission
    # is waiting for the signal even if the model stopped it short of the line.
    brake_latched_for: Optional[str] = None
    # Detector liveness and whether a RED light is currently reported (fresh,
    # carried or held): a visible red always resets the timeout-departure timer.
    signal_alive: bool = False
    sees_red: bool = False
    # Pause missions (zone-brake style): every speed source at a standstill NOW
    # (None = use `stopped`), and the distance the vehicle needs to stop from
    # its current speed (measured envelope + pause.stop_margin_m).
    standing_still: Optional[bool] = None
    stop_distance_m: float = 0.0
    # Signal missions: distance within which the next signal's stop point is
    # already a stop target BEFORE detect_idx (node: braking.required_distance(v)
    # + signal.pre_detect_margin_m). 0 = only from detect_idx (2026-10-17: with no
    # approach cap the vehicle reached detect at 40 km/h with 7-19 m left and overran).
    signal_brake_m: float = 0.0


@dataclass
class MissionOutput:
    state: str
    mission_id: Optional[str]
    kind: Optional[str]
    direction: Optional[str]
    speed_cap_mps: Optional[float]
    stop_s: Optional[float]  # control-point arc position where the vehicle must be at rest
    hold_brake: bool
    permit: object
    reason: str
    signal_active: bool
    distance_to_stop_control_m: Optional[float] = None
    end_s: Optional[float] = None
    stop_control_s: Optional[float] = None   # control-point arc position of the stop anchor (signal missions, until committed)
    stop_anchor_s: Optional[float] = None
    # Permit valid but not yet committed: keep the speed STOPPABLE for this
    # target (speed cap only, no brake) until commit_s is reached.
    approach_stop_s: Optional[float] = None
    commit_s: Optional[float] = None
    permit_source: str = ""              # "detector" | "timeout" | ""
    # Signal: seconds waited toward a timeout departure / its limit.
    # Pause: seconds stood still at the stop / hold_s.
    timeout_wait_s: Optional[float] = None
    timeout_limit_s: Optional[float] = None


class MissionManager:
    def __init__(
        self,
        resolved: list[ResolvedMission],
        *,
        cruise_mps: float,
        approach_mps: float,
        obstacle_mps: float,
        turn_mps: Optional[float],
        stop_buffer_m: float,
        hold_distance_m: float,
        commit_distance_m: float = 0.0,
        commit_mode: str = "last_stoppable_point",
        timeout_departure_s: Optional[float] = None,
        timeout_requires_alive: bool = True,
        pause_hold_s: float = 3.0,
        pause_stop_gap_m: float = 0.5,
        pause_max_brake_s: float = 5.0,
    ):
        self.missions = list(resolved)
        self.cruise = cruise_mps
        self.approach = approach_mps
        self.obstacle = obstacle_mps
        self.turn = turn_mps
        self.stop_buffer = stop_buffer_m
        self.hold_distance = hold_distance_m
        # Commit point: with a valid permit the vehicle is committed to cross
        # once its stop-control point is within commit_distance_m of the stop
        # target ("last_stoppable_point") or when the front bumper is at the
        # line ("front_bumper_at_line", the strict 8.1 C reading).
        if commit_mode not in ("last_stoppable_point", "front_bumper_at_line"):
            raise ValueError(f"unknown commit_mode '{commit_mode}'")
        self.commit_distance = max(float(commit_distance_m), 0.0)
        self.commit_mode = commit_mode
        self.timeout_departure_s = timeout_departure_s
        self.timeout_requires_alive = bool(timeout_requires_alive)
        if pause_hold_s < 0.0 or pause_max_brake_s <= 0.0:
            raise ValueError(f"pause needs hold_s >= 0 and max_brake_s > 0: {pause_hold_s}, {pause_max_brake_s}")
        self.pause_hold = float(pause_hold_s)
        # Front bumper distance before a pause's stop_idx (the signal stop uses
        # stop_buffer_m): smaller = closer to the line.
        self.pause_gap = float(pause_stop_gap_m)
        self.pause_max_brake_ns = int(pause_max_brake_s * 1e9)
        self.done: set[str] = set()
        self.committed: set[str] = set()
        self.waited: set[str] = set()
        self.timeout_permits: set[str] = set()
        self.timeout_since: dict[str, int] = {}
        self.paused: set[str] = set()          # pause missions whose hold has been served
        self.pause_since: dict[str, int] = {}  # standstill start at a pause (ns)
        self.pause_brake_since: dict[str, int] = {}  # pause brake running since (ns)
        self.pause_timeouts = 0                # pause brakes ended by max_brake_s, not by a standstill
        self.state = CRUISE_GLOBAL
        self.active_id: Optional[str] = None
        self.events: list[str] = []

    def _event(self, text: str) -> None:
        self.events.append(text)
        if len(self.events) > 50:
            self.events.pop(0)

    def reset_progress(self) -> None:
        """Only for a fresh run from the start line; latched progress is otherwise kept."""

        self.done.clear()
        self.committed.clear()
        self.waited.clear()
        self.timeout_permits.clear()
        self.timeout_since.clear()
        self.paused.clear()
        self.pause_since.clear()
        self.pause_brake_since.clear()
        self.state = CRUISE_GLOBAL
        self.active_id = None

    def carry_over(self, previous: "MissionManager | None") -> None:
        """Keep latched progress across a rebuild (config reload, new trajectory id)."""

        if previous is None:
            return
        ids = {rm.mission_id for rm in self.missions}
        self.done = {m for m in previous.done if m in ids}
        self.committed = {m for m in previous.committed if m in ids}
        self.waited = {m for m in previous.waited if m in ids}
        self.timeout_permits = {m for m in previous.timeout_permits if m in ids}
        self.timeout_since = {m: t for m, t in previous.timeout_since.items() if m in ids}
        self.paused = {m for m in previous.paused if m in ids}
        self.pause_since = {m: t for m, t in previous.pause_since.items() if m in ids}
        self.pause_brake_since = {m: t for m, t in previous.pause_brake_since.items() if m in ids}

    def initialize_at(self, route_s: float, front_offset_m: float) -> list[str]:
        """Mark missions whose stop line is already behind the vehicle as done.

        Starting mid-course (a test from waypoint 80) must not re-activate M1 as
        an 'overrun without permit' stop. Returns the ids skipped.
        """

        skipped = []
        for rm in self.missions:
            if rm.mission_id in self.done:
                continue
            behind = (
                route_s + front_offset_m >= rm.stop_anchor_s
                if rm.kind in (SIGNAL, PAUSE)
                else route_s >= rm.end_s
            )
            if behind and rm.mission_id not in self.committed:
                self.done.add(rm.mission_id)
                skipped.append(rm.mission_id)
        if skipped:
            self._event(f"initialized at s={route_s:.1f}: skipped {skipped} (already behind the vehicle)")
        return skipped

    def _next_pending(self) -> Optional[ResolvedMission]:
        for rm in self.missions:
            if rm.mission_id not in self.done:
                return rm
        return None

    def _select(self, route_s: float) -> Optional[ResolvedMission]:
        for rm in self.missions:
            if rm.mission_id in self.done:
                continue
            if route_s < rm.start_s:
                return None  # next mission not reached yet; earlier ones are done
            if route_s >= rm.end_s and rm.mission_id not in self.pause_brake_since:
                self.done.add(rm.mission_id)
                self._event(f"{rm.mission_id} DONE at s={route_s:.1f}")
                continue
            return rm
        return None

    def step(self, inp: MissionInputs) -> MissionOutput:
        rm = self._select(inp.route_s)
        if rm is None and inp.signal_brake_m > 0.0:
            nxt = self._next_pending()
            if nxt is not None and nxt.kind == SIGNAL and nxt.mission_id not in self.committed:
                d = nxt.stop_anchor_s - inp.front_offset_m - self.stop_buffer - inp.route_s
                if d <= inp.signal_brake_m:
                    rm = nxt  # early activation: the stop line is within braking distance
        if rm is None:
            self.state = CRUISE_GLOBAL
            self.active_id = None
            return MissionOutput(CRUISE_GLOBAL, None, None, None, self.cruise, None, False, None, "no active mission", False)

        m = rm.mission
        if self.active_id != rm.mission_id:
            self.active_id = rm.mission_id
            self._event(f"{rm.mission_id} ACTIVE ({m.kind}/{m.direction}) at s={inp.route_s:.1f}")

        if m.kind == TURN:
            self.state = TURN_GLOBAL
            cap = self.turn if self.turn is not None else self.cruise
            return MissionOutput(TURN_GLOBAL, m.mission_id, m.kind, m.direction, cap, None, False, None, "global path through the turn", False, end_s=rm.end_s)
        if m.kind == OBSTACLE:
            self.state = OBSTACLE_RESERVED
            return MissionOutput(OBSTACLE_RESERVED, m.mission_id, m.kind, None, self.obstacle, None, False, None, "obstacle zone (avoidance algorithm pending)", False, end_s=rm.end_s)
        if m.kind == PAUSE:
            return self._pause(rm, inp)

        # ---- signal mission -------------------------------------------------
        mid = m.mission_id
        permit = inp.permit(m.direction, m.vehicle_signal_roi, m.require_vehicle_signal_roi)
        stop_control_s = rm.stop_anchor_s - inp.front_offset_m - self.stop_buffer
        d_stop = stop_control_s - inp.route_s
        crossed = inp.route_s + inp.front_offset_m >= rm.stop_anchor_s
        commit_s = stop_control_s - self.commit_distance
        at_commit_point = crossed or (
            self.commit_mode == "last_stoppable_point" and d_stop <= self.commit_distance
        )
        limit = m.timeout_departure_s if m.timeout_departure_s is not None else self.timeout_departure_s
        if not m.allow_timeout_departure:
            limit = None

        # A visible red light revokes a timeout permit that has not been committed.
        if mid in self.timeout_permits and mid not in self.committed and inp.sees_red:
            self.timeout_permits.discard(mid)
            self.timeout_since.pop(mid, None)
            self._event(f"{mid} timeout permit REVOKED: red light visible at s={inp.route_s:.1f}")
        timeout_ok = mid in self.timeout_permits
        effective = permit.valid or timeout_ok
        source = "detector" if permit.valid else ("timeout" if timeout_ok else "")
        waited = None
        if mid in self.timeout_since:
            waited = (inp.now_ns - self.timeout_since[mid]) * 1e-9

        def out(state, cap, stop_s, hold, reason, approach=None, extra_commit=None):
            return MissionOutput(state, mid, m.kind, m.direction, cap, stop_s, hold, permit, reason, True,
                                 d_stop, rm.end_s, stop_control_s, rm.stop_anchor_s, approach, extra_commit,
                                 source, waited, limit)

        if mid in self.committed:
            self.state = COMMITTED_CROSSING
            return out(COMMITTED_CROSSING, self.approach, None, False, "committed: signal no longer a stop condition")

        if effective and at_commit_point:
            self.committed.add(mid)
            self.timeout_since.pop(mid, None)
            self.state = COMMITTED_CROSSING
            self._event(f"{mid} COMMITTED ({source}) at s={inp.route_s:.1f}, d_stop={d_stop:.2f}")
            return out(COMMITTED_CROSSING, self.approach, None, False,
                       f"commit point reached with a valid {source} permit")

        if crossed:
            # Past the anchor without permission: stop where we are and wait.
            self.state = WAIT_SIGNAL
            self.waited.add(mid)
            return out(WAIT_SIGNAL, 0.0, inp.route_s, True, f"STOPLINE_OVERRUN_NO_PERMIT ({permit.reason})")

        if effective:
            self.timeout_since.pop(mid, None)
            if inp.route_s >= rm.decision_s or mid in self.waited:
                self.state = RELEASE_TO_LINE
                reason = f"{source} permit; speed kept stoppable until commit point"
            else:
                self.state = SIGNAL_DETECT
                reason = f"{source} permit before decision point; speed kept stoppable"
            return out(self.state, self.approach, None, False, reason, stop_control_s, commit_s)

        # No permit: the stop target is enforced; the braking model in the node
        # decides how fast the approach may still be.
        at_line = d_stop <= self.hold_distance
        braking_here = inp.brake_latched_for == mid
        if inp.stopped and (at_line or braking_here):
            self.state = WAIT_SIGNAL
            self.waited.add(mid)
            hold = True
        elif inp.route_s >= rm.decision_s or at_line:
            self.state = BRAKE_TO_STOP
            hold = at_line
        else:
            self.state = SIGNAL_DETECT
            hold = False

        # Timeout departure: count only while stopped at the line with NO red
        # light reported (and, by default, only while the detector is alive).
        if self.state == WAIT_SIGNAL and limit is not None:
            counting = (not inp.sees_red) and (inp.signal_alive or not self.timeout_requires_alive)
            if counting:
                since = self.timeout_since.setdefault(mid, inp.now_ns)
                waited = (inp.now_ns - since) * 1e-9
                if waited >= limit:
                    self.timeout_permits.add(mid)
                    self.state = RELEASE_TO_LINE
                    self._event(f"{mid} TIMEOUT DEPARTURE after {waited:.1f}s without a detected signal")
                    return MissionOutput(RELEASE_TO_LINE, mid, m.kind, m.direction, self.approach, None, False,
                                         permit, f"timeout departure ({waited:.0f}s >= {limit:.0f}s, no signal detected)",
                                         True, d_stop, rm.end_s, stop_control_s, rm.stop_anchor_s,
                                         stop_control_s, commit_s, "timeout", waited, limit)
            else:
                self.timeout_since.pop(mid, None)
                waited = None
        else:
            self.timeout_since.pop(mid, None)
            waited = None
        why = "red light" if inp.sees_red else permit.reason
        if not inp.signal_alive:
            why += " (detector offline)"
        return out(self.state, self.approach, stop_control_s, hold, f"no permit: {why}")

    def _finish_pause(self, mid: str) -> None:
        self.paused.add(mid)
        self.pause_since.pop(mid, None)
        self.pause_brake_since.pop(mid, None)

    def _pause(self, rm: ResolvedMission, inp: MissionInputs) -> MissionOutput:
        """Stop before the turn the way the avoidance zone brake stops (zone_brake.py).

        No slow-down before it: the vehicle keeps its speed until the stop target
        (front bumper stop_gap_m before stop_idx) is within its stopping distance,
        then brakes to a standstill, holds it for hold_s and goes - once.
        """

        m = rm.mission
        mid = m.mission_id
        hold_s = m.hold_s if m.hold_s is not None else self.pause_hold
        stop_control_s = rm.stop_anchor_s - inp.front_offset_m - self.pause_gap
        d_stop = stop_control_s - inp.route_s
        still = inp.stopped if inp.standing_still is None else inp.standing_still

        def out(state, hold, reason, waited=None, target=True):
            # No speed cap of its own and no stop target for the speed planner
            # (that would glide in): the brake alone stops the vehicle.
            return MissionOutput(state, mid, m.kind, m.direction, self.cruise, None, hold, None, reason, False,
                                 d_stop, rm.end_s, stop_control_s if target else None, rm.stop_anchor_s,
                                 timeout_wait_s=waited, timeout_limit_s=hold_s)

        if mid in self.paused:
            self.state = PAUSE_GO
            return out(PAUSE_GO, False, f"paused {hold_s:.1f}s, going", target=False)

        if mid not in self.pause_brake_since:
            # stop_needed (zone_brake.py): brake once the target is no farther than
            # the stopping distance, and always once it has been reached.
            if d_stop > max(inp.stop_distance_m, 0.0):
                self.state = PAUSE_APPROACH
                return out(PAUSE_APPROACH, False, f"pause ahead: d={d_stop:.1f} m, brake at {inp.stop_distance_m:.1f} m")
            self.pause_brake_since[mid] = inp.now_ns
            self._event(f"{mid} PAUSE BRAKE at s={inp.route_s:.1f}, v={inp.speed_mps * 3.6:.1f} km/h, d_stop={d_stop:.2f}")

        if still:
            since = self.pause_since.setdefault(mid, inp.now_ns)
            waited = (inp.now_ns - since) * 1e-9
            if waited >= hold_s:
                self._finish_pause(mid)
                self.state = PAUSE_GO
                self._event(f"{mid} PAUSE served ({waited:.1f}s) at s={inp.route_s:.1f}, d_stop={d_stop:.2f}")
                return out(PAUSE_GO, False, f"paused {waited:.1f}s, going", waited, target=False)
            self.state = PAUSE_WAIT
            return out(PAUSE_WAIT, True, f"pause: stopped {waited:.1f}/{hold_s:.1f}s", waited)

        self.pause_since.pop(mid, None)
        if inp.now_ns - self.pause_brake_since[mid] >= self.pause_max_brake_ns:
            # As zone_brake max_brake_s: a brake that never sees a standstill
            # (no speed input, stuck vehicle) ends instead of holding forever.
            self._finish_pause(mid)
            self.pause_timeouts += 1
            self.state = PAUSE_GO
            self._event(f"{mid} PAUSE brake released by max_brake_s without a standstill")
            return out(PAUSE_GO, False, "PAUSE_BRAKE_TIMEOUT: no standstill seen, going", target=False)
        self.state = PAUSE_BRAKE
        return out(PAUSE_BRAKE, True, f"pause brake to a stop (d={d_stop:.1f} m)")
