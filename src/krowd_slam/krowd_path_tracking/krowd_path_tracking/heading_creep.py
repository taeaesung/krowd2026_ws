"""Drive straight, slowly, until a vehicle pose exists.

gnss_imu_pose_node commits its heading only after ~5 m of straight driving, so
on a plain RTK start there is no vehicle state to steer against until the
vehicle has moved - and nothing moves it. This is the automatic form of the
manual first drive: while a route is loaded and no vehicle state has EVER
arrived, command straight steering at a creep speed. It ends the moment the
first state arrives (tracking takes over, through the start ramp) or once
max_duration_s has passed without one (the node then brakes; restart it to try
again). Losing the state later is STALE_VEHICLE_STATE and brakes as before.

The creep is the LAST way to a heading, not the first. gnss_imu_pose_node
looks for one in order - a heading the sensor-stack ESKF already verified,
then the verified LiDAR map alignment - and reports where it stands on
/gnss_imu/heading_acquisition. Only its NEEDS_MOTION starts a creep; while it
is still waiting on the ESKF or LiDAR the vehicle stays put, so a standstill
start is never pre-empted by a blind drive. creep_permitted() is that gate.
Plain numbers, no ROS, like start_ramp.py.
"""

NEEDS_MOTION = "NEEDS_MOTION"


def creep_permitted(heading_state: str | None, require_request: bool) -> bool:
    """Whether localization has asked for motion to acquire the heading.

    require_request False restores the old behaviour: creep whenever a route
    exists and no vehicle state does.
    """

    return not require_request or heading_state == NEEDS_MOTION


class HeadingCreep:
    CREEP = "CREEP"
    TIMEOUT = "TIMEOUT"
    OFF = "OFF"

    def __init__(self, speed_mps: float, max_duration_s: float):
        if speed_mps < 0.0 or max_duration_s < 0.0:
            raise ValueError(
                f"heading creep needs non-negative values: {speed_mps}, {max_duration_s}"
            )
        self.speed_mps = speed_mps
        self.max_duration_s = max_duration_s
        self._started_ns: int | None = None
        self._done = False

    @property
    def enabled(self) -> bool:
        return self.speed_mps > 0.0 and self.max_duration_s > 0.0

    def vehicle_state_seen(self) -> None:
        """A pose exists now; never creep again in this node's lifetime."""

        self._done = True

    @property
    def started(self) -> bool:
        return self._started_ns is not None

    def elapsed_s(self, now_ns: int) -> float:
        if self._started_ns is None:
            return 0.0
        return (now_ns - self._started_ns) * 1e-9

    def command(self, now_ns: int) -> str:
        """What to do this cycle while no vehicle state has arrived yet."""

        if not self.enabled or self._done:
            return self.OFF
        if self._started_ns is None:
            self._started_ns = now_ns
        if self.elapsed_s(now_ns) > self.max_duration_s:
            return self.TIMEOUT
        return self.CREEP
