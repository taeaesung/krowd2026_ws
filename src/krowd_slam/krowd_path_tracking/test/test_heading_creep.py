import pytest

from krowd_path_tracking.heading_creep import HeadingCreep


S = 1_000_000_000


def test_creeps_until_the_time_limit_then_times_out():
    creep = HeadingCreep(speed_mps=1.39, max_duration_s=12.0)
    assert creep.command(100 * S) == HeadingCreep.CREEP
    assert creep.command(111 * S) == HeadingCreep.CREEP
    assert creep.elapsed_s(111 * S) == pytest.approx(11.0)
    assert creep.command(113 * S) == HeadingCreep.TIMEOUT
    # Stays timed out: the node brakes and a restart is the way to retry.
    assert creep.command(114 * S) == HeadingCreep.TIMEOUT


def test_the_first_vehicle_state_ends_it_for_good():
    creep = HeadingCreep(speed_mps=1.39, max_duration_s=12.0)
    assert creep.command(100 * S) == HeadingCreep.CREEP
    creep.vehicle_state_seen()
    # A later loss of the state is STALE_VEHICLE_STATE territory, not a creep.
    assert creep.command(101 * S) == HeadingCreep.OFF


def test_timing_starts_at_the_first_cycle_that_asks_not_at_construction():
    creep = HeadingCreep(speed_mps=1.39, max_duration_s=5.0)
    assert creep.elapsed_s(50 * S) == 0.0
    assert creep.command(50 * S) == HeadingCreep.CREEP
    assert creep.command(54 * S) == HeadingCreep.CREEP
    assert creep.command(56 * S) == HeadingCreep.TIMEOUT


def test_zero_disables_it():
    assert HeadingCreep(0.0, 12.0).command(0) == HeadingCreep.OFF
    assert HeadingCreep(1.0, 0.0).command(0) == HeadingCreep.OFF
    assert not HeadingCreep(0.0, 12.0).enabled


def test_negative_values_are_rejected():
    for speed, duration in ((-1.0, 12.0), (1.0, -1.0)):
        with pytest.raises(ValueError):
            HeadingCreep(speed, duration)


def test_creep_waits_for_localization_to_ask_for_motion():
    from krowd_path_tracking.heading_creep import NEEDS_MOTION, creep_permitted

    # ESKF or LiDAR may still give the heading at a standstill: stay put.
    for state in (None, "WAITING_ESKF", "WAITING_LIDAR", "ACQUIRED ESKF (rtk_course)"):
        assert not creep_permitted(state, require_request=True)
    assert creep_permitted(NEEDS_MOTION, require_request=True)
    # The old behaviour on request.
    assert creep_permitted(None, require_request=False)


def test_started_reports_a_creep_in_progress():
    creep = HeadingCreep(speed_mps=1.39, max_duration_s=12.0)
    assert not creep.started
    creep.command(5 * S)
    assert creep.started
