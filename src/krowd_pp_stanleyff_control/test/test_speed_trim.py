import pytest

from krowd_pp_stanleyff_control.speed_trim import SpeedTrim

KMH = 1.0 / 3.6


def _trim():
    return SpeedTrim(gain_per_s=0.3, max_trim_mps=5.0 * KMH, min_speed_mps=1.0 * KMH)


def test_a_vehicle_at_its_target_gets_the_target():
    trim = _trim()
    for _ in range(200):
        assert trim.command(5.0 * KMH, 5.0 * KMH, 0.05) == pytest.approx(5.0 * KMH)


def test_a_slow_vehicle_gets_a_growing_but_bounded_command():
    trim = _trim()
    first = trim.command(5.0 * KMH, 3.8 * KMH, 0.05)
    for _ in range(100):
        later = trim.command(5.0 * KMH, 3.8 * KMH, 0.05)
    assert 5.0 * KMH < first < later
    for _ in range(100000):
        last = trim.command(5.0 * KMH, 3.8 * KMH, 0.05)
    assert last == pytest.approx(10.0 * KMH)


def test_waiting_at_a_standstill_does_not_wind_up():
    trim = _trim()
    for _ in range(2000):  # 100 s waiting for the ASMS
        command = trim.command(5.0 * KMH, 0.0, 0.05)
    assert command == pytest.approx(5.0 * KMH) and trim.trim_mps == 0.0


def test_overspeed_takes_the_trim_back_but_never_below_the_target():
    trim = _trim()
    for _ in range(400):
        trim.command(5.0 * KMH, 3.0 * KMH, 0.05)
    assert trim.trim_mps > 0.0
    for _ in range(4000):
        command = trim.command(5.0 * KMH, 8.0 * KMH, 0.05)
    assert command == pytest.approx(5.0 * KMH)


def test_a_stop_freezes_the_trim_and_only_reset_clears_it():
    """A traffic-light stop must not throw away what the trim learnt (2026-09-25)."""

    trim = _trim()
    for _ in range(400):
        trim.command(5.0 * KMH, 3.0 * KMH, 0.05)
    learnt = trim.trim_mps
    assert learnt > 0.0
    assert trim.command(0.0, 3.0 * KMH, 0.05) == 0.0 and trim.trim_mps == learnt
    # Straight after the stop the learnt trim is back on the target.
    assert trim.command(5.0 * KMH, 0.5 * KMH, 0.05) == pytest.approx(5.0 * KMH + learnt)
    trim.reset()
    assert trim.trim_mps == 0.0


def test_inactive_freezes_and_returns_the_bare_target():
    """While a stopping-distance cap or the start ramp owns the target the trim
    neither applies nor learns: a trim on top of a stoppable cap defeats it."""

    trim = _trim()
    for _ in range(400):
        trim.command(5.0 * KMH, 3.0 * KMH, 0.05)
    learnt = trim.trim_mps
    assert trim.command(4.0 * KMH, 3.0 * KMH, 0.05, active=False) == pytest.approx(4.0 * KMH)
    assert trim.trim_mps == learnt


def test_disabled_passes_the_target_through():
    trim = SpeedTrim(0.0, 0.0, 0.0)
    assert not trim.enabled
    assert trim.command(5.0 * KMH, 1.0 * KMH, 0.05) == pytest.approx(5.0 * KMH)


def test_negative_settings_are_refused():
    with pytest.raises(ValueError):
        SpeedTrim(-0.1, 1.0, 0.0)
