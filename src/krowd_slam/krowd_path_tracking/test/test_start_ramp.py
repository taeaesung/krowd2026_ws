import math

import pytest

from krowd_path_tracking.start_ramp import StartRamp


def test_starts_at_the_start_speed_and_accelerates_with_distance():
    ramp = StartRamp(start_speed_mps=1.0, accel_mps2=2.0)
    assert ramp.cap(0.0, 0.0, 0.0) == pytest.approx(1.0)
    assert ramp.cap(0.0, 3.0, 4.0) == pytest.approx(math.sqrt(1.0 + 2.0 * 2.0 * 5.0))
    assert ramp.cap(0.0, 3.0, 4.0) == pytest.approx(math.sqrt(21.0))  # no move


def test_a_resume_at_speed_starts_from_that_speed_not_below_it():
    ramp = StartRamp(start_speed_mps=1.0, accel_mps2=2.0)
    assert ramp.cap(4.0, 0.0, 0.0) == pytest.approx(4.0)


def test_reset_starts_a_fresh_ramp():
    ramp = StartRamp(start_speed_mps=1.0, accel_mps2=2.0)
    ramp.cap(0.0, 0.0, 0.0)
    ramp.cap(0.0, 10.0, 0.0)
    ramp.reset()
    assert ramp.cap(0.0, 10.0, 0.0) == pytest.approx(1.0)


def test_zero_disables_the_cap():
    assert StartRamp(0.0, 1.0).cap(0.0, 0.0, 0.0) == math.inf
    assert StartRamp(1.0, 0.0).cap(0.0, 0.0, 0.0) == math.inf
    with pytest.raises(ValueError):
        StartRamp(-1.0, 1.0)
