"""ZoneEntryBrake (2026-10-10): on entering an avoidance zone brake to a standstill, hold it stop_hold_s."""

from krowd_pp_stanleyff_control.zone_brake import ZoneEntryBrake

MS = 1_000_000


def run(zb, samples, still_from_ms=None, step_ms=20, until_ms=None):
    """samples: (t_ms, inside) messages. The vehicle stands still from still_from_ms on
    (None = never). Returns [(t_ms, braking)] every step."""
    msgs = sorted(samples); out = []; k = 0
    end = until_ms if until_ms is not None else msgs[-1][0] + 1000
    for t in range(0, end + 1, step_ms):
        while k < len(msgs) and msgs[k][0] <= t:
            zb.observe(msgs[k][1], msgs[k][0] * MS); k += 1
        still = still_from_ms is not None and t >= still_from_ms
        out.append((t, zb.update(t * MS, still)))
    return out


def braking_ms(trace):
    on = [t for t, b in trace if b]
    return (on[0], on[-1]) if on else None


def test_brakes_until_stopped_then_holds_one_second():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    msgs = [(t, t >= 1000) for t in range(0, 6000, 100)]
    # entered at 1.0 s, standing still from 2.6 s -> released at 3.6 s
    assert braking_ms(run(zb, msgs, still_from_ms=2600)) == (1000, 3580)
    assert zb.triggers == 1 and zb.timed_out == 0


def test_a_standstill_must_last_the_whole_hold():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    zb.observe(True, 0)
    assert zb.update(0, False)
    assert zb.update(500 * MS, True)          # still from 0.5 s
    assert zb.update(1000 * MS, False)        # creeps: hold restarts
    assert zb.update(1200 * MS, True)
    assert zb.update(2100 * MS, True)
    assert not zb.update(2200 * MS, True)     # 1.0 s still since 1.2 s


def test_no_standstill_is_released_by_max_brake():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    msgs = [(t, True) for t in range(0, 8000, 100)]
    assert braking_ms(run(zb, msgs)) == (0, 4980)
    assert zb.timed_out == 1 and zb.triggers == 1


def test_no_second_brake_while_staying_inside():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    msgs = [(t, True) for t in range(0, 6000, 100)]
    trace = run(zb, msgs, still_from_ms=500)
    assert braking_ms(trace) == (0, 1480) and zb.triggers == 1


def test_boundary_flicker_does_not_brake_again():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    msgs = [(t, (1000 <= t < 1200) or t >= 1600) for t in range(0, 4000, 100)]
    run(zb, msgs, still_from_ms=1500)
    assert zb.triggers == 1


def test_next_zone_brakes_again_after_being_outside_long_enough():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    msgs = [(t, (1000 <= t < 3000) or t >= 6000) for t in range(0, 9000, 100)]
    trace = run(zb, msgs, still_from_ms=None)
    assert zb.triggers == 2
    assert any(b for t, b in trace if 6000 <= t < 6500)


def test_stale_true_is_outside():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    zb.observe(True, 0)
    assert zb.update(0, False) is True
    zb2 = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    zb2.observe(True, 0)
    assert zb2.update(1500 * MS, False) is False
    assert zb2.triggers == 0


def test_reset_rearms_and_clears():
    zb = ZoneEntryBrake(1.0, 2.0, 1.0, 5.0)
    zb.observe(True, 0); assert zb.update(0, False)
    zb.reset(); assert zb.update(10 * MS, False) is True and zb.triggers == 2


def test_zone_file_ranges_and_lead_regions(tmp_path):
    from krowd_pp_stanleyff_control.zone_brake import in_brake_region, in_lead_region, load_zone_ranges
    f = tmp_path / "avoidance_zones.yaml"
    f.write_text('localization_monitor_node:\n  ros__parameters:\n    avoidance_zones: ["121-135", "150-152"]  # c\n')
    ranges = load_zone_ranges(str(f))
    assert ranges == [(121, 135), (150, 152)]
    assert not in_brake_region(ranges, 110.9, 10) and in_brake_region(ranges, 111.0, 10)
    assert in_brake_region(ranges, 135.0, 10) and not in_brake_region(ranges, 135.1, 10)
    assert in_lead_region(ranges, 115.0, 10) and not in_lead_region(ranges, 121.0, 10)
    f.write_text('localization_monitor_node:\n  ros__parameters:\n    avoidance_zones: []\n')
    assert load_zone_ranges(str(f)) == []
    f.write_text('localization_monitor_node:\n  ros__parameters:\n    avoidance_zones: ["135-121"]\n')
    import pytest
    with pytest.raises(ValueError):
        load_zone_ranges(str(f))


def test_no_trigger_on_the_splice_and_no_rearm():
    zb = ZoneEntryBrake(0.5, 2.0, 1.0, 5.0)
    zb.observe(True, 0)
    assert zb.update(0, False, can_trigger=False) is False and zb.triggers == 0


def test_brake_starts_within_the_stopping_distance_of_the_zone_start():
    from krowd_pp_stanleyff_control.braking import BrakingModel
    from krowd_pp_stanleyff_control.zone_brake import stop_needed, zone_at
    ranges = [(121, 135)]
    assert zone_at(ranges, 112.9, 8) is None and zone_at(ranges, 113.0, 8) == (121, 135)
    assert zone_at(ranges, 135.0, 8) == (121, 135) and zone_at(ranges, 135.5, 8) is None
    brake = BrakingModel(3.95, 1.0, 0.0, ((5.0, 1.5), (10.0, 1.9), (15.0, 2.2), (20.0, 4.17), (30.0, 7.3)))
    # slow (5 km/h): drive on until 1.5 m before the zone start
    assert not stop_needed(3.0, brake.envelope_distance(5 / 3.6))
    assert stop_needed(1.5, brake.envelope_distance(5 / 3.6))
    # fast (25 km/h): brake at once from 7 m out (stops inside the zone - accepted)
    assert stop_needed(7.0, brake.envelope_distance(25 / 3.6))
    # already in the zone: always
    assert stop_needed(-0.3, 0.0)
