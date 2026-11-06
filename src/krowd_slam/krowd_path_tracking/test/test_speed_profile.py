from dataclasses import replace
import math

import pytest

from krowd_path_tracking.route_geometry import path_curvatures, segment_lengths
from krowd_path_tracking.speed_profile import (
    SpeedLimits,
    curvature_speed_ceiling,
    speed_profile,
)


LIMITS = SpeedLimits(
    max_speed_mps=9.0,
    max_lateral_accel_mps2=3.0,
    max_accel_mps2=1.5,
    max_decel_mps2=2.0,
    min_speed_mps=1.5,
)


KMH = 1.0 / 3.6

# Straights 10-15 km/h, curves 5-10 km/h: config/path_tracking.yaml's values.
CURVE_LIMITS = SpeedLimits(
    max_speed_mps=15.0 * KMH,
    max_lateral_accel_mps2=0.5,
    max_accel_mps2=1.5,
    max_decel_mps2=2.0,
    min_speed_mps=5.0 * KMH,
    curve_max_speed_mps=10.0 * KMH,
    curve_radius_m=30.0,
)


def assert_longitudinally_feasible(speeds, lengths, limits):
    """Every step must be reachable by accelerating or braking within it."""

    for index, length in enumerate(lengths):
        entry, exit_ = speeds[index], speeds[index + 1]
        if exit_ > entry:
            assert exit_**2 - entry**2 <= 2.0 * limits.max_accel_mps2 * length + 1e-6
        else:
            assert entry**2 - exit_**2 <= 2.0 * limits.max_decel_mps2 * length + 1e-6


def test_a_straight_route_runs_at_the_speed_ceiling(straight_route):
    route = straight_route(50)

    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), LIMITS, stop_at_end=False
    )

    assert speeds[-1] == pytest.approx(LIMITS.max_speed_mps)
    assert max(speeds) == pytest.approx(LIMITS.max_speed_mps)


def test_a_corner_is_limited_by_lateral_acceleration(circular_route):
    radius = 15.0
    route = circular_route(radius, 80)

    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), LIMITS, stop_at_end=False
    )

    expected = math.sqrt(LIMITS.max_lateral_accel_mps2 * radius)
    assert expected < LIMITS.max_speed_mps
    # Well inside the arc, away from the entry ramp and the endpoint copies.
    assert speeds[40] == pytest.approx(expected, rel=0.02)


def test_braking_for_a_corner_starts_on_the_straight_before_it(
    straight_route, circular_route
):
    straight = straight_route(60)
    arc = circular_route(12.0, 60)
    offset_x, offset_y = straight[-1]
    route = straight + [(offset_x + x, offset_y + y) for x, y in arc[1:]]

    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), LIMITS, stop_at_end=False
    )

    # The last waypoint of the straight is already below the ceiling: the
    # backward pass has reached back out of the corner.
    assert speeds[58] < LIMITS.max_speed_mps
    assert speeds[10] == pytest.approx(LIMITS.max_speed_mps)


def test_stop_at_end_reaches_zero_and_ramps_down_to_it(straight_route):
    route = straight_route(60)
    lengths = segment_lengths(route)

    speeds = speed_profile(lengths, path_curvatures(route), LIMITS, stop_at_end=True)

    assert speeds[-1] == pytest.approx(0.0)
    assert speeds[-2] > 0.0
    assert_longitudinally_feasible(speeds, lengths, LIMITS)


def test_stop_at_end_overrides_the_minimum_speed_floor(straight_route):
    route = straight_route(30)
    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), LIMITS, stop_at_end=True
    )
    assert speeds[-1] < LIMITS.min_speed_mps


def test_profile_is_feasible_on_a_mixed_route(straight_route, circular_route):
    route = straight_route(40) + [
        (40.0 + x, y) for x, y in circular_route(10.0, 40)[1:]
    ]
    lengths = segment_lengths(route)

    speeds = speed_profile(lengths, path_curvatures(route), LIMITS, stop_at_end=True)

    assert_longitudinally_feasible(speeds, lengths, LIMITS)
    assert all(speed >= 0.0 for speed in speeds)


def test_curvature_ceiling_floors_at_the_minimum_speed():
    # A 0.5 m radius corner profiles to 1.22 m/s on grip alone; the floor keeps
    # noisy curvature from taking the vehicle lower still.
    ceiling = curvature_speed_ceiling([1.0 / 0.5], LIMITS)
    speeds = speed_profile([], [1.0 / 0.5], LIMITS, stop_at_end=False)
    assert ceiling[0] < LIMITS.min_speed_mps
    assert speeds[0] == pytest.approx(LIMITS.min_speed_mps)


def test_a_straight_has_no_curvature_ceiling():
    assert curvature_speed_ceiling([0.0], LIMITS) == pytest.approx([9.0])


def test_empty_route_profiles_to_nothing():
    assert speed_profile([], [], LIMITS, stop_at_end=True) == []


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        speed_profile([1.0], [0.0, 0.0, 0.0], LIMITS, stop_at_end=False)


@pytest.mark.parametrize(
    "field, value",
    [
        ("max_speed_mps", 0.0),
        ("max_lateral_accel_mps2", 0.0),
        ("max_accel_mps2", -1.0),
        ("max_decel_mps2", 0.0),
        ("min_speed_mps", 20.0),
    ],
)
def test_limits_validate_rejects_unusable_values(field, value):
    fields = {
        "max_speed_mps": 9.0,
        "max_lateral_accel_mps2": 3.0,
        "max_accel_mps2": 1.5,
        "max_decel_mps2": 2.0,
        "min_speed_mps": 1.5,
    }
    fields[field] = value
    with pytest.raises(ValueError):
        SpeedLimits(**fields).validate()


def test_curve_cap_is_off_unless_configured():
    assert not LIMITS.curve_cap_enabled
    assert CURVE_LIMITS.curve_cap_enabled


def test_straights_run_at_the_maximum_and_curves_at_the_curve_cap(
    straight_route, circular_route
):
    straight = straight_route(60)
    arc = circular_route(20.0, 60)
    offset_x, offset_y = straight[-1]
    route = straight + [(offset_x + x, offset_y + y) for x, y in arc[1:]]
    lengths = segment_lengths(route)

    speeds = speed_profile(lengths, path_curvatures(route), CURVE_LIMITS, stop_at_end=False)

    assert speeds[10] == pytest.approx(15.0 * KMH)
    # Grip alone would allow sqrt(0.5 * 20) = 11.4 km/h; a 20 m radius is a
    # curve (< 30 m), so the cap holds it to 10.
    assert math.sqrt(0.5 * 20.0) > 10.0 * KMH
    assert speeds[90] == pytest.approx(10.0 * KMH, rel=0.02)
    assert_longitudinally_feasible(speeds, lengths, CURVE_LIMITS)


def test_a_tighter_curve_lands_between_the_floor_and_the_cap(circular_route):
    route = circular_route(8.0, 40)
    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), CURVE_LIMITS, stop_at_end=False
    )
    # sqrt(0.5 * 8) = 2.0 m/s = 7.2 km/h
    assert speeds[20] == pytest.approx(2.0, rel=0.03)
    assert 5.0 * KMH < speeds[20] < 10.0 * KMH


def test_a_very_tight_curve_is_held_at_the_floor(circular_route):
    route = circular_route(3.0, 12)
    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), CURVE_LIMITS, stop_at_end=False
    )
    # sqrt(0.5 * 3) = 4.4 km/h, under the 5 km/h floor.
    assert speeds[6] == pytest.approx(5.0 * KMH)


def test_a_bend_gentler_than_the_curve_radius_is_not_capped(circular_route):
    route = circular_route(40.0, 40)
    speeds = speed_profile(
        segment_lengths(route), path_curvatures(route), CURVE_LIMITS, stop_at_end=False
    )
    assert speeds[20] == pytest.approx(15.0 * KMH)


@pytest.mark.parametrize(
    "changes",
    [
        {"curve_max_speed_mps": 4.0 * KMH},  # below the 5 km/h floor
        {"curve_radius_m": 0.0},  # only one of the pair set
        {"curve_max_speed_mps": 0.0},  # only one of the pair set
        {"curve_max_speed_mps": -1.0},
    ],
)
def test_curve_cap_validation_rejects_inconsistent_values(changes):
    with pytest.raises(ValueError):
        replace(CURVE_LIMITS, **changes).validate()


def test_curve_lead_reaches_corner_speed_before_the_corner(
    straight_route, circular_route
):
    straight = straight_route(60)
    arc = circular_route(12.0, 60)
    offset_x, offset_y = straight[-1]
    route = straight + [(offset_x + x, offset_y + y) for x, y in arc[1:]]
    lengths = segment_lengths(route)
    curvatures = path_curvatures(route)

    plain = speed_profile(lengths, curvatures, LIMITS, stop_at_end=False)
    led = speed_profile(
        lengths, curvatures, replace(LIMITS, curve_lead_m=5.0), stop_at_end=False
    )

    corner_speed = min(plain)
    # Where the plain profile first reaches corner speed, the led profile
    # (1 m spacing) already held it 5 m earlier, while the plain one was still
    # braking towards it there.
    at_corner_speed = next(
        i for i, v in enumerate(plain) if v == pytest.approx(corner_speed, rel=1e-6)
    )
    five_before = at_corner_speed - 5
    assert led[five_before] == pytest.approx(corner_speed, rel=1e-6)
    assert plain[five_before] > corner_speed * 1.05
    # Still feasible, and never faster than the plain profile anywhere.
    assert_longitudinally_feasible(led, lengths, LIMITS)
    assert all(a <= b + 1e-9 for a, b in zip(led, plain))
    assert speed_profile(lengths, curvatures, replace(LIMITS, curve_lead_m=0.0), stop_at_end=False) == plain


def test_curve_exit_lead_leaves_the_corner_at_straight_speed(
    straight_route, circular_route
):
    straight = straight_route(40)
    arc = circular_route(12.0, 40)
    ox, oy = straight[-1]
    route = straight + [(ox + x, oy + y) for x, y in arc[1:]]
    # ...and a straight after the corner, continuing the arc's end tangent.
    end_x, end_y = route[-1]
    px, py = route[-2]
    tx, ty = end_x - px, end_y - py
    route += [(end_x + i * tx, end_y + i * ty) for i in range(1, 40)]
    lengths = segment_lengths(route)
    curvatures = path_curvatures(route)

    plain = speed_profile(lengths, curvatures, LIMITS, stop_at_end=False)
    released = speed_profile(
        lengths, curvatures, replace(LIMITS, curve_exit_lead_m=16.0), stop_at_end=False
    )

    corner_speed = min(plain)
    # Where the plain profile last holds corner speed (the corner's end, give
    # or take the curvature smoothing), the released profile is already back
    # near straight speed (6 -> 9 m/s at 1.5 m/s2 needs 15 m of the 16 m
    # lead); before the tightest point both are the same.
    last_at_corner_speed = max(
        i for i, v in enumerate(plain) if v == pytest.approx(corner_speed, rel=1e-6)
    )
    assert released[last_at_corner_speed] > 0.9 * LIMITS.max_speed_mps
    tightest = min(range(len(plain)), key=lambda i: plain[i])
    assert released[: tightest + 1] == pytest.approx(plain[: tightest + 1])
    assert_longitudinally_feasible(released, lengths, LIMITS)


def test_exit_release_starts_after_the_geometric_apex_even_with_a_flat_ceiling(
    straight_route, circular_route
):
    """min_speed == curve cap: every curve waypoint has the same ceiling."""

    straight = straight_route(40)
    arc = circular_route(8.0, 12)  # short and tight
    ox, oy = straight[-1]
    route = straight + [(ox + x, oy + y) for x, y in arc[1:]]
    end_x, end_y = route[-1]
    px, py = route[-2]
    route += [(end_x + i * (end_x - px), end_y + i * (end_y - py)) for i in range(1, 40)]
    lengths = segment_lengths(route)
    curvatures = path_curvatures(route)
    flat = replace(
        CURVE_LIMITS,
        min_speed_mps=10.0 * KMH,
        curve_max_speed_mps=10.0 * KMH,
        max_speed_mps=20.0 * KMH,
        max_decel_mps2=0.5,
        curve_exit_lead_m=8.0,
    )

    speeds = speed_profile(lengths, curvatures, flat, stop_at_end=False)

    apex = max(range(len(curvatures)), key=lambda i: abs(curvatures[i]))
    assert speeds[apex] == pytest.approx(10.0 * KMH)
    curve = [i for i, k in enumerate(curvatures) if abs(k) > 1.0 / 30.0]
    # Nothing in the curve before the apex speeds up.
    assert all(speeds[i] == pytest.approx(10.0 * KMH) for i in curve if i <= apex)
