from dataclasses import replace
import math

import pytest

from krowd_pp_stanleyff_control.route_geometry import path_curvatures, segment_lengths
from krowd_pp_stanleyff_control.speed_profile import (
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


def test_step_down_turns_braking_ramps_into_one_step():
    from krowd_pp_stanleyff_control.speed_profile import step_down_decelerations

    ramp = [7.0, 7.0, 6.0, 5.0, 4.0, 3.0, 3.0, 4.0, 5.0, 7.0, 6.0, 4.0, 2.0, 0.0]
    out = step_down_decelerations(ramp, 2.0)
    # corner ramp 7->3 becomes a step to 3 at the point where the ramp began;
    # the acceleration 3->7 stays
    assert out[:9] == [7.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 5.0]
    # the ramp into the final stop becomes the floor, only the last point is 0
    assert out[9:] == [2.0, 2.0, 2.0, 2.0, 0.0]


def test_curvature_only_profile_steps_down_and_ramps_up(straight_route, circular_route):
    """Straight -> R 12 m corner -> straight: no braking zone, the corner speed
    starts curve_lead_m before the corner as one step, and the way out climbs
    at max_accel_mps2."""

    from krowd_pp_stanleyff_control.speed_profile import curvature_only_profile

    limits = replace(LIMITS, curve_max_speed_mps=2.0, curve_radius_m=30.0, min_speed_mps=2.0,
                     curve_lead_m=3.0, curve_exit_lead_m=0.0)
    straight = straight_route(60)
    arc = circular_route(12.0, 20)
    ox, oy = straight[-1]
    route = straight + [(ox + x, oy + y) for x, y in arc[1:]]
    ex, ey = route[-1]
    dx, dy = ex - route[-2][0], ey - route[-2][1]
    norm = math.hypot(dx, dy)
    route += [(ex + k * dx / norm, ey + k * dy / norm) for k in range(1, 40)]
    lengths = segment_lengths(route)
    curvatures = path_curvatures(route)
    speeds = curvature_only_profile(lengths, curvatures, limits, stop_at_end=False)
    first_slow = next(i for i, v in enumerate(speeds) if v < limits.max_speed_mps)
    first_curve = next(i for i, k in enumerate(curvatures) if abs(k) > 1.0 / 30.0)
    # one step, about 3 m before the curvature rises; nothing in between
    assert speeds[first_slow] == pytest.approx(2.0)
    assert speeds[first_slow - 1] == pytest.approx(limits.max_speed_mps)
    assert sum(lengths[first_slow:first_curve]) == pytest.approx(3.0, abs=1.01)
    # the exit is a ramp at max_accel, never a step up
    rises = [(speeds[i + 1] ** 2 - speeds[i] ** 2) / (2 * lengths[i]) for i in range(len(lengths)) if speeds[i + 1] > speeds[i]]
    assert rises and max(rises) <= limits.max_accel_mps2 + 1e-6
    assert speeds[-1] == pytest.approx(limits.max_speed_mps)


def test_short_curve_runs_are_ignored():
    from krowd_pp_stanleyff_control.speed_profile import drop_short_curves

    ceil = [5.0, 5.0, 2.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 5.0]
    out = drop_short_curves(ceil, [1.0] * 9, 5.0, 3.0)
    assert out == [5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 5.0]


def test_route_takes_a_profile_drop_whole_at_the_first_waypoint():
    from krowd_pp_stanleyff_control.route import RouteModel

    xy = [(0.0, 0.0), (2.0, 0.0), (4.0, 0.0), (6.0, 0.0)]
    route = RouteModel.from_arrays(xy, [0.0] * 4, [0.0] * 4, [7.0, 3.0, 3.0, 7.0])
    assert route.speed_at_s(0.5) == pytest.approx(3.0)  # the drop, at once
    assert route.speed_at_s(5.0) == pytest.approx(5.0)  # the rise, interpolated


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


# Graded corner speed (2026-10-03): the high mode's numbers, grip out of the way.
GRADED = SpeedLimits(
    max_speed_mps=40.0 * KMH,
    max_lateral_accel_mps2=100.0,
    max_accel_mps2=3.0,
    max_decel_mps2=0.5,
    min_speed_mps=12.0 * KMH,
    curve_max_speed_mps=12.0 * KMH,
    curve_radius_m=30.0,
    tight_curve_radius_m=10.0,
)


@pytest.mark.parametrize(
    "radius, kmh",
    [(5.0, 12.0), (10.0, 12.0), (20.0, 26.0), (25.0, 33.0), (29.9, 39.86), (30.0, 40.0), (1e6, 40.0)],
)
def test_graded_corner_speed_is_linear_in_the_radius(radius, kmh):
    from krowd_pp_stanleyff_control.speed_profile import corner_speed

    assert corner_speed(1.0 / radius, GRADED) == pytest.approx(kmh * KMH, abs=0.01)


def test_tight_curve_radius_zero_keeps_the_flat_curve_cap():
    from krowd_pp_stanleyff_control.speed_profile import corner_speed

    flat = replace(GRADED, tight_curve_radius_m=0.0)
    assert corner_speed(1.0 / 20.0, flat) == pytest.approx(12.0 * KMH)
    assert corner_speed(1.0 / 31.0, flat) == pytest.approx(40.0 * KMH)


def test_each_corner_runs_at_its_slowest_point_s_speed():
    from krowd_pp_stanleyff_control.speed_profile import one_speed_per_corner

    assert one_speed_per_corner([5.0, 4.0, 3.0, 2.0, 3.0, 5.0, 4.5, 5.0], 5.0) == [5.0, 2.0, 2.0, 2.0, 2.0, 5.0, 4.5, 5.0]


def test_grip_slows_a_bend_too_short_for_the_noise_filter():
    """A 3 m bend of R 18 m under a 5 m noise filter: the curve cap is filtered
    away, the grip limit (applied after the filter) still holds it to
    sqrt(3.0 x 18) - the high mode used to take it at 40 km/h."""

    from krowd_pp_stanleyff_control.speed_profile import curvature_only_profile

    limits = replace(GRADED, max_lateral_accel_mps2=3.0)
    curvatures = [0.0] * 20 + [1.0 / 18.0] * 4 + [0.0] * 20
    lengths = [1.0] * (len(curvatures) - 1)
    speeds = curvature_only_profile(lengths, curvatures, limits, stop_at_end=False, min_curve_length_m=5.0)
    assert min(speeds) == pytest.approx(math.sqrt(3.0 * 18.0))
    assert speeds[0] == pytest.approx(40.0 * KMH)


def test_a_graded_corner_steps_down_once_to_its_tightest_radius_speed():
    """A corner tightening from R 25 m to R 12 m: one step to the R 12 m speed,
    not one step per radius."""

    from krowd_pp_stanleyff_control.speed_profile import curvature_only_profile

    curvatures = [0.0] * 20 + [1.0 / 25.0] * 5 + [1.0 / 18.0] * 5 + [1.0 / 12.0] * 5 + [0.0] * 30
    lengths = [1.0] * (len(curvatures) - 1)
    speeds = curvature_only_profile(lengths, curvatures, GRADED, stop_at_end=False, min_curve_length_m=5.0)
    slow = [v for v in speeds if v < 40.0 * KMH - 1e-9]
    first = speeds.index(slow[0])
    expected = (12.0 + 28.0 * (12.0 - 10.0) / 20.0) * KMH
    assert slow[0] == pytest.approx(expected)
    assert all(v == pytest.approx(expected) for v in speeds[first:35])


@pytest.mark.parametrize(
    "changes",
    [
        {"tight_curve_radius_m": 30.0},  # not below curve_radius_m
        {"tight_curve_radius_m": -1.0},
        {"curve_max_speed_mps": 45.0 * KMH},  # above the straight speed
    ],
)
def test_graded_validation_rejects_inconsistent_values(changes):
    with pytest.raises(ValueError):
        replace(GRADED, **changes).validate()


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
