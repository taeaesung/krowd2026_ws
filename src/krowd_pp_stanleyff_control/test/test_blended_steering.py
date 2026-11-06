"""The blend WITH feedforward: weight curve, sign conventions, the geometry
preserved at every weight, and gain 0 reproducing the sibling package."""

import math

import pytest

from krowd_pp_stanleyff_control.blended_steering import (
    BlendParams,
    blend,
    compute_steering,
    geometric_steering,
    stanley_weight,
)
from krowd_pp_stanleyff_control.geometry import to_body_frame
from krowd_pp_stanleyff_control.pure_pursuit import pure_pursuit_steering

L = 1.34
P = BlendParams(wheelbase_m=L)
P_NO_FF = BlendParams(wheelbase_m=L, feedforward_gain=0.0)


def test_weight_curve_is_zero_on_straights_and_capped_in_tight_bends():
    def w(radius):
        return stanley_weight(1.0 / radius, P.straight_radius_m, P.curve_radius_m, P.max_stanley_weight)

    assert w(1e9) == 0.0
    assert w(50.0) == 0.0
    assert w(30.0) == pytest.approx(0.5 * (1 / 30 - 1 / 50) / (1 / 5 - 1 / 50))
    assert w(5.0) == pytest.approx(0.5)
    assert w(3.0) == pytest.approx(0.5)
    assert stanley_weight(-1.0 / 5.0, 50.0, 5.0, 0.5) == pytest.approx(0.5)


def test_blend_params_validate():
    with pytest.raises(ValueError):
        BlendParams(wheelbase_m=L, max_stanley_weight=1.2).validate()
    with pytest.raises(ValueError):
        BlendParams(wheelbase_m=L, feedforward_gain=-0.1).validate()
    P.validate()
    P_NO_FF.validate()


def test_both_laws_steer_left_for_a_route_to_the_left():
    bx, by = to_body_frame(0.0, 0.0, 0.0, 6.0, 1.0)
    assert pure_pursuit_steering(bx, by, L) > 0.0
    left_of_route = compute_steering(0.0, 0.0, +0.5, 2.0, 0.0, 0.0, P)
    assert left_of_route.cross_track_term < 0.0 and left_of_route.delta_stanley < 0.0
    route_to_the_left = compute_steering(0.0, math.radians(10.0), 0.0, 2.0, 0.0, 0.0, P)
    assert route_to_the_left.heading_term == pytest.approx(0.6 * math.radians(10.0))
    # A left-hand curve (kappa > 0) feeds forward a LEFT steering angle.
    assert compute_steering(0.0, 0.0, 0.0, 2.0, 1.0 / 10.0, 0.2, P).feedforward > 0.0
    assert compute_steering(0.0, 0.0, 0.0, 2.0, -1.0 / 10.0, -0.2, P).feedforward < 0.0


def test_on_a_straight_feedforward_is_zero_and_the_output_is_pure_pursuit():
    bd = compute_steering(math.radians(4.0), math.radians(20.0), 1.0, 3.0, 0.0, math.radians(10.0), P)
    assert bd.weight == 0.0 and bd.feedforward == 0.0
    assert math.degrees(bd.yaw_damping) == pytest.approx(-1.5)
    assert bd.delta == pytest.approx(math.radians(4.0) + bd.yaw_damping)


def test_pure_pursuit_on_an_arc_returns_the_geometric_angle():
    R = 4.9
    for ld in (2.0, 3.0, 6.0):
        ang = ld / R
        tx, ty = R * math.sin(ang), R * (1.0 - math.cos(ang))
        bx, by = to_body_frame(0.0, 0.0, 0.0, tx, ty)
        assert pure_pursuit_steering(bx, by, L) == pytest.approx(geometric_steering(L, 1.0 / R), rel=1e-9)


def test_with_feedforward_the_blend_preserves_the_geometry_at_any_weight():
    """On the correct line both laws return atan(L/R), so the blend does too -
    whatever the weight. This is what lets max_stanley_weight rise above 0.5."""

    R = 4.9
    needed = geometric_steering(L, 1.0 / R)
    v = 12.0 / 3.6
    for w_max in (0.0, 0.5, 0.7, 1.0):
        p = BlendParams(wheelbase_m=L, max_stanley_weight=w_max)
        bd = compute_steering(needed, 0.0, 0.0, v, 1.0 / R, v / R, p)
        assert bd.weight == pytest.approx(w_max)
        assert bd.feedforward == pytest.approx(needed)
        assert bd.delta_stanley == pytest.approx(needed)
        assert bd.delta == pytest.approx(needed)


def test_gain_zero_reproduces_the_no_feedforward_sibling():
    R = 4.9
    needed = geometric_steering(L, 1.0 / R)
    v = 12.0 / 3.6
    bd = compute_steering(needed, 0.0, 0.0, v, 1.0 / R, v / R, P_NO_FF)
    assert bd.feedforward == 0.0 and bd.delta_stanley == pytest.approx(0.0)
    assert bd.delta == pytest.approx(0.5 * needed)  # the geometry loss the sibling accepts
    assert blend(needed, 0.0, 0.5) == pytest.approx(0.5 * needed)
