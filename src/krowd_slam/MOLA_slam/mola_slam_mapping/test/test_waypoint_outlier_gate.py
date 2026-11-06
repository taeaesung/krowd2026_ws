import math

from mola_slam_mapping.route_visualization import make_route_point
from mola_slam_mapping.waypoint_saver_node import (
    waypoint_jump_limit_m,
    waypoint_step_distance_3d,
)


def test_waypoint_step_distance_uses_all_three_dimensions():
    previous = make_route_point(10.0, 20.0, 30.0, 0.0, 1.0)
    current = make_route_point(13.0, 24.0, 42.0, 0.0, 1.0)

    assert math.isclose(
        waypoint_step_distance_3d(previous, current), 13.0, rel_tol=1e-12
    )


def test_0824_pose_spike_exceeds_default_gate_but_return_does_not():
    last_good = make_route_point(
        326400.823, 4129999.257, 120.167, 0.0, 1.0
    )
    spike = make_route_point(
        326412.280, 4130000.869, 198.077, 0.0, 1.0
    )
    returned = make_route_point(
        326398.934, 4129999.296, 115.723, 0.0, 1.0
    )

    assert waypoint_step_distance_3d(last_good, spike) > 5.0
    assert waypoint_step_distance_3d(last_good, returned) < 5.0


def test_jump_gate_expands_after_a_real_callback_gap():
    assert waypoint_jump_limit_m(5.0, 20.0, 0.1) == 5.0
    assert waypoint_jump_limit_m(5.0, 20.0, 2.0) == 40.0
