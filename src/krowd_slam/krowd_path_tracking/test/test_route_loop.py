"""Following a recorded loop round indefinitely instead of stopping at its end."""

import math

import pytest

from krowd_path_tracking.route_geometry import loop_closure, unroll_loop


def _loop(radius, arc_m, spacing=1.0):
    """Counter-clockwise circle from (0, -radius), heading +x, `arc_m` long."""

    count = int(arc_m / spacing) + 1
    return [
        (
            radius * math.sin(index * spacing / radius),
            -radius * math.cos(index * spacing / radius),
        )
        for index in range(count)
    ]


def _out_and_back(length_m, lane_offset_m):
    """Out along +x, a U-turn, and back along the next lane - not a loop."""

    out = [(float(x), 0.0) for x in range(int(length_m) + 1)]
    radius = 0.5 * lane_offset_m
    turn = [
        (
            length_m + radius * math.sin(step * math.pi / 8),
            radius - radius * math.cos(step * math.pi / 8),
        )
        for step in range(1, 8)
    ]
    back = [(float(x), lane_offset_m) for x in range(int(length_m), -1, -1)]
    return out + turn + back


def test_a_loop_stopped_short_of_its_start_is_joined_over_the_gap():
    radius = 20.0
    points = _loop(radius, 2.0 * math.pi * radius - 5.0)

    closure = loop_closure(points, max_gap_m=10.0)

    assert closure is not None
    head, tail = closure
    assert head == 0
    assert tail == len(points) - 1
    assert math.dist(points[head], points[tail]) == pytest.approx(5.0, abs=1.0)


def test_a_loop_driven_past_its_start_is_joined_where_the_passes_meet():
    """Joining the very end back to the very start would point backwards."""

    radius = 20.0
    points = _loop(radius, 2.0 * math.pi * radius + 15.0)

    head, tail = loop_closure(points, max_gap_m=10.0)

    assert math.dist(points[head], points[tail]) < 1.0
    # The loop left over is one lap, not the overlap as well.
    assert (tail - head) == pytest.approx(2.0 * math.pi * radius, abs=1.5)


def test_an_out_and_back_route_is_not_a_loop():
    """Its end lies one lane beside its start, facing the other way."""

    points = _out_and_back(200.0, 5.0)
    assert math.dist(points[0], points[-1]) == pytest.approx(5.0)

    assert loop_closure(points, max_gap_m=10.0) is None


def test_a_gap_wider_than_the_limit_is_not_joined():
    radius = 20.0
    points = _loop(radius, 2.0 * math.pi * radius - 15.0)

    assert loop_closure(points, max_gap_m=10.0) is None


def test_loop_closure_can_be_disabled_and_ignores_short_routes():
    radius = 20.0
    points = _loop(radius, 2.0 * math.pi * radius - 5.0)
    assert loop_closure(points, max_gap_m=0.0) is None
    assert loop_closure(_loop(5.0, 2.0 * math.pi * 5.0 - 2.0), max_gap_m=10.0) is None


def test_unrolled_laps_share_their_seam_waypoint():
    loop = ["a", "b", "c", "a"]
    assert unroll_loop(loop, 1) == loop
    assert unroll_loop(loop, 3) == ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
    with pytest.raises(ValueError):
        unroll_loop(loop, 0)


# ---- controller side: needs the interface messages -------------------------


def _trajectory(points, goal_mode):
    trajectory_msg = pytest.importorskip("krowd_control_interfaces.msg")
    from geometry_msgs.msg import PoseStamped

    message = trajectory_msg.Trajectory()
    message.trajectory_id = "test"
    message.goal_mode = goal_mode
    message.path.header.frame_id = "map"
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0
        message.path.poses.append(pose)
    message.curvatures = [0.0] * len(points)
    message.target_speeds = [1.0] * len(points)
    return message


def _closed(points):
    head, tail = loop_closure(points, max_gap_m=10.0)
    return points[head : tail + 1] + [points[head]]


def test_the_lookahead_walks_across_the_seam_of_a_loop():
    pytest.importorskip("rclpy")
    from krowd_control_interfaces.msg import Trajectory
    from mola_slam_mapping.route_io import point_ahead_on_route, project_onto_route

    from krowd_path_tracking.pure_pursuit_node import track_trajectory

    radius = 20.0
    loop = _closed(_loop(radius, 2.0 * math.pi * radius - 5.0))
    tracked = track_trajectory(
        _trajectory(loop, Trajectory.FOLLOW_UNTIL_REPLACED), 3.0
    )
    assert tracked.closed_loop

    # 1 m before the last waypoint, 8 m ahead lies past the start.
    x, y = loop[-3]
    projection = project_onto_route(tracked.route, x, y)
    target = point_ahead_on_route(
        tracked.ahead_route, tracked.ahead_of(projection), 8.0
    )

    assert not target.reached_route_end
    start_x, start_y = loop[0]
    to_start = math.dist(loop[-3], loop[-2]) + math.dist(loop[-2], loop[-1])
    along_after_start = 8.0 - to_start
    assert math.hypot(target.x - start_x, target.y - start_y) == pytest.approx(
        along_after_start, abs=0.1
    )


def test_only_open_routes_end_and_only_stop_at_end_latches_a_goal():
    pytest.importorskip("rclpy")
    from krowd_control_interfaces.msg import Trajectory

    from krowd_path_tracking.pure_pursuit_node import (
        end_of_route_reason,
        track_trajectory,
    )

    radius = 20.0
    open_route = _loop(radius, 2.0 * math.pi * radius - 5.0)
    loop = _closed(open_route)

    follow_loop = track_trajectory(
        _trajectory(loop, Trajectory.FOLLOW_UNTIL_REPLACED), 3.0
    )
    follow_open = track_trajectory(
        _trajectory(open_route, Trajectory.FOLLOW_UNTIL_REPLACED), 3.0
    )
    stop_loop = track_trajectory(_trajectory(loop, Trajectory.STOP_AT_END), 3.0)

    assert end_of_route_reason(follow_loop, 0.0, 1.0) is None
    assert end_of_route_reason(follow_open, 0.5, 1.0).startswith("END_OF_ROUTE")
    assert end_of_route_reason(follow_open, 1.5, 1.0) is None
    # STOP_AT_END never treats a route as a loop.
    assert not stop_loop.closed_loop
    assert end_of_route_reason(stop_loop, 0.5, 1.0).startswith("GOAL_REACHED")
