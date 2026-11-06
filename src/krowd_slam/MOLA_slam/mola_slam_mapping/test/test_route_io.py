import csv
import math

import numpy as np
import pytest

from mola_slam_mapping.route_io import (
    load_route_files,
    nearest_waypoint_id,
    off_track_reason,
    point_ahead_on_route,
    project_onto_route,
    route_to_path,
)
from mola_slam_mapping.route_visualization import make_route_point


def _straight_route(count=6, spacing=1.0):
    """Waypoints one metre apart along +x, as recorded."""

    return [
        make_route_point(float(i) * spacing, 0.0, 0.0, 0.0, 3.0) for i in range(count)
    ]


def test_route_is_published_as_data_with_heading_not_only_as_a_drawing():
    """The MarkerArray carries float32 offsets from an anchor and no heading -
    a rendering format. nav_msgs/Path is the interface a follower consumes."""

    from builtin_interfaces.msg import Time

    route = [
        make_route_point(302473.404, 4123843.543, 27.9, 1.093938, 3.0),
        make_route_point(302474.000, 4123844.000, 28.0, -0.5, 3.0),
    ]

    path = route_to_path(route, "map", Time(sec=7, nanosec=500))

    assert path.header.frame_id == "map"
    assert len(path.poses) == len(route)
    first = path.poses[0]
    assert first.header.frame_id == "map"
    assert (first.header.stamp.sec, first.header.stamp.nanosec) == (7, 500)
    # Full double precision, unlike the marker's anchor-plus-float32 vertices.
    assert first.pose.position.x == 302473.404
    assert first.pose.position.y == 4123843.543
    assert first.pose.position.z == 27.9

    # The recorded yaw survives as a quaternion, both signs of it.
    for pose, point in zip(path.poses, route):
        yaw = 2.0 * math.atan2(pose.pose.orientation.z, pose.pose.orientation.w)
        assert yaw == pytest.approx(point.yaw)
        norm = math.hypot(pose.pose.orientation.z, pose.pose.orientation.w)
        assert norm == pytest.approx(1.0)

    assert route_to_path([], "map", Time()).poses == []


def test_follow_target_stays_exactly_the_lookahead_distance_ahead():
    """Interpolated, not snapped to a waypoint: snapping would let the distance
    drift between ld and ld + waypoint spacing and jump a metre at a time."""

    route = _straight_route(count=16)  # 15 m, long enough to stay off the end

    for vehicle_x in (0.0, 0.4, 1.0, 2.7, 5.5):
        projection = project_onto_route(route, vehicle_x, 0.0)
        target = point_ahead_on_route(route, projection, 5.0)
        assert target.distance_m == pytest.approx(5.0)
        assert target.x == pytest.approx(vehicle_x + 5.0)
        assert not target.reached_route_end

    # The fractional index advances continuously with the vehicle, and is the
    # same coordinate the traffic-detect zones use.
    positions = [
        point_ahead_on_route(route, project_onto_route(route, x, 0.0), 5.0).route_position
        for x in (2.0, 2.25, 2.5, 2.75, 3.0)
    ]
    assert positions == pytest.approx([7.0, 7.25, 7.5, 7.75, 8.0])
    assert all(b > a for a, b in zip(positions, positions[1:]))


def test_follow_target_reports_the_waypoint_it_sits_nearest():
    route = _straight_route(count=16)
    target = point_ahead_on_route(route, project_onto_route(route, 2.4, 0.0), 5.0)

    assert target.route_position == pytest.approx(7.4)
    assert target.nearest_waypoint_index == 7
    assert target.segment_index == 7
    # Just past the midpoint it rounds to the next waypoint.
    later = point_ahead_on_route(route, project_onto_route(route, 2.6, 0.0), 5.0)
    assert later.nearest_waypoint_index == 8


def test_follow_target_follows_the_route_around_a_corner():
    """Arc length along the route, not a straight-line radius: a Euclidean
    lookahead would cut the corner and leave the route."""

    # 5 m east, then turn north for 5 m.
    route = [make_route_point(float(i), 0.0, 0.0, 0.0, 3.0) for i in range(6)]
    route += [
        make_route_point(5.0, float(i), 0.0, math.pi / 2.0, 3.0) for i in range(1, 6)
    ]

    target = point_ahead_on_route(route, project_onto_route(route, 2.0, 0.0), 5.0)

    # 3 m to the corner, then 2 m up the second leg - on the route.
    assert (target.x, target.y) == pytest.approx((5.0, 2.0))
    # A straight 5 m radius from (2,0) would have landed off the route entirely.
    assert math.hypot(target.x - 2.0, target.y - 0.0) < 5.0


def test_follow_target_clamps_at_the_route_end_and_says_so():
    """The controller must be able to tell "5 m ahead" from "the route ran out"."""

    route = _straight_route(count=6)  # 5 m long
    target = point_ahead_on_route(route, project_onto_route(route, 3.0, 0.0), 5.0)

    assert (target.x, target.y) == pytest.approx((5.0, 0.0))
    assert target.reached_route_end
    assert target.distance_m == pytest.approx(2.0)
    assert target.nearest_waypoint_index == 5


def test_degenerate_lookahead_inputs():
    route = _straight_route()
    projection = project_onto_route(route, 1.0, 0.0)

    # Zero distance is the projection itself, not an error.
    here = point_ahead_on_route(route, projection, 0.0)
    assert (here.x, here.y) == pytest.approx((1.0, 0.0))

    assert point_ahead_on_route(route, None, 5.0) is None
    assert point_ahead_on_route(route[:1], projection, 5.0) is None
    assert point_ahead_on_route(route, projection, math.nan) is None
    with pytest.raises(ValueError, match="negative"):
        point_ahead_on_route(route, projection, -1.0)


def test_repeated_waypoints_do_not_stall_the_walk():
    """A stopped vehicle can record two waypoints at the same place."""

    route = [
        make_route_point(0.0, 0.0, 0.0, 0.0, 3.0),
        make_route_point(0.0, 0.0, 0.0, 0.0, 3.0),
        make_route_point(3.0, 0.0, 0.0, 0.0, 3.0),
        make_route_point(9.0, 0.0, 0.0, 0.0, 3.0),
    ]

    target = point_ahead_on_route(route, project_onto_route(route, 0.0, 0.0), 5.0)
    assert target.x == pytest.approx(5.0)
    assert not target.reached_route_end


def test_not_reaching_the_route_yet_is_not_reported_as_a_corridor_violation():
    """Measured on a real replay: 0.08 m off a 3 m half-width corridor was
    reported as "outside the saved corridor boundaries". The vehicle had simply
    not reached the first waypoint yet - a normal start-of-run state that read
    like a bug."""

    route = [make_route_point(float(i) * 5.0, 0.0, 0.0, 0.0, 3.0) for i in range(4)]

    on_route = project_onto_route(route, 7.0, 0.5)
    assert on_route.on_track
    assert off_track_reason(on_route) is None

    before = project_onto_route(route, -2.0, 0.08)
    assert not before.on_track
    reason = off_track_reason(before)
    assert "before the first waypoint" in reason
    assert "not a corridor violation" in reason

    past = project_onto_route(route, 20.0, 0.0)
    assert not past.on_track
    assert "past the last waypoint" in off_track_reason(past)

    # A genuine violation still says so, with the boundaries it broke.
    wide = project_onto_route(route, 7.0, 4.0)
    assert not wide.on_track
    violation = off_track_reason(wide)
    assert "outside the corridor" in violation
    assert "4.00 m off the saved centerline" in violation

    assert off_track_reason(None) == "no saved route segment to compare against"


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_saved_corridor_is_joined_by_idx_and_used_for_on_track(tmp_path):
    waypoints = tmp_path / "waypoints.csv"
    corridor = tmp_path / "corridor.csv"
    _write_csv(
        waypoints,
        ["idx", "utm_zone", "utm_x", "utm_y", "utm_z", "yaw_rad"],
        [
            {"idx": 0, "utm_zone": "52N", "utm_x": 0, "utm_y": 0, "utm_z": 0, "yaw_rad": 0},
            {"idx": 1, "utm_zone": "52N", "utm_x": 10, "utm_y": 0, "utm_z": 0, "yaw_rad": 0},
        ],
    )
    # Deliberately reverse row order. Saved width is 2m, not fallback 1m.
    _write_csv(
        corridor,
        ["idx", "utm_zone", "left_x", "left_y", "right_x", "right_y"],
        [
            {"idx": 1, "utm_zone": "52N", "left_x": 10, "left_y": 2, "right_x": 10, "right_y": -2},
            {"idx": 0, "utm_zone": "52N", "left_x": 0, "left_y": 2, "right_x": 0, "right_y": -2},
        ],
    )

    loaded = load_route_files(waypoints, corridor, 1.0)
    assert loaded.used_saved_corridor
    assert loaded.utm_zone == "52N"
    assert loaded.points[0].left_y == 2.0
    assert project_onto_route(loaded.points, 5.0, 1.5).on_track
    assert not project_onto_route(loaded.points, 5.0, 2.1).on_track
    assert not project_onto_route(loaded.points, -0.1, 0.0).on_track
    assert not project_onto_route(loaded.points, 10.1, 0.0).on_track

    numpy_projection = project_onto_route(
        loaded.points, np.float64(5.0), np.float64(1.5)
    )
    assert numpy_projection is not None
    assert type(numpy_projection.on_track) is bool


def test_legacy_route_rebuilds_corridor_and_defaults_z(tmp_path):
    waypoints = tmp_path / "waypoints.csv"
    _write_csv(
        waypoints,
        ["utm_zone", "utm_x", "utm_y", "yaw_rad"],
        [
            {"utm_zone": "52N", "utm_x": 0, "utm_y": 0, "yaw_rad": 0},
            {"utm_zone": "52N", "utm_x": 1, "utm_y": 0, "yaw_rad": 0},
        ],
    )
    loaded = load_route_files(waypoints, None, 1.25)
    assert not loaded.used_saved_corridor
    assert loaded.points[0].z == 0.0
    assert loaded.points[0].left_y == pytest.approx(1.25)


def test_route_loader_rejects_non_finite_coordinate(tmp_path):
    waypoints = tmp_path / "waypoints.csv"
    _write_csv(
        waypoints,
        ["utm_zone", "utm_x", "utm_y", "yaw_rad"],
        [
            {"utm_zone": "52N", "utm_x": "nan", "utm_y": 0, "yaw_rad": 0},
            {"utm_zone": "52N", "utm_x": 1, "utm_y": 0, "yaw_rad": 0},
        ],
    )
    with pytest.raises(ValueError, match="non-finite"):
        load_route_files(waypoints, None, 1.0)


def test_nearest_waypoint_id_is_the_closer_end_of_the_nearest_segment():
    route = [make_route_point(index * 2.0, 0.0, 0.0, 0.0, 1.5) for index in range(6)]

    # Between waypoints 2 (x=4) and 3 (x=6): nearer 2, then nearer 3.
    assert nearest_waypoint_id(project_onto_route(route, 4.9, 0.8)) == 2
    assert nearest_waypoint_id(project_onto_route(route, 5.1, -0.8)) == 3
    # Before the start and past the end clamp to the end waypoints.
    assert nearest_waypoint_id(project_onto_route(route, -3.0, 0.0)) == 0
    assert nearest_waypoint_id(project_onto_route(route, 20.0, 0.0)) == 5
    assert nearest_waypoint_id(None) == -1
