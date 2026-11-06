import math

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped

from mola_slam_mapping.route_visualization import (
    AVOIDANCE_FILL_MARKER_ID,
    FIRST_LABEL_MARKER_ID,
    HIGHLIGHT_FILL_MARKER_ID,
    SLOW_DOWN_FILL_MARKER_ID,
    WAYPOINT_DOT_MARKER_ID,
    ZONE_FILL_MARKER_IDS,
    build_route_marker_array,
    make_route_point,
    route_label_indices,
    transform_route,
)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2.0, -math.pi / 4.0])
def test_corridor_offsets_are_perpendicular(yaw):
    point = make_route_point(4.0, -2.0, 0.5, yaw, 1.7)
    heading = (math.cos(yaw), math.sin(yaw))
    for boundary in (
        (point.left_x - point.x, point.left_y - point.y),
        (point.right_x - point.x, point.right_y - point.y),
    ):
        assert math.hypot(*boundary) == pytest.approx(1.7)
        assert boundary[0] * heading[0] + boundary[1] * heading[1] == pytest.approx(
            0.0
        )


def test_large_utm_markers_use_anchor_and_relative_vertices():
    route = [
        make_route_point(350000.0 + index * 10.0, 4100000.0, 4.0, 0.0, 1.0)
        for index in range(3)
    ]
    markers = build_route_marker_array(route, "map", Time(), label_every_n=0).markers

    assert {marker.id for marker in markers} == {
        0, 1, 2, 3, WAYPOINT_DOT_MARKER_ID, *ZONE_FILL_MARKER_IDS
    }
    drawn = [marker for marker in markers if marker.action == marker.ADD]
    assert {marker.id for marker in drawn} == {0, 3, WAYPOINT_DOT_MARKER_ID}
    for marker in drawn:
        assert marker.pose.position.x == route[0].x
        assert marker.pose.position.y == route[0].y
        assert marker.pose.orientation.w == 1.0
        assert all(abs(point.x) < 100.0 for point in marker.points)
        assert all(abs(point.y) < 100.0 for point in marker.points)
    fill = next(marker for marker in markers if marker.id == 3)
    assert len(fill.points) == 6 * (len(route) - 1)


def test_corridor_is_a_filled_surface_with_no_boundary_lines():
    """Operators asked for the drivable area as a painted region, edges off.

    The two boundary ids stay reserved and are published as DELETE so a viewer
    that still holds the old polylines clears them.
    """

    route = [make_route_point(index * 5.0, 0.0, 0.0, 0.0, 1.5) for index in range(4)]
    markers = build_route_marker_array(route, "map", Time()).markers

    fill = next(marker for marker in markers if marker.id == 3)
    assert fill.type == fill.TRIANGLE_LIST
    assert fill.action == fill.ADD
    assert fill.color.g > fill.color.r and fill.color.g > fill.color.b
    assert 0.0 < fill.color.a < 1.0

    centerline = next(marker for marker in markers if marker.id == 0)
    assert centerline.type == centerline.LINE_STRIP
    assert centerline.action == centerline.ADD

    for marker_id in (1, 2):
        edge = next(marker for marker in markers if marker.id == marker_id)
        assert edge.action == edge.DELETE
        assert list(edge.points) == []


def test_single_waypoint_is_visible_without_invalid_line_or_triangle_geometry():
    route = [make_route_point(3.0, -4.0, 0.5, 0.0, 1.0)]
    markers = build_route_marker_array(
        route, "map", Time(), "waypoints", label_every_n=0
    ).markers

    assert len(markers) == 9
    start = next(marker for marker in markers if marker.id == 0)
    assert start.type == start.SPHERE
    assert start.action == start.ADD
    assert start.pose.position.x == pytest.approx(3.0)
    assert start.pose.position.y == pytest.approx(-4.0)
    assert start.scale.x > 0.0
    assert all(
        marker.action == marker.DELETE
        for marker in markers
        if marker.id in (1, 2, 3, *ZONE_FILL_MARKER_IDS)
    )


def test_empty_route_deletes_all_stable_marker_ids():
    markers = build_route_marker_array(
        [], "map", Time(), "waypoints", previous_label_count=3
    ).markers

    assert {marker.id for marker in markers} == {
        0, 1, 2, 3, WAYPOINT_DOT_MARKER_ID, *ZONE_FILL_MARKER_IDS,
        FIRST_LABEL_MARKER_ID, FIRST_LABEL_MARKER_ID + 1, FIRST_LABEL_MARKER_ID + 2,
    }
    assert all(marker.action == marker.DELETE for marker in markers)


def test_highlighted_index_ranges_move_their_corridor_segments_to_the_red_fill():
    """Traffic-light zones: red where the range is, green elsewhere, no overlap."""

    route = _metre_spaced_route(30)

    markers = build_route_marker_array(
        route,
        "map",
        Time(),
        label_every_n=0,
        highlight_index_ranges=[(5, 8), (20, 22)],
    ).markers

    green = next(m for m in markers if m.id == 3)
    red = next(m for m in markers if m.id == HIGHLIGHT_FILL_MARKER_ID)
    assert red.type == red.TRIANGLE_LIST
    assert red.action == red.ADD
    assert red.color.r > red.color.g and red.color.r > red.color.b
    assert red.color.a > green.color.a  # stands out against the green fill
    # Range (5, 8) is the stretch from waypoint 5 to 8: segments 5, 6, 7;
    # (20, 22) adds segments 20, 21. Six vertices per segment.
    assert len(red.points) == 6 * (3 + 2)
    assert len(green.points) == 6 * (29 - 5)
    # Both fills share the anchor and are route-relative.
    assert red.pose.position.x == green.pose.position.x == route[0].x
    red_xs = sorted({round(p.x, 6) for p in red.points})
    assert red_xs[0] == pytest.approx(5.0) and red_xs[-1] == pytest.approx(22.0)


def test_no_highlight_ranges_deletes_the_red_fill():
    """A cleared zone list must clear the red a viewer still holds."""

    route = _metre_spaced_route(5)
    markers = build_route_marker_array(route, "map", Time(), label_every_n=0).markers

    red = next(m for m in markers if m.id == HIGHLIGHT_FILL_MARKER_ID)
    assert red.action == red.DELETE
    green = next(m for m in markers if m.id == 3)
    assert green.action == green.ADD
    assert len(green.points) == 6 * 4


def test_slow_down_is_orange_and_avoidance_is_sky_blue():
    route = _metre_spaced_route(30)

    markers = build_route_marker_array(
        route,
        "map",
        Time(),
        label_every_n=0,
        highlight_index_ranges=[(10, 12)],
        slow_down_index_ranges=[(2, 10)],
        avoidance_index_ranges=[(20, 25)],
    ).markers

    green = next(m for m in markers if m.id == 3)
    red = next(m for m in markers if m.id == HIGHLIGHT_FILL_MARKER_ID)
    orange = next(m for m in markers if m.id == SLOW_DOWN_FILL_MARKER_ID)
    blue = next(m for m in markers if m.id == AVOIDANCE_FILL_MARKER_ID)
    # Orange: red full, green about half, no blue.
    assert orange.color.r == 1.0 and 0.3 < orange.color.g < 0.7
    assert orange.color.b == 0.0
    # Sky blue: blue full, green high, red low.
    assert blue.color.b == 1.0 and blue.color.g > 0.6 and blue.color.r < 0.5
    for surface in (orange, blue):
        assert surface.type == surface.TRIANGLE_LIST
        assert surface.action == surface.ADD
        assert surface.color.a == red.color.a
    # Zones that only touch at a waypoint (slow-down 2-10, traffic 10-12)
    # share no segment, and no segment is painted twice.
    assert len(red.points) == 6 * 2
    assert len(orange.points) == 6 * 8
    assert len(blue.points) == 6 * 5
    assert len(green.points) == 6 * (29 - 2 - 8 - 5)


def test_a_segment_in_two_zone_kinds_is_painted_once_in_the_first():
    route = _metre_spaced_route(10)

    markers = build_route_marker_array(
        route,
        "map",
        Time(),
        label_every_n=0,
        highlight_index_ranges=[(3, 5)],
        slow_down_index_ranges=[(0, 5)],
        avoidance_index_ranges=[(4, 9)],
    ).markers

    by_id = {m.id: m for m in markers}
    assert len(by_id[HIGHLIGHT_FILL_MARKER_ID].points) == 6 * 2  # 3, 4
    assert len(by_id[SLOW_DOWN_FILL_MARKER_ID].points) == 6 * 3  # 0, 1, 2
    assert len(by_id[AVOIDANCE_FILL_MARKER_ID].points) == 6 * 4  # 5, 6, 7, 8
    assert by_id[3].action == by_id[3].DELETE  # nothing left for green


def test_a_range_reaching_past_the_route_only_paints_what_exists():
    route = _metre_spaced_route(5)  # waypoints 0..4, segments 0..3

    markers = build_route_marker_array(
        route, "map", Time(), label_every_n=0, highlight_index_ranges=[(3, 9)]
    ).markers

    red = next(m for m in markers if m.id == HIGHLIGHT_FILL_MARKER_ID)
    assert len(red.points) == 6 * 1  # segment 3 only


def test_drawn_markers_are_frame_locked_so_a_moving_fixed_frame_stays_smooth():
    """RViz places a marker once unless frame_locked, then per render frame.

    The route is world-fixed and published rarely, so without this it freezes
    wherever it last landed and jumps on the next publication as soon as the
    fixed frame itself moves - any vehicle-frame view.
    """

    route = _metre_spaced_route(6)
    markers = build_route_marker_array(
        route, "map", Time(), label_every_n=2, highlight_index_ranges=[(1, 2)]
    ).markers

    drawn = [m for m in markers if m.action == m.ADD]
    assert drawn, "expected geometry to be drawn"
    assert all(m.frame_locked for m in drawn)


def test_waypoints_are_drawn_as_purple_dots():
    route = [make_route_point(index * 2.0, 0.0, 0.5, 0.0, 1.5) for index in range(5)]

    dots = next(
        m
        for m in build_route_marker_array(route, "map", Time(), label_every_n=0).markers
        if m.id == WAYPOINT_DOT_MARKER_ID
    )

    # One SPHERE_LIST, not five SPHERE markers.
    assert dots.type == dots.SPHERE_LIST
    assert dots.action == dots.ADD
    assert len(dots.points) == len(route)
    assert dots.scale.x == dots.scale.y == dots.scale.z > 0.0
    # Purple: red and blue present, green suppressed, blue dominant.
    assert dots.color.b > dots.color.r > dots.color.g


def test_labels_carry_the_waypoint_id_and_short_coordinates():
    route = [make_route_point(index * 2.0, -1.25, 0.5, 0.0, 1.5) for index in range(11)]

    labels = [
        m
        for m in build_route_marker_array(route, "map", Time(), label_every_n=5).markers
        if m.id >= FIRST_LABEL_MARKER_ID
    ]

    assert [m.text for m in labels] == [
        "#0  0.0, -1.2",
        "#5  10.0, -1.2",
        "#10  20.0, -1.2",
    ]
    for label in labels:
        assert label.type == label.TEXT_VIEW_FACING
        assert label.action == label.ADD
        assert label.scale.z > 0.0
        # Text has no `points`, so it must carry an absolute position.
        assert label.pose.position.z == pytest.approx(1.3)
    # Raised above the dots so the two do not overlap in the view.
    assert labels[0].pose.position.x == pytest.approx(0.0)
    assert labels[1].pose.position.x == pytest.approx(10.0)


def _metre_spaced_route(count: int):
    """A straight route along +x with one waypoint per metre, as recorded."""

    return [make_route_point(float(index), 0.0, 0.0, 0.0, 1.5) for index in range(count)]


def test_label_indices_always_include_the_last_waypoint():
    """Whole-route fallback, for a viewer that has no vehicle position."""

    assert route_label_indices(_metre_spaced_route(11), 5) == [0, 5, 10]
    assert route_label_indices(_metre_spaced_route(12), 5) == [0, 5, 10, 11]
    assert route_label_indices(_metre_spaced_route(3), 1) == [0, 1, 2]
    # Disabled, and degenerate inputs.
    assert route_label_indices(_metre_spaced_route(10), 0) == []
    assert route_label_indices([], 5) == []


def test_label_indices_form_a_window_around_the_vehicle():
    route = _metre_spaced_route(20)

    assert route_label_indices(route, 0, (10.0, 0.0), 3.5) == [7, 8, 9, 10, 11, 12, 13]
    # Driven off the route: the window narrows to what is genuinely nearby.
    assert route_label_indices(route, 0, (10.0, 3.0), 3.5) == [9, 10, 11]
    # A vehicle position with no radius means no labels - never a silent
    # fallback to labelling every n-th waypoint of the whole route.
    assert route_label_indices(route, 5, (10.0, 0.0), 0.0) == []


def test_only_waypoints_near_the_vehicle_are_labelled():
    route = _metre_spaced_route(30)

    markers = build_route_marker_array(
        route,
        "map",
        Time(),
        label_every_n=0,
        label_center=(20.0, 0.0),
        label_radius_m=2.5,
    ).markers

    labels = [m for m in markers if m.id >= FIRST_LABEL_MARKER_ID]
    assert [m.text for m in labels] == [
        "#18  18.0, 0.0",
        "#19  19.0, 0.0",
        "#20  20.0, 0.0",
        "#21  21.0, 0.0",
        "#22  22.0, 0.0",
    ]
    # Only the text is windowed: every waypoint keeps its dot.
    dots = next(m for m in markers if m.id == WAYPOINT_DOT_MARKER_ID)
    assert len(dots.points) == len(route)


def test_a_moving_window_deletes_the_labels_it_left_behind():
    """The window shrinks at the route ends; orphan text must not stay."""

    route = _metre_spaced_route(30)

    markers = build_route_marker_array(
        route,
        "map",
        Time(),
        label_every_n=0,
        label_center=(0.0, 0.0),
        label_radius_m=2.5,
        previous_label_count=5,
    ).markers

    labels = [m for m in markers if m.id >= FIRST_LABEL_MARKER_ID]
    added = [m for m in labels if m.action == m.ADD]
    deleted = [m for m in labels if m.action == m.DELETE]
    # At the very start only waypoints 0..2 are within the radius.
    assert [m.text for m in added] == ["#0  0.0, 0.0", "#1  1.0, 0.0", "#2  2.0, 0.0"]
    assert [m.id for m in deleted] == [FIRST_LABEL_MARKER_ID + 3, FIRST_LABEL_MARKER_ID + 4]


def test_a_shortened_route_deletes_the_labels_it_no_longer_has():
    """A bag replayed from the start must not leave orphan text floating."""

    route = [make_route_point(index * 2.0, 0.0, 0.0, 0.0, 1.5) for index in range(3)]

    markers = build_route_marker_array(
        route, "map", Time(), label_every_n=5, previous_label_count=7
    ).markers

    labels = [m for m in markers if m.id >= FIRST_LABEL_MARKER_ID]
    added = [m for m in labels if m.action == m.ADD]
    deleted = [m for m in labels if m.action == m.DELETE]
    assert len(added) == 2  # index 0 and the final index 2
    assert [m.id for m in deleted] == [FIRST_LABEL_MARKER_ID + i for i in range(2, 7)]


def test_transform_route_preserves_geometry_at_large_coordinates():
    route = [
        make_route_point(350000.0, 4100000.0, 3.0, 0.0, 1.25),
        make_route_point(350010.0, 4100000.0, 3.0, 0.0, 1.25),
    ]
    transform = TransformStamped()
    transform.transform.translation.x = -350000.0
    transform.transform.translation.y = -4100000.0
    transform.transform.rotation.w = 1.0

    local = transform_route(route, transform)
    assert (local[0].x, local[0].y) == pytest.approx((0.0, 0.0))
    assert (local[1].x, local[1].y) == pytest.approx((10.0, 0.0))
    assert math.hypot(
        local[0].left_x - local[0].x,
        local[0].left_y - local[0].y,
    ) == pytest.approx(1.25)
