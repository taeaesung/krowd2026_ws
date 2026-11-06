from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


WAYPOINT_DOT_MARKER_ID = 4
"""SPHERE_LIST carrying every waypoint as one marker - 381 individual SPHERE
markers would cost 381 RViz objects for the same picture."""

HIGHLIGHT_FILL_MARKER_ID = 5
"""TRIANGLE_LIST with the corridor segments inside the highlighted index
ranges (traffic-light detection zones). They are left OUT of the green fill
(id 3) rather than painted over it: two translucent fills stacked would blend
to a muddy colour and z-fight."""

HIGHLIGHT_FILL_COLOR_RGBA = (1.0, 0.1, 0.1, 0.5)
"""Red, more opaque than the green fill so the zone reads at a glance."""

SLOW_DOWN_FILL_MARKER_ID = 6
"""TRIANGLE_LIST with the corridor segments inside the slow-down zones, cut out
of the green fill the same way as the red one."""

SLOW_DOWN_FILL_COLOR_RGBA = (1.0, 0.55, 0.0, 0.5)
"""Orange, as opaque as the red fill."""

AVOIDANCE_FILL_MARKER_ID = 7
"""TRIANGLE_LIST with the corridor segments inside the obstacle-avoidance
zones, cut out of the green fill the same way as the red one."""

AVOIDANCE_FILL_COLOR_RGBA = (0.3, 0.8, 1.0, 0.5)
"""Sky blue, as opaque as the red fill."""

SCHOOL_ZONE_FILL_MARKER_ID = 8
"""TRIANGLE_LIST with the corridor segments inside the child-protection (school)
zones, cut out of the green fill the same way as the red one."""

SCHOOL_ZONE_FILL_COLOR_RGBA = (1.0, 0.9, 0.0, 0.5)
"""Yellow, as opaque as the red fill (the centerline is opaque yellow on top)."""

ZONE_FILL_MARKER_IDS = (
    HIGHLIGHT_FILL_MARKER_ID,
    SLOW_DOWN_FILL_MARKER_ID,
    AVOIDANCE_FILL_MARKER_ID,
    SCHOOL_ZONE_FILL_MARKER_ID,
)

FIRST_LABEL_MARKER_ID = 100
"""Text labels occupy 100, 101, ... Kept well clear of the geometry ids (0-8)
so adding geometry later cannot silently overwrite a label."""

WAYPOINT_DOT_COLOR_RGBA = (0.75, 0.3, 1.0, 1.0)
"""Purple, brighter than the fused-localization path (170;0;255) it may lie on."""

WAYPOINT_LABEL_COLOR_RGBA = (0.9, 0.8, 1.0, 1.0)
"""Pale lavender: tied to the dots by hue, readable on the dark RViz ground."""


def _color(rgba) -> ColorRGBA:
    """ColorRGBA from a 4-tuple; the message constructor is keyword-only."""

    red, green, blue, alpha = rgba
    return ColorRGBA(r=red, g=green, b=blue, a=alpha)


def route_label_indices(
    points: Sequence["RoutePoint"],
    every_n: int = 0,
    center: tuple[float, float] | None = None,
    radius_m: float = 0.0,
) -> list[int]:
    """Which waypoint indices get a text label. Two disjoint modes.

    With a vehicle position (`center`) the labels are a window that follows the
    vehicle: every waypoint within `radius_m` of it and nothing outside, so
    `radius_m` <= 0 means no labels. This is the mapping view - the numbers
    worth reading are the ones next to the car, and a 500 m route's other ~380
    labels are unreadable clutter at any density.

    Without a centre - a viewer that has no vehicle pose, e.g. the reference
    route in localization mode - every n-th waypoint plus always the last, so
    the whole route stays sparsely annotated. `every_n` <= 0 then means no
    labels.
    """

    count = len(points)
    if count <= 0:
        return []
    if center is not None:
        if radius_m <= 0.0:
            return []
        # Planar distance: the car is on the ground with its waypoints, and a
        # sloped route would otherwise drop labels that are visually adjacent.
        squared_radius = radius_m * radius_m
        return [
            index
            for index, point in enumerate(points)
            if (point.x - center[0]) ** 2 + (point.y - center[1]) ** 2
            <= squared_radius
        ]
    if every_n <= 0:
        return []
    indices = list(range(0, count, every_n))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    return indices


@dataclass(frozen=True)
class RoutePoint:
    """One globally referenced route sample and its two corridor edges."""

    x: float
    y: float
    z: float
    yaw: float
    left_x: float
    left_y: float
    right_x: float
    right_y: float


def make_route_point(
    x: float, y: float, z: float, yaw: float, corridor_half_width: float
) -> RoutePoint:
    """Build the simple perpendicular-offset corridor used by this package."""

    width = float(corridor_half_width)
    return RoutePoint(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        left_x=x - width * math.sin(yaw),
        left_y=y + width * math.cos(yaw),
        right_x=x + width * math.sin(yaw),
        right_y=y - width * math.cos(yaw),
    )


def transform_route(
    route: Iterable[RoutePoint], transform
) -> list[RoutePoint]:
    """Apply one TransformStamped to an entire route using double precision."""

    q = transform.transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm == 0.0:
        qx = qy = qz = 0.0
        qw = 1.0
    else:
        qx, qy, qz, qw = q.x / norm, q.y / norm, q.z / norm, q.w / norm

    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    translation = transform.transform.translation

    def transform_xyz(x: float, y: float, z: float) -> tuple[float, float, float]:
        return (
            rotation[0][0] * x
            + rotation[0][1] * y
            + rotation[0][2] * z
            + translation.x,
            rotation[1][0] * x
            + rotation[1][1] * y
            + rotation[1][2] * z
            + translation.y,
            rotation[2][0] * x
            + rotation[2][1] * y
            + rotation[2][2] * z
            + translation.z,
        )

    transformed = []
    for point in route:
        x, y, z = transform_xyz(point.x, point.y, point.z)
        left_x, left_y, _ = transform_xyz(
            point.left_x, point.left_y, point.z
        )
        right_x, right_y, _ = transform_xyz(
            point.right_x, point.right_y, point.z
        )
        heading_x = rotation[0][0] * math.cos(point.yaw) + rotation[0][1] * math.sin(
            point.yaw
        )
        heading_y = rotation[1][0] * math.cos(point.yaw) + rotation[1][1] * math.sin(
            point.yaw
        )
        transformed.append(
            RoutePoint(
                x=x,
                y=y,
                z=z,
                yaw=math.atan2(heading_y, heading_x),
                left_x=left_x,
                left_y=left_y,
                right_x=right_x,
                right_y=right_y,
            )
        )
    return transformed


def build_route_marker_array(
    route: Iterable[RoutePoint],
    frame_id: str,
    stamp,
    namespace: str = "reference_route",
    dot_scale_m: float = 0.35,
    label_every_n: int = 10,
    label_text_size_m: float = 0.6,
    previous_label_count: int = 0,
    label_center: tuple[float, float] | None = None,
    label_radius_m: float = 0.0,
    highlight_index_ranges: Sequence[tuple[int, int]] = (),
    slow_down_index_ranges: Sequence[tuple[int, int]] = (),
    avoidance_index_ranges: Sequence[tuple[int, int]] = (),
    school_zone_index_ranges: Sequence[tuple[int, int]] = (),
) -> MarkerArray:
    """Create globally anchored route and corridor markers for RViz.

    Stable marker ids: 0 yellow centerline, 3 green filled drivable area,
    4 purple waypoint dots, 5 red filled drivable area inside
    `highlight_index_ranges` (inclusive waypoint-index pairs, e.g. traffic-light
    detection zones), 6 orange inside `slow_down_index_ranges`, 7 sky blue
    inside `avoidance_index_ranges`, 8 yellow inside `school_zone_index_ranges`,
    100+ text labels, and 1/2 always DELETE
    (the retired boundary polylines, kept reserved so viewers still holding
    them clear them). A segment inside ranges of more than one kind is painted
    once, in the first of red, orange, sky blue, yellow that claims it.

    `label_center`/`label_radius_m` (both in the marker frame) restrict the text
    labels to the waypoints around the vehicle; without a centre `label_every_n`
    thins them over the whole route instead. See route_label_indices().

    `previous_label_count` is how many labels the previous publication carried;
    the surplus is DELETEd so a shrinking label set - a moving window as much as
    a bag replayed from the start - does not leave orphan text floating in the
    view. Callers get the current count from route_label_indices().
    """

    points = list(route)
    result = MarkerArray()

    def stale_label_deletes(current_count: int) -> None:
        for offset in range(current_count, previous_label_count):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = FIRST_LABEL_MARKER_ID + offset
            marker.action = Marker.DELETE
            result.markers.append(marker)

    if not points:
        # Clear a route left by a previous bag pass/reset. DELETE markers do not
        # require geometry and therefore cannot trigger RViz point-count errors.
        for marker_id in (
            0, 1, 2, 3, WAYPOINT_DOT_MARKER_ID, *ZONE_FILL_MARKER_IDS
        ):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = namespace
            marker.id = marker_id
            marker.action = Marker.DELETE
            result.markers.append(marker)
        stale_label_deletes(0)
        return result

    # Keep Ogre vertices route-relative instead of passing UTM northings around 4e6
    # through single-precision floats. The marker pose retains the global anchor.
    anchor = points[0]

    def relative_point(x: float, y: float, z: float, z_offset: float) -> Point:
        return Point(
            x=x - anchor.x,
            y=y - anchor.y,
            z=z - anchor.z + z_offset,
        )

    def initialize_marker(marker_id: int, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        # Re-place this marker every render frame instead of once on arrival.
        # RViz keeps frame-locked markers in their own set and re-transforms
        # them in every update() (rviz_default_plugins MarkerCommon::
        # updateMarkersWithLockedFrame); everything else is positioned once,
        # when the message is processed. The route is world-fixed and published
        # rarely, so without this it would sit frozen wherever it last landed
        # and only jump on the next publication whenever the fixed frame itself
        # moves - i.e. in any vehicle-frame view. Costs one transform lookup per
        # marker per frame, against a handful of markers.
        marker.frame_locked = True
        marker.pose.position = Point(x=anchor.x, y=anchor.y, z=anchor.z)
        marker.pose.orientation.w = 1.0
        return marker

    def waypoint_dots() -> Marker:
        dots = initialize_marker(WAYPOINT_DOT_MARKER_ID, Marker.SPHERE_LIST)
        dots.scale.x = dots.scale.y = dots.scale.z = dot_scale_m
        dots.color = _color(WAYPOINT_DOT_COLOR_RGBA)
        dots.points = [relative_point(p.x, p.y, p.z, 0.08) for p in points]
        return dots

    def waypoint_labels() -> list[Marker]:
        """One TEXT_VIEW_FACING per labelled waypoint: "#12  4.3, -1.8".

        Text markers have no `points` array, so unlike the geometry above they
        cannot use the anchor-plus-relative trick and must carry an absolute
        position. That is fine here because the visualization route is kept in
        the small local map frame - the UTM copy goes to the CSV, never to RViz.
        """

        labels = []
        indices = route_label_indices(
            points, label_every_n, label_center, label_radius_m
        )
        for offset, index in enumerate(indices):
            point = points[index]
            label = initialize_marker(
                FIRST_LABEL_MARKER_ID + offset, Marker.TEXT_VIEW_FACING
            )
            label.pose.position = Point(
                x=point.x, y=point.y, z=point.z + 0.8
            )
            label.scale.z = label_text_size_m
            label.color = _color(WAYPOINT_LABEL_COLOR_RGBA)
            label.text = f"#{index}  {point.x:.1f}, {point.y:.1f}"
            labels.append(label)
        return labels

    labels = waypoint_labels()
    stale_label_deletes(len(labels))

    if len(points) == 1:
        # LINE_STRIP needs at least two points and TRIANGLE_LIST needs triples.
        # Show the first accepted waypoint immediately as a visible sphere, and
        # explicitly remove any corridor geometry from an earlier bag pass.
        start = initialize_marker(0, Marker.SPHERE)
        start.scale.x = start.scale.y = start.scale.z = 0.7
        start.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        result.markers.append(start)
        for marker_id in (1, 2, 3, *ZONE_FILL_MARKER_IDS):
            marker = initialize_marker(marker_id, Marker.LINE_STRIP)
            marker.action = Marker.DELETE
            result.markers.append(marker)
        result.markers.append(waypoint_dots())
        result.markers.extend(labels)
        return result

    centerline = initialize_marker(0, Marker.LINE_STRIP)
    centerline.scale.x = 0.15
    centerline.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
    centerline.points = [relative_point(p.x, p.y, p.z, 0.05) for p in points]

    # The drivable area is shown as a filled surface only: the centerline plus
    # the green fill below. Marker ids 1 and 2 used to carry the left/right
    # boundary polylines. They are still published, as DELETE, so an RViz that
    # holds them from an earlier build or an earlier pass of the same bag drops
    # them instead of drawing stale edges over the fill; reserving the ids is
    # what makes that cleanup reach it.
    stale_edges = []
    for marker_id in (1, 2):
        edge = initialize_marker(marker_id, Marker.LINE_STRIP)
        edge.action = Marker.DELETE
        stale_edges.append(edge)

    def segment_is_inside(index: int, ranges) -> bool:
        # Segment `index` joins waypoints index and index + 1 and counts as
        # inside a range only when both ends are, so a range (170, 180) paints
        # exactly the stretch from waypoint 170 to waypoint 180 - the same
        # stretch traffic_detect_zones.WaypointIndexRange.contains() covers.
        return any(first <= index and index + 1 <= last for first, last in ranges)

    def zone_fill(marker_id: int, rgba) -> Marker:
        surface = initialize_marker(marker_id, Marker.TRIANGLE_LIST)
        surface.scale.x = surface.scale.y = surface.scale.z = 1.0
        surface.color = _color(rgba)
        return surface

    fill = initialize_marker(3, Marker.TRIANGLE_LIST)
    fill.scale.x = fill.scale.y = fill.scale.z = 1.0
    fill.color = ColorRGBA(r=0.0, g=0.8, b=0.0, a=0.28)
    zone_fills = [
        (
            zone_fill(HIGHLIGHT_FILL_MARKER_ID, HIGHLIGHT_FILL_COLOR_RGBA),
            highlight_index_ranges,
        ),
        (
            zone_fill(SLOW_DOWN_FILL_MARKER_ID, SLOW_DOWN_FILL_COLOR_RGBA),
            slow_down_index_ranges,
        ),
        (
            zone_fill(AVOIDANCE_FILL_MARKER_ID, AVOIDANCE_FILL_COLOR_RGBA),
            avoidance_index_ranges,
        ),
        (
            zone_fill(SCHOOL_ZONE_FILL_MARKER_ID, SCHOOL_ZONE_FILL_COLOR_RGBA),
            school_zone_index_ranges,
        ),
    ]
    for index in range(len(points) - 1):
        current = points[index]
        following = points[index + 1]
        target = next(
            (
                surface
                for surface, ranges in zone_fills
                if segment_is_inside(index, ranges)
            ),
            fill,
        )
        current_left = relative_point(
            current.left_x, current.left_y, current.z, 0.02
        )
        current_right = relative_point(
            current.right_x, current.right_y, current.z, 0.02
        )
        following_left = relative_point(
            following.left_x, following.left_y, following.z, 0.02
        )
        following_right = relative_point(
            following.right_x, following.right_y, following.z, 0.02
        )
        target.points.extend(
            [
                current_left,
                current_right,
                following_left,
                current_right,
                following_right,
                following_left,
            ]
        )
    # A fill that received no segment (no zones, or a route that is one big
    # zone) is DELETEd, never ADDed empty: it also clears a stale fill a viewer
    # holds from the previous zone list.
    surfaces = [fill, *(surface for surface, _ in zone_fills)]
    for surface in surfaces:
        if not surface.points:
            surface.action = Marker.DELETE

    # Stable IDs make complete republishing replace the previous RViz route.
    result.markers.extend(
        [*surfaces, centerline, *stale_edges, waypoint_dots(), *labels]
    )
    return result
