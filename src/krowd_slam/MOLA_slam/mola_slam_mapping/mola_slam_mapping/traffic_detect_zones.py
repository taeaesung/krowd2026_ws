"""Route stretches, addressed by waypoint index, where a mission applies:
traffic-light detection, slowing down, obstacle avoidance, child-protection
(school) zones.

A zone is an inclusive [first, last] range over the `idx` column of the map's
waypoints.csv - the same numbers the "#idx" RViz labels show. Waypoints are
recorded ~1 m apart, so "170-180" is roughly 10 m of track. The indices belong
to ONE map: re-mapping renumbers them.

The vehicle's position along the route is a float, segment index plus the
fraction along that segment (see route_io.RouteProjection), so a zone
[170, 180] is the stretch from waypoint 170 to waypoint 180 exactly - the same
stretch route_visualization paints in the zone's colour.

Each kind of zone has its own per-map file, parameter and std_msgs/Bool topic
(ZONE_KINDS), all in the same format, so a kind can be filled in, emptied or
copied between maps without touching the others.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


@dataclass(frozen=True)
class ZoneKind:
    """One kind of zone: a per-map file, a node parameter and a Bool topic."""

    name: str
    """Short name used in log lines and error messages, e.g. "slow-down"."""
    title: str
    """Heading of the per-map file, e.g. "Slow-down zones"."""
    parameter: str
    """localization_monitor_node parameter holding the ranges. The per-map
    file is `<parameter>.yaml` and the launch argument `<parameter>_yaml`."""
    topic: str
    """std_msgs/Bool topic, true while the vehicle is inside one of the zones."""
    color: str
    """How RViz paints the zones (route_visualization), named for the file
    header so the operator knows what to look for."""

    @property
    def filename(self) -> str:
        return f"{self.parameter}.yaml"


TRAFFIC_DETECT = ZoneKind(
    name="traffic-detect",
    title="Traffic-light detection zones",
    parameter="traffic_detect_zones",
    topic="traffic_detect_space",
    color="red",
)
SLOW_DOWN = ZoneKind(
    name="slow-down",
    title="Slow-down zones",
    parameter="slow_down_zones",
    topic="slow_down_space",
    color="orange",
)
AVOIDANCE = ZoneKind(
    name="avoidance",
    title="Obstacle-avoidance zones",
    parameter="avoidance_zones",
    topic="avoidance_space",
    color="sky blue",
)
SCHOOL_ZONE = ZoneKind(
    name="school-zone",
    title="Child-protection (school) zones",
    parameter="school_zones",
    topic="school_zone_space",
    color="yellow",
)
ZONE_KINDS = (TRAFFIC_DETECT, SLOW_DOWN, AVOIDANCE, SCHOOL_ZONE)

ZONE_FILENAME = TRAFFIC_DETECT.filename
"""Per-map zone file, written next to waypoints.csv by the mapping run and
picked up automatically by localization.launch.py. The indices only mean
something for the route in that same folder, which is why the file lives with
the map rather than with the code. The other kinds follow the same rule under
their own ZoneKind.filename."""

NODE_NAME = "localization_monitor_node"
"""ROS params files are keyed by node name; this is the node that reads them."""


@dataclass(frozen=True)
class WaypointIndexRange:
    first: int
    last: int

    @property
    def label(self) -> str:
        return f"#{self.first}-{self.last}"

    def contains(self, route_position: float) -> bool:
        return self.first <= route_position <= self.last


def parse_index_ranges(
    specs: Iterable[str], kind: ZoneKind = TRAFFIC_DETECT
) -> list[WaypointIndexRange]:
    """"170-180" -> WaypointIndexRange(170, 180).

    Strict on purpose: a malformed or reversed range raises ValueError so a typo
    in the zone list stops the launch instead of silently leaving one stretch
    without its mission.
    """

    ranges = []
    for spec in specs:
        text = str(spec).strip()
        first_text, separator, last_text = text.partition("-")
        try:
            if not separator:
                raise ValueError
            first, last = int(first_text), int(last_text)
        except ValueError:
            raise ValueError(
                f"{kind.name} zone '{spec}' is not of the form "
                "'<first idx>-<last idx>', e.g. '170-180'"
            ) from None
        if first < 0 or last < first:
            raise ValueError(
                f"{kind.name} zone '{spec}': indices must satisfy "
                "0 <= first <= last"
            )
        ranges.append(WaypointIndexRange(first, last))
    return ranges


def check_ranges_fit_route(
    ranges: Sequence[WaypointIndexRange],
    waypoint_count: int,
    kind: ZoneKind = TRAFFIC_DETECT,
) -> None:
    """ValueError when a zone reaches past the last waypoint.

    That is almost always a zone list written for a different map, and the
    launch must refuse rather than drop the zone.
    """

    for zone in ranges:
        if zone.last > waypoint_count - 1:
            raise ValueError(
                f"{kind.name} zone {zone.label} reaches past the last "
                f"waypoint (#{waypoint_count - 1}) - is the zone list for this "
                "map?"
            )


def zone_containing(
    ranges: Sequence[WaypointIndexRange], route_position: float
) -> WaypointIndexRange | None:
    """The first zone the route position lies in, or None."""

    for zone in ranges:
        if zone.contains(route_position):
            return zone
    return None


def zone_template_yaml(
    run_id: str,
    waypoint_count: int,
    specs: Sequence[str] = (),
    kind: ZoneKind = TRAFFIC_DETECT,
) -> str:
    """The per-map zone file's text, with this map's index range in the header.

    Hand-written rather than yaml.safe_dump()'d because the comments are the
    point: the operator opens this file to choose numbers and needs to know
    which numbers exist in this particular map.
    """

    listed = ", ".join(f'"{spec}"' for spec in specs)
    return (
        f"# {kind.title} for THIS map.\n"
        f"# Map: {run_id} - waypoints #0 to #{waypoint_count - 1} "
        f"({waypoint_count} of them, ~1 m apart).\n"
        "# The numbers are waypoints.csv's `idx` column, i.e. the \"#170\"\n"
        "# labels RViz draws next to the purple waypoint dots.\n"
        "#\n"
        '# Each entry is an inclusive "first-last" range, e.g. ["170-180",\n'
        f'# "250-260"]. Inside one, the route corridor is drawn {kind.color} and\n'
        f"# localization_monitor_node publishes true on /{kind.topic}.\n"
        "#\n"
        "# This file belongs to this map alone - another mapping run renumbers\n"
        "# the waypoints. localization.launch.py reads it automatically when\n"
        "# localizing against this folder; no launch argument needed.\n"
        f"{NODE_NAME}:\n"
        "  ros__parameters:\n"
        f"    {kind.parameter}: [{listed}]\n"
    )


def read_zone_specs(path: Path, kind: ZoneKind = TRAFFIC_DETECT) -> list[str]:
    """The zone strings a params file lists; [] when it lists none.

    Propagates OSError/yaml.YAMLError - a caller deciding whether to overwrite
    the file must treat an unreadable one as "leave it alone", not as empty.
    """

    with open(path) as stream:
        document = yaml.safe_load(stream) or {}
    parameters = (document.get(NODE_NAME) or {}).get("ros__parameters") or {}
    return [str(spec) for spec in (parameters.get(kind.parameter) or [])]


def template_may_be_rewritten(path: Path, kind: ZoneKind = TRAFFIC_DETECT) -> bool:
    """True when rewriting the template would not discard chosen zones.

    Absent, or present and still empty. An unreadable file is left alone: it
    may be a half-finished edit, and losing an operator's zone list is worse
    than leaving a stale waypoint count in a comment.
    """

    if not path.exists():
        return True
    try:
        return not read_zone_specs(path, kind)
    except (OSError, yaml.YAMLError):
        return False
