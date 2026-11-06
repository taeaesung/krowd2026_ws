import pytest
import yaml

from mola_slam_mapping.route_visualization import make_route_point
from mola_slam_mapping.route_io import project_onto_route
from mola_slam_mapping.traffic_detect_zones import (
    AVOIDANCE,
    NODE_NAME,
    SLOW_DOWN,
    TRAFFIC_DETECT,
    ZONE_FILENAME,
    ZONE_KINDS,
    WaypointIndexRange,
    check_ranges_fit_route,
    parse_index_ranges,
    read_zone_specs,
    template_may_be_rewritten,
    zone_containing,
    zone_template_yaml,
)


def test_ranges_parse_from_first_dash_last_strings():
    assert parse_index_ranges(["170-180", " 250 - 260 "]) == [
        WaypointIndexRange(170, 180),
        WaypointIndexRange(250, 260),
    ]
    assert parse_index_ranges([]) == []
    assert WaypointIndexRange(170, 180).label == "#170-180"


@pytest.mark.parametrize(
    "spec", ["170", "170:180", "a-b", "180-170", "-5-10", "", "170-180-190"]
)
def test_malformed_or_reversed_ranges_are_refused(spec):
    """A typo must stop the launch, not silently drop a traffic-light zone."""

    with pytest.raises(ValueError, match="traffic-detect zone"):
        parse_index_ranges([spec])


def test_a_range_past_the_last_waypoint_is_refused():
    zones = parse_index_ranges(["170-180"])
    check_ranges_fit_route(zones, waypoint_count=181)  # last waypoint is #180
    with pytest.raises(ValueError, match="is the zone list for this map"):
        check_ranges_fit_route(zones, waypoint_count=180)


def test_zone_membership_covers_exactly_waypoint_first_to_last():
    zones = parse_index_ranges(["170-180", "250-260"])

    assert zone_containing(zones, 169.99) is None
    assert zone_containing(zones, 170.0) == zones[0]
    assert zone_containing(zones, 175.5) == zones[0]
    assert zone_containing(zones, 180.0) == zones[0]
    assert zone_containing(zones, 180.01) is None
    assert zone_containing(zones, 255.0) == zones[1]


def test_projection_route_position_is_the_waypoint_index_coordinate():
    """The Bool and the red corridor must agree on where a zone starts."""

    route = [make_route_point(float(i) * 2.0, 0.0, 0.0, 0.0, 1.5) for i in range(10)]
    zones = parse_index_ranges(["3-5"])

    # x = 6.0 is waypoint 3 exactly; x = 7.0 is halfway along segment 3.
    assert project_onto_route(route, 6.0, 0.0).route_position == pytest.approx(3.0)
    assert project_onto_route(route, 7.0, 0.4).route_position == pytest.approx(3.5)
    assert zone_containing(zones, project_onto_route(route, 5.9, 0.0).route_position) is None
    assert zone_containing(zones, project_onto_route(route, 6.0, 0.0).route_position) is not None
    assert zone_containing(zones, project_onto_route(route, 10.0, 0.0).route_position) is not None
    assert zone_containing(zones, project_onto_route(route, 10.1, 0.0).route_position) is None
    # Beyond the route's end the projection clamps to the last waypoint.
    assert project_onto_route(route, 50.0, 0.0).route_position == pytest.approx(9.0)


def test_template_records_this_map_identity_and_index_range(tmp_path):
    """The operator picks numbers from this header, so it must name the map."""

    text = zone_template_yaml("20260907_114123", waypoint_count=381)
    assert "20260907_114123" in text
    assert "#0 to #380" in text and "(381 of them" in text

    path = tmp_path / "traffic_detect_zones.yaml"
    path.write_text(text)
    document = yaml.safe_load(text)
    assert document[NODE_NAME]["ros__parameters"]["traffic_detect_zones"] == []
    assert read_zone_specs(path) == []
    # A filled-in template round-trips through the parser it feeds.
    filled = zone_template_yaml("m", 400, ["170-180", "250-260"])
    path.write_text(filled)
    assert read_zone_specs(path) == ["170-180", "250-260"]
    assert parse_index_ranges(read_zone_specs(path)) == [
        WaypointIndexRange(170, 180),
        WaypointIndexRange(250, 260),
    ]


def test_a_filled_in_template_is_never_overwritten(tmp_path):
    """Re-running the saver on a directory must not discard chosen zones."""

    path = tmp_path / "traffic_detect_zones.yaml"
    assert template_may_be_rewritten(path)  # absent

    path.write_text(zone_template_yaml("m", 100))
    assert template_may_be_rewritten(path)  # present but still empty

    path.write_text(zone_template_yaml("m", 100, ["10-20"]))
    assert not template_may_be_rewritten(path)

    # A half-finished edit is left alone rather than treated as empty.
    path.write_text("localization_monitor_node: [unclosed\n")
    assert not template_may_be_rewritten(path)


def test_missing_or_empty_zone_documents_read_as_no_zones(tmp_path):
    path = tmp_path / "zones.yaml"
    path.write_text("")
    assert read_zone_specs(path) == []
    path.write_text("localization_monitor_node:\n  ros__parameters:\n")
    assert read_zone_specs(path) == []
    path.write_text("some_other_node:\n  ros__parameters:\n    x: 1\n")
    assert read_zone_specs(path) == []


def test_each_zone_kind_has_its_own_file_parameter_and_topic():
    assert [kind.filename for kind in ZONE_KINDS] == [
        "traffic_detect_zones.yaml",
        "slow_down_zones.yaml",
        "avoidance_zones.yaml",
        "school_zones.yaml",
    ]
    assert [kind.topic for kind in ZONE_KINDS] == [
        "traffic_detect_space",
        "slow_down_space",
        "avoidance_space",
        "school_zone_space",
    ]
    assert [kind.color for kind in ZONE_KINDS] == ["red", "orange", "sky blue", "yellow"]
    assert ZONE_FILENAME == TRAFFIC_DETECT.filename


@pytest.mark.parametrize("kind", [SLOW_DOWN, AVOIDANCE])
def test_other_kinds_template_and_read_back_under_their_own_parameter(
    tmp_path, kind
):
    text = zone_template_yaml("20260913_103732", 586, ["51-64"], kind=kind)
    assert kind.title in text and "#0 to #585" in text
    assert f"drawn {kind.color}" in text and f"/{kind.topic}" in text

    path = tmp_path / kind.filename
    path.write_text(text)
    document = yaml.safe_load(text)
    assert document[NODE_NAME]["ros__parameters"] == {kind.parameter: ["51-64"]}
    assert read_zone_specs(path, kind) == ["51-64"]
    # The traffic-light parameter is absent from this file.
    assert read_zone_specs(path) == []


@pytest.mark.parametrize("kind", [SLOW_DOWN, AVOIDANCE])
def test_errors_name_the_zone_kind(kind):
    with pytest.raises(ValueError, match=f"{kind.name} zone '5'"):
        parse_index_ranges(["5"], kind)
    with pytest.raises(ValueError, match=f"{kind.name} zone #10-20 reaches past"):
        check_ranges_fit_route(parse_index_ranges(["10-20"], kind), 20, kind)


def test_a_filled_zone_file_protects_only_itself(tmp_path):
    """Filling one kind in must not freeze - or unfreeze - another."""

    filled = tmp_path / SLOW_DOWN.filename
    filled.write_text(zone_template_yaml("m", 100, ["10-20"], kind=SLOW_DOWN))
    assert not template_may_be_rewritten(filled, SLOW_DOWN)

    empty = tmp_path / AVOIDANCE.filename
    empty.write_text(zone_template_yaml("m", 100, kind=AVOIDANCE))
    assert template_may_be_rewritten(empty, AVOIDANCE)
