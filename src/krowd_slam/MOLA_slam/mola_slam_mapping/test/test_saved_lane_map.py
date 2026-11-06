import pytest
from visualization_msgs.msg import Marker

from mola_slam_mapping.saved_lane_map_node import lane_map_markers, load_lane_map

HEADER = "segment_id,point_index,stamp_ns,lane_class,namespace,quality,map_x,map_y,map_z\n"


def write(tmp_path, body, header=HEADER):
    path = tmp_path / "lane_map.csv"
    path.write_text(header + body)
    return path


def test_segments_keep_their_id_class_and_point_order(tmp_path):
    path = write(
        tmp_path,
        "1,1,0,stopLine,stopline_ransac,0.5,5.0,1.0,0.1\n"
        "0,1,0,leftSolid,leftBoundary_continuous_inferred,0.4,1.0,0.0,0.1\n"
        "0,0,0,leftSolid,leftBoundary_continuous_inferred,0.4,0.0,0.0,0.1\n"
        "1,0,0,stopLine,stopline_ransac,0.5,5.0,0.0,0.1\n",
    )
    segments = load_lane_map(path)
    assert [s.segment_id for s in segments] == [0, 1]
    assert segments[0].lane_class == "leftSolid"
    assert segments[0].points[:, 0].tolist() == [0.0, 1.0]
    assert segments[1].points[:, 1].tolist() == [0.0, 1.0]

    markers = lane_map_markers(segments, "map", 0.18, 0.35).markers
    assert markers[0].action == Marker.DELETEALL
    lane, stop = markers[1:]
    assert lane.ns == "leftBoundary_continuous_inferred"
    assert (lane.id, lane.type) == (0, Marker.LINE_STRIP)
    assert (stop.ns, stop.id) == ("stopline_ransac", 1)
    assert lane.scale.x == pytest.approx(0.18)
    assert stop.scale.x == pytest.approx(0.35)
    assert all(m.header.frame_id == "map" for m in markers)
    assert [p.x for p in lane.points] == [0.0, 1.0]


@pytest.mark.parametrize(
    "header,body",
    [
        ("segment_id,point_index,lane_class,map_x,map_y,map_z\n", "0,0,leftSolid,0,0,0\n"),
        (HEADER, ""),
        (HEADER, "0,0,0,leftSolid,a,0,0,0,0\n"),
        (HEADER, "0,0,0,leftSolid,a,0,0,0,0\n0,0,0,leftSolid,a,0,1,0,0\n"),
        (HEADER, "0,0,0,leftSolid,a,0,0,0,0\n0,1,0,rightSolid,b,0,1,0,0\n"),
        (HEADER, "0,0,0,leftSolid,a,0,x,0,0\n0,1,0,leftSolid,a,0,1,0,0\n"),
    ],
)
def test_an_unusable_lane_map_is_rejected(tmp_path, header, body):
    with pytest.raises(ValueError):
        load_lane_map(write(tmp_path, body, header))
