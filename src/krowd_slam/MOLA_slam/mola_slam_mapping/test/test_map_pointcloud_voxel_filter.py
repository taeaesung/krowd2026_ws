import struct

from sensor_msgs.msg import PointCloud2, PointField

from mola_slam_mapping.map_pointcloud_voxel_filter_node import (
    republish_is_due,
    voxel_filter_cloud,
)


def test_a_publish_once_cloud_is_repeated_but_a_live_one_is_not():
    """Changing RViz's Fixed Frame resets displays without resubscribing.

    A transient-local cloud sent once - the loaded map in localization - would
    be wiped for good, so it is repeated when it goes stale. While mapping, MOLA
    keeps sending updates and the repeat must stay out of the way.
    """

    interval = 2_000_000_000

    assert republish_is_due(0, interval, interval)
    assert republish_is_due(0, interval * 5, interval)
    # Something was published recently: leave it alone.
    assert not republish_is_due(0, interval - 1, interval)
    # Nothing has ever been published: there is nothing to repeat.
    assert not republish_is_due(None, interval * 5, interval)
    # Disabled.
    assert not republish_is_due(0, interval * 5, 0)


def _cloud(points):
    message = PointCloud2()
    message.height = 1
    message.width = len(points)
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.point_step = 16
    message.row_step = message.width * message.point_step
    message.data = b"".join(struct.pack("<ffff", *point) for point in points)
    message.is_dense = True
    return message


def test_voxel_filter_preserves_first_complete_point_record():
    source = _cloud(
        [
            (0.01, 0.01, 0.01, 10.0),
            (0.20, 0.20, 0.20, 20.0),
            (0.30, 0.01, 0.01, 30.0),
        ]
    )

    filtered = voxel_filter_cloud(source, 0.25)

    assert source.width == 3
    assert filtered.width == 2
    assert filtered.height == 1
    assert filtered.row_step == 32
    assert struct.unpack_from("<ffff", filtered.data, 0) == (
        0.009999999776482582,
        0.009999999776482582,
        0.009999999776482582,
        10.0,
    )
    assert struct.unpack_from("<f", filtered.data, 16 + 12)[0] == 30.0
