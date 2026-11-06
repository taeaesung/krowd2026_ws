import math
import unittest

from sensor_msgs.msg import Imu, NavSatFix

from mola_slam_mapping.gnss_covariance_floor_node import floor_navsat_covariance
from mola_slam_mapping.imu_downsample_node import TimestampRateGate, stamp_nanoseconds


class SensorSanitizerTests(unittest.TestCase):
    def test_timestamp_rate_gate_is_timestamp_based_and_resets_on_rewind(self):
        gate = TimestampRateGate(10.0)
        self.assertTrue(gate.accept(1_000_000_000))
        self.assertFalse(gate.accept(1_099_999_999))
        self.assertTrue(gate.accept(1_100_000_000))
        self.assertTrue(gate.accept(500_000_000))

    def test_stamp_nanoseconds(self):
        msg = Imu()
        msg.header.stamp.sec = 12
        msg.header.stamp.nanosec = 345
        self.assertEqual(stamp_nanoseconds(msg), 12_000_000_345)

    def test_covariance_floor_preserves_measurement_and_large_variances(self):
        msg = NavSatFix()
        msg.header.frame_id = "gps_link"
        msg.latitude = 37.5
        msg.longitude = 127.0
        msg.altitude = 42.0
        msg.position_covariance = [
            1e-8, 0.1, 0.0, 0.1, 4.0, 0.0, 0.0, 0.0, math.nan
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN

        result = floor_navsat_covariance(msg, 0.05, 1.0)

        self.assertIsNot(result, msg)
        self.assertEqual(
            (result.latitude, result.longitude, result.altitude),
            (37.5, 127.0, 42.0),
        )
        self.assertEqual(result.header.frame_id, "gps_link")
        self.assertAlmostEqual(result.position_covariance[0], 0.05**2)
        self.assertEqual(result.position_covariance[4], 4.0)
        self.assertEqual(result.position_covariance[8], 1.0)
        self.assertEqual(result.position_covariance[1], 0.1)
        self.assertEqual(msg.position_covariance[0], 1e-8)

    def test_unknown_covariance_becomes_diagonal_known(self):
        msg = NavSatFix()
        msg.position_covariance = [9.0] * 9
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        result = floor_navsat_covariance(msg, 0.2, 1.5)

        self.assertEqual(
            result.position_covariance_type,
            NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN,
        )
        expected = [0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 2.25]
        for actual, wanted in zip(result.position_covariance, expected):
            self.assertAlmostEqual(actual, wanted)


if __name__ == "__main__":
    unittest.main()
