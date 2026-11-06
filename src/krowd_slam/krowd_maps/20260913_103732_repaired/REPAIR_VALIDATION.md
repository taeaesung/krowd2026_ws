# 20260913_103732 map repair (2026-09-22)

Source run: `../20260913_103732`; source bag: `/media/krowd/T7/rosbags/0913_1`.
The source run and its files were not overwritten.

## Cause and artifacts

The MOLA odometry lost approximately 20 m during the first 11.3 s. Its first
two corrected keyframes were both placed about 20.5 m away from the GNSS/IMU
reference. The final keyframe was about 6.07 m away. The intervening 535
keyframes agreed within 0.5 m. The old MOLA-derived PLY did not provide a
reliable scan-to-map match at the stationary start.

`keyframes_repaired.tum` moves only keyframes 0, 1 and 537 to their
time-matched reference *positions*. Their rotations and all other keyframes
are unchanged. See `keyframe_repair.yaml`. `map_repaired.simplemap`,
`global_map_repaired.mm`, its sidecar georeference, and
`global_map_repaired_localmap.ply` were generated from those poses.

`gnss_reference_localmap.ply` instead uses 1,451 original bag LiDAR scans,
the bag's static LiDAR-to-base TF, the GNSS/IMU reference trajectory, and
10 ms motion-compensation bins. It has 662,691 finite points (0.20 m voxels).
`localization.launch.py` prefers this PLY for the one-shot stationary LiDAR
heading initializer. The MM is used by MOLA localization separately.

## Verification

| Quantity | Original | Repaired |
| --- | ---: | ---: |
| Keyframe-reference position error, max | 20.52 m | 0.456 m |
| Keyframes with position error >0.5 m | 3/538 | 0/538 |
| Keyframe-reference position error, p95 | 0.177 m | 0.160 m |

In isolated ROS_DOMAIN_ID 212, playing the stationary start of `0913_1` with
the GNSS-reference PLY and a route-like yaw prior of 0 +/- 15 degrees led to
`/map_localization/heading_verified=True` by scan stamp 1789258106.876,
about 0.3 s after the first matched stationary scan. Position was within
0.14 m of the stationary reference, scan fitness about 0.05, and reported yaw
about +2.7 degrees versus the reference's 0 degrees. This is an offline
subsystem test with an explicitly supplied yaw prior, not a full launch or
real-vehicle validation.

With the original `global_map_corrected_localmap.ply` under the same replay,
heading did **not** verify; only one scan matched, at an implausible +55.4
degree yaw and roughly (1.82, -4.35) m.

The localizer was also changed to retain a consistent accepted-scan count
across occasional failed ICP scans, provided accepted scans stay within 0.75 s.
Long gaps still reset the count. This change is compiled in
`krowd_lidar_odometry`; it takes effect when that node is restarted.

## Use

```bash
ros2 launch mola_slam_mapping localization.launch.py \
  map_dir:=/home/krowd/krowd_maps/20260913_103732_repaired \
  map_filename:=global_map_repaired.mm
```

Do not start autonomous driving from the old map directory. Before driving,
verify live RTK Fixed, map-localizer `heading_verified=True`, ESKF
`heading_verified=True`, and agreement of MOLA, ESKF and route positions.
The source mapping run's missing LiDAR keyframes cannot be recreated inside
the MOLA simplemap; the GNSS-reference PLY fills the scan-to-map initializer
from the original raw scans. Runtime MOLA map localization and cross-date
repeatability still need a live check.
