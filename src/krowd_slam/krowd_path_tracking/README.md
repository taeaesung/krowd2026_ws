# krowd_path_tracking

Pure Pursuit (default) or Stanley path tracking against a route recorded by
`mola_slam_mapping`.

```
waypoints.csv (UTM)                    /gnss_imu/pose (Odometry, map)
        |                                        |
  route_publisher_node                   vehicle_state_node
        |                                        |
  /planning/trajectory                /localization/vehicle_state
  (krowd_control_interfaces/Trajectory) (krowd_control_interfaces/VehicleState)
        \                                       /
         \______________ pure_pursuit_node _____/
                                |
              /target_angle  /target_speed  /brake   -> ros_can_bridge
```

## Why Pure Pursuit rather than Stanley

Stanley's first term is the heading error at **unity gain**. This vehicle's
heading comes from the RTK velocity course in `gnss_imu_pose_node`, which is
gated at 0.5 m/s and carries roughly 9° of error near that gate — and 9° of
steering error is a real 8.5 m radius turn, whatever the steering limit is set
to.

Pure Pursuit takes the same heading error through the lookahead geometry, which
attenuates it by about `2L/ld` (0.45x at a 6 m lookahead), and it never reads
the route's own yaw, so the recorded drive's yaw noise cannot reach the steering
command at all. It is also defined at a standstill, where Stanley's
`atan(k·e/v)` is not.

Revisit this if heading is ever propagated with the IMU yaw rate between course
samples (proposed but unimplemented — see `AI_CONTEXT.md`). Stanley tracks more
tightly at low speed once its heading input is trustworthy.

## Nodes

| Node | In | Out |
|---|---|---|
| `route_publisher_node` | `waypoints.csv`, `corridor.csv`, TF `map<-utm` | `/planning/trajectory` |
| `vehicle_state_node` | `/gnss_imu/pose` | `/localization/vehicle_state` |
| `pure_pursuit_node` | both of the above | `/target_angle`, `/target_speed`, `/brake` |
| `stanley_node` | both of the above | the same three — one controller at a time (`controller_type`) |

`route_publisher_node` is the Planning side of the `krowd_control_interfaces`
contract, which previously had no implementation. It resolves the UTM route into
the localization frame, computes curvature, builds a speed profile from it, and
republishes every 0.5 s so `valid_until` stays ahead of now — giving the
controller a positive liveness signal rather than a plan nobody maintains.

## Running

Runs alongside a localization stack that is already publishing a pose; this
launch does not start one. It needs two things from it: the static TF
`map <- utm` and `/gnss_imu/pose`. Both come from `gnss_imu_pose_node`, which
needs both GNSS topics and about 5 m of straight driving to commit a heading —
so neither appears at t=0, and `route_publisher_node` says so while it waits.

```bash
# 1. localization, against the run that holds the route
ros2 launch mola_slam_mapping localization.launch.py \
    map_dir:=$HOME/krowd_maps/20260907_114123

# 2a. path data/state only (default): safe alongside stop_mission.
#     No map_dir: it follows the run localization loaded.
ros2 launch krowd_path_tracking path_tracking.launch.py

# 2b. normal route driving: enable the actuator controller explicitly
ros2 launch krowd_path_tracking path_tracking.launch.py controller:=true
```

**The controller defaults to disabled.** With `controller:=false` (or with the
argument omitted), the launch publishes the trajectory and vehicle state without
publishing `/target_angle`, `/target_speed`, or `/brake`. Use
`controller:=true` only for normal route driving after confirming no mission
controller is active.

**The map is named once, in localization.** It announces the run it loaded on
`/localization/map_dir` (latched), and `route_publisher_node` takes its route
from there, so the two cannot end up on different runs. `map_dir:=<run>` or
`waypoints_csv:=<file>` still override it - name the run itself rather than
`~/krowd_maps/latest`, which `mapping.launch.py` repoints at whatever run it is
starting, including one that then produces nothing (as of 2026-09-10 it pointed
at an empty `20260910_085043` with no `waypoints.csv` at all).

Check a new map with the default controller-disabled mode — trajectory shape,
speed profile, dropped-waypoint warnings — before letting it drive.

For a bag replay add `use_sim_time:=true` to both launches and replay only the
sensor topics (never the recorded `/tf` or `/tf_static`):

```bash
ros2 bag play ~/testweek_0905_4 --clock --rate 0.5 --disable-keyboard-controls \
    --topics /ouster/points /imu/data_ekf \
             /ublox_gps_node/fix /ublox_gps_node/fix_velocity
```

Localization against these saved maps is a known-open problem (ICP does not
converge; see `AI_CONTEXT.md`), but path tracking does not depend on it: the
pose and the geo-reference both come from RTK, not from ICP.

Nothing else may publish `/target_angle`, `/target_speed` or `/brake` while
`pure_pursuit_node` runs. That includes `stop_mission`, which has its own
independent Pure Pursuit for the cone course.

## Stanley

`stanley_node` has the same inputs, outputs and stop behaviour as
`pure_pursuit_node`; the launch starts one or the other.

```bash
ros2 launch krowd_path_tracking path_tracking.launch.py \
    map_dir:=$HOME/krowd_maps/<run> controller:=true controller_type:=stanley
```

```
delta = heading_gain * (route_yaw - vehicle_yaw) + atan(k * e / (k_soft + v))
```

- `e` is the cross-track error at the **front axle** (`wheelbase_m` ahead of the
  rear axle), positive when the vehicle is left of the route.
- `route_yaw` is taken from the route's **geometry** — the chord over
  `heading_window_m` of arc — not from the recorded waypoint yaw, so the mapping
  drive's heading noise does not reach the steering.
- The section above still applies to the *vehicle's* heading: it reaches the
  steering at `heading_gain` (1.0) instead of being attenuated by a lookahead.
- **The heading gate is off** (`max_heading_error_deg: 0.0` since 2026-09-17):
  the vehicle steers onto the route from whatever direction it starts in. Set
  it to 60.0 to hold `STOP HEADING_NOT_ALIGNED` with the brake on while
  `|route_yaw - vehicle_yaw|` exceeds it — which also catches a heading that is
  simply wrong (stale after turning while stationary, flipped by reversing).
- `steering_sign` must be exactly ±1.0 here; the node refuses to start otherwise.

Watch `/stanley_node/cross_track_error` and `/stanley_node/heading_error` while
tuning; the symptom table at the top of `config/path_tracking.yaml` has a Stanley
block.

## Before the first real run

Three numbers in `config/vehicle.yaml` are assumptions until measured, and each
is wrong in a way that is hard to see from the outside:

- **`max_steering_angle_deg`** defaults to `0.0`, meaning "derive from
  `min_turning_radius_m`" - 27.5° from the 5.15 m turning circle measured by
  hand on 2026-09-19. It is a **road-wheel** angle. What goes out on
  `/target_angle` is not: the steer Teensy takes a column-side angle, and the
  measured ratio between the two is **`steering_command_gain` = 8.0**
  (0912_2 drive, encoder against the road-wheel angle implied by IMU yaw rate
  and RTK speed). The Teensy also rejects commands beyond its own limit, ±2.5
  rad (143°) since 2026-09-20 - it was 1.7453 rad (100°) - and
  `max_steering_command_deg` 142 stays just under it, so the road-wheel limit
  actually applied is 142° / 8 = 17.8°, a 4.2 m turning radius (it was 12.4°
  and 6.1 m); the node logs the reduction. Corners tighter than 4.2 m still
  cannot be followed. `steering_command_offset_rad` is the command
  that centres the wheels; the Teensy zero is set per session, so read it off
  `/encoder` on a straight.
- **`rear_axle_offset_m`** defaults to `0.0`, asserting that `base_link` *is*
  the rear axle. Pure Pursuit's geometry is derived there; an unnoticed offset
  biases every command.
`steering_sign` is a sign, not a gain: only +1.0 or -1.0 are valid, in both
controllers. Until 2026-09-19 pure_pursuit_node carried 2.0 here, a steering
ratio in disguise that gave the road wheel a quarter of the angle asked for.

`max_lateral_accel_mps2` (3.0, about 0.31 g) is the tuning parameter that most
wants evidence. Raise it from measurement, not from hope.

## Tuning

`config/path_tracking.yaml` opens with a symptom-to-parameter table. Two things
about it are not obvious and cost time otherwise:

**Route ends.** The configured `goal_mode` is `FOLLOW_UNTIL_REPLACED`: there
is no goal. From wherever the vehicle starts it tracks the nearest part of the
route; at the route's end the speed ramps to 0 and the controller holds
`STOP END_OF_ROUTE` without latching. `route_publisher_node` can also close a
route whose end returns near its start into a loop driven round until stopped
(`loop_closure_max_gap_m`, off by default): the join is found where the
recorded drive's passes meet, but on a drive that stopped short of its start
it is a straight cut with a kink at the start point, so enable it only for a
map recorded by driving past the start.

**Starting from a standstill.** At rest there is no vehicle state until
`gnss_imu_pose_node` has an initial heading. It looks for one in order and
reports where it stands on `/gnss_imu/heading_acquisition` (`WAITING_GNSS`
while there is no GNSS fix at all - nothing below can start without one):

1. `WAITING_ESKF` (up to `eskf_wait_sec`, 3 s after the first GNSS fix): a
   heading the sensor-stack ESKF has already verified from the RTK course
   (`heading_source` `rtk_course` or `lidar_map`; the unverified
   `saved_prior` never counts). Driving to the start line after all-sensor
   start verifies it, so the pose appears at once, at a standstill.
2. `WAITING_LIDAR` (up to `lidar_seed_wait_sec`, 10 s): the verified LiDAR map
   alignment.
3. `NEEDS_MOTION`: neither came. Only now do both controllers *creep*:
   steering 0 at `heading_creep_speed_kmph` (5) for at most
   `heading_creep_max_sec` (12 s, ~17 m), status `CREEP HEADING_ACQUISITION`,
   until ~5 m of straight RTK course commits the heading.

Until `NEEDS_MOTION` the controllers WAIT and command nothing, so the vehicle
never drives blind while a standstill heading is still coming. The first
vehicle state hands over to tracking through the start ramp; a creep timeout
brakes with `STOP HEADING_NOT_ACQUIRED` (restart path tracking to retry). The
way ahead must be clear for a creep; `heading_creep_speed_kmph: 0` disables it
(wait, command nothing), `heading_creep_requires_request: false` creeps
without waiting for `NEEDS_MOTION`. Whatever seeded the heading, the RTK course
filters it once moving and the first straight 5 m segment replaces it; the
log names the seed and how far the two disagreed (warning over 20 deg).

**Stopping at the end, and starting again.** With `goal_mode: STOP_AT_END`
both controllers stop within `goal_tolerance_m` of the last waypoint and stay
stopped while the same trajectory keeps being republished. They resume on their
own once the vehicle is back at the route start: within `restart_near_start_m`
(10 m) of it along the route, inside the corridor, and more than
2 x `goal_tolerance_m` from the end point - so a loop whose ends nearly touch
does not restart where it stopped. The window has to exceed the ~5 m of driving
before the RTK pose reappears on a restarted drive. `restart_near_start_m: 0`
keeps it stopped until path tracking restarts.

**Starting.** The route's speed profile has no ramp from standstill, so
both controllers cap the command for the first metres after tracking starts
or resumes: `sqrt(v0^2 + 2 a d)` from `start_speed_kmph` (5) at
`start_accel_mps2` (1.0) - 10 km/h after 2.9 m, 15 km/h after 7.7 m; the
status says `ramp` while it binds. Before the first vehicle state arrives the
controllers command nothing at all (`WAITING NO_VEHICLE_STATE`), so a manual
start is not fighting a brake on the CAN bus. How the first state is reached
at a standstill - ESKF, then LiDAR, then a creep - is under "Starting from a
standstill" above (`seed_heading_from_eskf` / `seed_heading_from_lidar` in
localization.launch.py, both default true).

**Curves have their own speed cap.** `route_publisher_node` limits each
waypoint to `min(max_speed_kmph, sqrt(max_lateral_accel_mps2 * R))`, then to
`curve_max_speed_kmph` wherever the smoothed radius is under `curve_radius_m`,
floored at `min_speed_kmph`, with braking/acceleration ramps between. As
configured (15 / 0.5 m/s² / 10 km/h under 30 m / 5), straights run 10-15 km/h
and curves 5-10 km/h; without the cap, 15-34 m bends ran 10-15 km/h. Both
`curve_*` at 0 disables it.

**Pure Pursuit has no steering gain.** The lookahead distance is the only knob
that sets steering aggressiveness — longer is smoother and cuts corners more,
shorter is tighter and can weave.

**At low speed the lookahead gain is inert.** `lookahead_gain_s` asks for
`gain * speed` metres and `lookahead_min_m` floors it. At the configured 1.4 s
gain and 5.0 m floor the floor wins below 3.57 m/s, which is most of the speed
range in use, so while testing slowly **tune `lookahead_min_m`**, not the gain.
`pure_pursuit_node` prints `gain binds above X m/s` at startup so this is
visible rather than inferred.

The same shape applied to speed while `max_speed_mps` was 3.0: the corner
limiter never bound on this route (its tightest 5.83 m radius allows 4.18 m/s),
so the profile was flat and `max_lateral_accel_mps2` did nothing. Since the
2026-09-12 raise to 4.2 m/s that slack is gone - the tightest corner now sits
exactly on the assumed 3.0 m/s^2 budget, which is an assumption and not a
measurement. Approach that speed in steps.

| `max_speed_mps` | mean | at ceiling | corner-limited |
|---|---|---|---|
| 3.0 | 2.99 | 376/379 | 0 |
| 4.0 | 3.98 | 375/379 | 0 |
| 5.0 | 4.96 | 368/379 | 5 |
| 9.0 | 7.95 | 189/379 | 85 |

## What this package assumes about the route

A saved route is a *recorded drive*, and `waypoint_saver_node` samples every
metre of **travel** without checking its **direction**. When localization jumps
backwards during mapping, the result is a waypoint pair that doubles back and
returns. On the real `20260907_114123` route that happens exactly once
(waypoints 224–225 re-tread 222–223) and it alone produces a **0.56 m radius**
corner, against a vehicle whose minimum is 3.4 m.

That is not a smoothing problem — arc-length lookahead cannot walk a polyline
that reverses, and projection onto it is ambiguous between the two passes. So
`route_publisher_node` drops those waypoints and says so in a warning naming the
indices. Measured on that route:

| | waypoints | length | tightest radius | corners below the 3.4 m spec |
|---|---|---|---|---|
| raw | 381 | 497.1 m | 3.17 m | 2 |
| filtered | 379 | 494.9 m | 5.83 m | 0 |

Curvature is smoothed over 5 samples on top of that. Unsmoothed, three-point
curvature differentiates the recorded line twice and brakes for noise on
straights: it costs 1.2 m/s of mean profiled speed on this route (6.79 vs
7.95 m/s) for corners that are not there.

## Tests

```bash
cd src/krowd_path_tracking && python3 -m pytest test -q
```

89 tests. The geometry, curvature, speed profile, vehicle, Pure Pursuit and
Stanley modules import nothing from `rclpy`; the one test that checks Stanley's
offset sign against `mola_slam_mapping.route_io` is skipped unless the workspace
is sourced.

## Portability

Intended to move to the main computer together with `mola_slam_mapping` and the
localization stack. No absolute paths: every location is a launch argument, and
the one default that names a place (`~/krowd_maps/latest`) is under `$HOME`.
Vehicle constants live in `config/vehicle.yaml` and are applied to all nodes
via `/**`, including `mola_slam_mapping`'s `wheel_odom_bicycle_node`, which
declares the same `wheelbase_m`.

The dependency on `mola_slam_mapping` is deliberate: `route_io` owns the
`waypoints.csv`/`corridor.csv` schema, the projection onto a route and the
arc-length lookahead (its `LookaheadPoint` docstring was already written for a
pure-pursuit consumer). Copying them here would let the reader and the writer of
those files drift apart.
