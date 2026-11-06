#!/bin/bash
# Map, then re-trajectory the map onto the RTK reference - one command.
#
#   map_and_retraject.sh [launch args...]
#   MAPPING_LAUNCH=mapping_bag.launch.py map_and_retraject.sh bag_path:=~/testweek_0905_3
#
# Runs the mapping launch in the foreground exactly as `ros2 launch` would;
# when you Ctrl-C it, mola-cli writes map.simplemap and this script then runs
# scripts/retraject_map.sh on that run. The result is
# <run>/global_map_corrected.mm, built on the fused/RTK trajectory rather than
# on LiDAR odometry's own drifting one.
#
# Why the correction cannot simply be applied while mapping: the pose that
# feeds ICP must stay LiDAR-only (see the smoother A/B in
# config/mapping_system.yaml), and the best estimate of where a keyframe
# really was needs the whole run - the pose graph in map_retraject uses
# keyframes recorded after it. During the run, the "Corrected Map Preview
# (fused pose)" RViz display shows an approximation of the same thing live.
#
# No -u: /opt/ros/humble/setup.bash reads unset variables.
set -eo pipefail

LAUNCH_FILE="${MAPPING_LAUNCH:-mapping.launch.py}"
MAP_ROOT="${KROWD_MAP_ROOT:-$HOME/krowd_maps}"

# local_setup.bash, NOT setup.bash - see run_mapping_with_waypoints.sh for why.
source /opt/ros/humble/setup.bash
source ~/krowd2026_ws/install/local_setup.bash

# `set -m` gives the job its own process group, so a terminal Ctrl-C reaches
# this script only and the trap below forwards it to the whole group. A bare
# SIGTERM to `ros2 launch` would signal no child and orphan mola-cli, which
# writes every map file from its own SIGINT path.
set -m
ros2 launch mola_slam_mapping "$LAUNCH_FILE" "$@" &
LAUNCH_PID=$!
trap 'kill -INT -"$LAUNCH_PID" 2>/dev/null || true' INT TERM
# `wait` returns as soon as a trapped signal fires, not when the job ends, so
# keep waiting until the process group is really gone (mola-cli takes seconds
# to write the simplemap and rebuild the global map).
while kill -0 "$LAUNCH_PID" 2>/dev/null; do
    wait "$LAUNCH_PID" || true
done
trap - INT TERM

RUN_DIR="$(readlink -f "$MAP_ROOT/latest" 2>/dev/null || true)"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "map_and_retraject: no '$MAP_ROOT/latest' run directory; nothing to post-process." >&2
    exit 1
fi

echo
echo "=== mapping finished: $RUN_DIR"
for required in map.simplemap trajectory_reference.csv georeference.yaml; do
    if [ ! -s "$RUN_DIR/$required" ]; then
        echo "map_and_retraject: '$required' is missing or empty, so the map cannot be" >&2
        echo "  re-trajectoried. The uncorrected map.simplemap / global_map.mm are still there." >&2
        echo "  (trajectory_reference.csv needs /gnss_imu/pose, i.e. both GNSS topics.)" >&2
        exit 1
    fi
done

# Invoked through `bash` on purpose: setup.py installs scripts/ with a plain
# data_files glob, which does NOT preserve the executable bit (the installed
# copy of run_mapping_with_waypoints.sh is 644 even though its source is 755).
# A bare exec of a 644 file dies with EACCES/126 - after the whole mapping run
# has already been driven and Ctrl-C'd, which is the worst possible moment.
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
bash "$SCRIPT_DIR/retraject_map.sh" "$RUN_DIR"

# Offline bag mapping can now produce a complete, fail-closed localization
# product in this same command. Live mapping has no bag containing all raw
# scans, so it deliberately stops at global_map_corrected.mm instead of
# pretending that the GNSS-reference initializer was built.
BAG_PATH=""
for argument in "$@"; do
    case "$argument" in
        bag_path:=*) BAG_PATH="${argument#bag_path:=}" ;;
    esac
done

if [ -n "$BAG_PATH" ]; then
    bash "$SCRIPT_DIR/finalize_mapping_run.sh" "$RUN_DIR" "$BAG_PATH"
else
    echo
    echo "map_and_retraject: no bag_path:=... argument; global_map_corrected.mm exists," >&2
    echo "  but READY_FOR_LOCALIZATION was intentionally not created." >&2
    echo "  To finalize later: bash '$SCRIPT_DIR/finalize_mapping_run.sh' '$RUN_DIR' <source_bag>" >&2
fi
