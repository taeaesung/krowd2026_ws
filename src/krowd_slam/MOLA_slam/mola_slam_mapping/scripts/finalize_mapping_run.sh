#!/bin/bash
# Convert a re-trajectoried run into one fail-closed localization product.
#
#   finalize_mapping_run.sh <run_dir> <source_bag>
#
# Success creates global_map_ready.mm, gnss_reference_localmap.ply,
# map_quality.yaml, and READY_FOR_LOCALIZATION. Any failed command leaves no
# READY marker, so an incomplete or poor map cannot be mistaken for deployable.
set -eo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <run_dir> <source_bag>" >&2
    exit 2
fi

RUN_DIR="$(readlink -f "$1")"
BAG_PATH="${2/#\~/$HOME}"
BAG_PATH="$(readlink -f "$BAG_PATH")"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

for required in \
    "$RUN_DIR/map.simplemap" \
    "$RUN_DIR/keyframes_corrected.tum" \
    "$RUN_DIR/trajectory_reference.csv" \
    "$RUN_DIR/global_map_corrected.georef.yaml" \
    "$RUN_DIR/waypoints.csv" \
    "$RUN_DIR/corridor.csv"; do
    if [ ! -s "$required" ]; then
        echo "finalize_mapping_run: missing or empty '$required'" >&2
        exit 1
    fi
done
if [ ! -d "$BAG_PATH" ] && [ ! -f "$BAG_PATH" ]; then
    echo "finalize_mapping_run: source bag does not exist: '$BAG_PATH'" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
source ~/krowd2026_ws/install/local_setup.bash

# Fail closed even when this is a re-run of a directory that previously passed.
rm -f "$RUN_DIR/READY_FOR_LOCALIZATION" "$RUN_DIR/MAP_NOT_READY"

PIPELINE="$(ros2 pkg prefix mola_slam_mapping)/share/mola_slam_mapping/config/sm2mm-retraject-incremental.yaml"
PLUGIN="$(ros2 pkg prefix mola_metric_maps)/lib/libmola_metric_maps.so"

echo "== ready 1/7 repair isolated start/end reference-anchor outliers"
python3 "$SCRIPT_DIR/repair_reference_anchor_outliers.py" \
    "$RUN_DIR" "$RUN_DIR/keyframes_ready.tum" "$RUN_DIR/keyframe_ready_repair.yaml" \
    --threshold-m 1.0

echo "== ready 2/7 apply final poses to a fresh simplemap"
sm-set-poses \
    "$RUN_DIR/map.simplemap" "$RUN_DIR/keyframes_ready.tum" \
    "$RUN_DIR/map_ready.simplemap"

echo "== ready 3/7 rebuild localization-compatible metric map"
sm2mm \
    -i "$RUN_DIR/map_ready.simplemap" \
    -o "$RUN_DIR/global_map_ready.mm" \
    -p "$PIPELINE" --no-progress-bar

echo "== ready 4/7 inject the corrected map georeference"
cp "$RUN_DIR/global_map_corrected.georef.yaml" "$RUN_DIR/global_map_ready.georef.yaml"
mm-georef \
    -m "$RUN_DIR/global_map_ready.mm" \
    -g "$RUN_DIR/global_map_ready.georef.yaml" \
    --inject-to-map -l "$PLUGIN"

echo "== ready 5/7 export the runtime MM layer for numeric quality checks"
mm2ply \
    -i "$RUN_DIR/global_map_ready.mm" \
    -o "$RUN_DIR/global_map_ready" -b --load-plugins "$PLUGIN"

echo "== ready 6/7 rebuild every raw scan on the continuous GNSS/IMU trajectory"
python3 "$SCRIPT_DIR/build_gnss_reference_lidar_map.py" \
    "$BAG_PATH" "$RUN_DIR" "$RUN_DIR/gnss_reference_localmap.ply" \
    --voxel-m 0.20 --stride 4 --report "$RUN_DIR/gnss_reference_build.yaml"

echo "== ready 7/7 run fail-closed trajectory, route, LiDAR, and surface checks"
python3 "$SCRIPT_DIR/validate_ready_map.py" "$RUN_DIR"

echo "READY: $RUN_DIR/global_map_ready.mm"
echo "localize with: ros2 launch mola_slam_mapping localization.launch.py map_dir:=$RUN_DIR map_filename:=global_map_ready.mm"
