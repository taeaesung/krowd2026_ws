#!/bin/bash
# Post-process one mapping run: move the simplemap keyframes onto the RTK
# reference trajectory and rebuild a localization-compatible point-cloud map.
#
#   retraject_map.sh <run_dir> [map_name]     (map_name default: global_map_corrected)
#
# Needs the run to have been mapped with trajectory_recorder_node running
# (mapping.launch.py since 2026-09-06), i.e. trajectory_reference.csv exists.
# Steps: sm-cli export-keyframes -> map_retraject (pose graph, Python) ->
# sm-set-poses (writes the poses back, C++) -> sm2mm (rebuild). The result is
# in the run's `map` frame with its own <map_name>.georef.yaml, so
#   ros2 launch mola_slam_mapping localization.launch.py map_dir:=<run_dir> map_filename:=<map_name>.mm
# localizes against it without any other change.
# No -u: /opt/ros/humble/setup.bash reads unset variables.
set -eo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "usage: $0 <run_dir> [map_name]" >&2
    exit 2
fi
RUN_DIR="$1"
MAP_NAME="${2:-global_map_corrected}"

for required in map.simplemap trajectory_reference.csv georeference.yaml; do
    if [ ! -f "$RUN_DIR/$required" ]; then
        echo "error: '$RUN_DIR/$required' is missing" >&2
        exit 1
    fi
done

# local_setup.bash, NOT setup.bash - see run_mapping_with_waypoints.sh.
source /opt/ros/humble/setup.bash
source ~/krowd2026_ws/install/local_setup.bash

PIPELINE="$(ros2 pkg prefix mola_slam_mapping)/share/mola_slam_mapping/config/sm2mm-retraject-incremental.yaml"

echo "== 1/5 export LiDAR-odometry keyframes"
sm-cli export-keyframes "$RUN_DIR/map.simplemap" --output "$RUN_DIR/keyframes_lo.tum"

echo "== 2/5 optimize keyframes onto the reference trajectory"
# ament_python console scripts live in lib/<pkg>/, not on PATH.
ros2 run mola_slam_mapping map_retraject -- --run-dir "$RUN_DIR" --map-name "$MAP_NAME.mm"

echo "== 3/5 write the corrected poses into a copy of the simplemap"
sm-set-poses "$RUN_DIR/map.simplemap" "$RUN_DIR/keyframes_corrected.tum" "$RUN_DIR/map_corrected.simplemap"

echo "== 4/5 rebuild the point-cloud map (IncrementalPointCloud, layer localmap)"
sm2mm -i "$RUN_DIR/map_corrected.simplemap" -o "$RUN_DIR/$MAP_NAME.mm" -p "$PIPELINE" --no-progress-bar

echo "== 5/5 embed the georeference so the map is self-describing"
# NOT sm2mm's own -g: that reprojects every point into ENU and stores an
# identity T_enu_to_map. The map must stay in the run's `map` frame with the
# real transform, which is what mm-georef --inject-to-map writes. The sidecar
# stays on disk and remains what localization.launch.py actually reads.
mm-georef -m "$RUN_DIR/$MAP_NAME.mm" -g "$RUN_DIR/$MAP_NAME.georef.yaml" --inject-to-map \
    -l "$(ros2 pkg prefix mola_metric_maps)/lib/libmola_metric_maps.so"

echo "done: $RUN_DIR/$MAP_NAME.mm (+ $MAP_NAME.georef.yaml, retraject.yaml)"
echo "localize with: ros2 launch mola_slam_mapping localization.launch.py map_dir:=$RUN_DIR map_filename:=$MAP_NAME.mm"
