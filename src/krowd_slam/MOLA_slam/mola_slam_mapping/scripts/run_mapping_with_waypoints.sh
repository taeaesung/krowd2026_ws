#!/bin/bash
# Runs mapping.launch.py plus mark_waypoint_keyboard_node together from one
# terminal/command. Any arguments passed to this script are forwarded to
# mapping.launch.py (e.g. wheel_odom_source:=bag).
#
# Why this can't just be another Node in mapping.launch.py: mark_waypoint_keyboard_node
# needs a real interactive terminal (raw stdin) to read SPACE keypresses, but
# processes launched by `ros2 launch` don't get one - confirmed by rosbag2_player's
# own "stdin is not a terminal device. Keyboard handling disabled." message when
# run the same way in bag_test.launch.py. So mapping.launch.py runs in the
# background here, and the keyboard listener runs in this script's own foreground,
# which does have a real tty.
set -e

# local_setup.bash, NOT setup.bash. setup.bash replays each workspace's recorded
# prefix chain, which was observed to move the prebuilt mola_3_1_0_ws packages ahead
# of the locally built ones, so mola_launcher/mola_bridge_ros2/the state estimator
# resolve from the wrong prefix and mola-cli SIGSEGVs before the first sensor sample.
source /opt/ros/humble/setup.bash
source ~/krowd2026_ws/install/local_setup.bash

# `set -m` gives the async job its own process group (PGID == $!) and restores
# SIGINT to the default disposition for it; without it a background job in a
# non-interactive script inherits SIGINT=SIG_IGN and `kill -INT` is a silent no-op.
set -m
ros2 launch mola_slam_mapping mapping.launch.py "$@" &
LAUNCH_PGID=$!
# Signal the GROUP, and with SIGINT. `ros2 launch` starts children without setsid,
# so they share this group and receive SIGINT exactly as they would from a terminal
# Ctrl-C. A bare `kill` sends SIGTERM to the launch process only: it cancels its own
# task without signalling any child, orphaning mola-cli. mola-cli handles SIGINT
# alone, and map.simplemap / rolling_local_map.mm / global_map.mm are written solely
# from its shutdown path - so SIGTERM here silently discards the entire map.
trap 'kill -INT -"$LAUNCH_PGID" 2>/dev/null' EXIT

# Give mola-cli + waypoint_saver_node a moment to come up before the
# ~/mark_waypoint service exists to wait for.
sleep 3

ros2 run mola_slam_mapping mark_waypoint_keyboard_node

# The keyboard node returning means the operator is done. Ask the stack to stop and
# then WAIT for it: the shutdown rebuilds global_map.mm from the simplemap, which
# takes seconds to minutes on a long run. Exiting here instead would fire the EXIT
# trap and leave no time for any of it to be written.
kill -INT -"$LAUNCH_PGID" 2>/dev/null || true
wait "$LAUNCH_PGID"
