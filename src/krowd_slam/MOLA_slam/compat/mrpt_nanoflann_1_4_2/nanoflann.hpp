#pragma once

// ROS 2 Humble's MRPT 2.15 shared libraries in this workspace were built
// against Ubuntu Jammy's nanoflann 1.4.2. A newer ros-humble-nanoflann header
// also exists in /opt/ros/humble and otherwise shadows that ABI at compile
// time. Keep every translation unit that includes MRPT point-map headers on
// MRPT's build-time ABI. mola::IncrementalPointCloud includes its private
// nanoflann 1.11 header by absolute path and in a renamed namespace, so this
// compatibility shim does not affect that isolated implementation.
#include </usr/include/nanoflann.hpp>
