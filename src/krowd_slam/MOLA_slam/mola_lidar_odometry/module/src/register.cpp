/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
 Closed-source licenses available upon request, for this odometry package
 alone or in combination with the complete SLAM system.
*/

/**
 * @file   register.cpp
 * @brief  Register MOLA modules in the factory
 * @author Jose Luis Blanco Claraco
 * @date   Jan 08, 2018
 *
 * Standalone entry point (no relation to the LidarOdometry.h/.cpp pipeline
 * logic) that makes the mola::LidarOdometry class known to MRPT's RTTI/class
 * factory, so a MOLA YAML pipeline file can instantiate it by class name
 * (e.g. `class: LidarOdometry`) without any of the other translation units
 * needing to reference the class directly.
 */

/** \defgroup mola_mola_slam_grp mola_lidar_odometry
 * MOLA module: LIDAR-based odometry systems.
 *
 */

#include <mola_lidar_odometry/LidarOdometry.h>
#include <mrpt/core/initializer.h>
#include <mrpt/rtti/CObject.h>

// MRPT_INITIALIZER schedules the given function to run automatically at
// static-initialization time (before main()), i.e. as soon as this
// translation unit's object file is linked in. That is why there is no
// explicit call to do_register_mola_lidar_odometry() anywhere else.
MRPT_INITIALIZER(do_register_mola_lidar_odometry)  // NOLINT
{
  using mrpt::rtti::registerClass;

  // Register modules:
  MOLA_REGISTER_MODULE(mola::LidarOdometry);
}
