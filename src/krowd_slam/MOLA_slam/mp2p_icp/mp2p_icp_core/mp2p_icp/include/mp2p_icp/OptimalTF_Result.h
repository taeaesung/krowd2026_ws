/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 A repertory of multi primitive-to-primitive (MP2P) ICP algorithms
 and map building tools. mp2p_icp is part of MOLA.

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: BSD-3-Clause
*/
/**
 * @file   OptimalTF_Result.h
 * @brief  Common types for all SE(3) optimal transformation methods.
 * @author Jose Luis Blanco Claraco
 * @date   Jun 16, 2019
 */
#pragma once

#include <mp2p_icp/Pairings.h>
#include <mrpt/poses/CPose3D.h>

namespace mp2p_icp
{
/** \addtogroup  mp2p_icp_grp
 * @{ */

/** The is the output structure for all optimal transformation methods
 * (optimal_tf_gauss_newton(), optimal_tf_olae(), optimal_tf_horn(), and the
 * Solver classes wrapping them). Also serves as the in/out state threaded
 * through ICP's main loop (see ICP::ICP_State::currentSolution), since each
 * iteration's solution is the seed for the next.
 */
struct OptimalTF_Result
{
    /** The estimated relative pose (local wrt global). */
    mrpt::poses::CPose3D optimalPose;
    /** Estimated uniform scale factor, only meaningful for solvers that
     * estimate a similarity transform (e.g. Horn's method with scale
     * enabled); left at 1.0 by solvers that only estimate rigid SE(3). */
    double optimalScale = 1.0;

    /** Correspondence that were detected as outliers (e.g. via the
     * scale-based outlier detector or a robust kernel), so they can be
     * excluded from covariance estimation and quality scoring downstream. */
    OutlierIndices outliers;
};

/** @} */

}  // namespace mp2p_icp
