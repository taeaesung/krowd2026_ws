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
 * @file   Solver.cpp
 * @brief  Virtual base class for optimal alignment solvers (one step in ICP).
 * @author Jose Luis Blanco Claraco
 * @date   August 3, 2020
 *
 * Implements the gating logic shared by all solvers (enabled flag,
 * iteration-range restriction, "stop once converged" early-out); the actual
 * pose-optimization math is in each derived Solver_*.cpp.
 */

#include <mp2p_icp/Solver.h>
#include <mrpt/core/exceptions.h>

IMPLEMENTS_VIRTUAL_MRPT_OBJECT(Solver, mrpt::rtti::CObject, mp2p_icp)

using namespace mp2p_icp;

void Solver::initialize(const mrpt::containers::yaml& params)
{
    MCP_LOAD_OPT(params, runFromIteration);
    MCP_LOAD_OPT(params, runUpToIteration);
    MCP_LOAD_OPT(params, enabled);
    MCP_LOAD_OPT(params, runUntilTranslationCorrectionSmallerThan);
}

bool Solver::optimal_pose(
    const Pairings& pairings, OptimalTF_Result& out, const SolverContext& sc) const
{
    if (!enabled) return false;

    // Iteration-range gating: skip entirely if outside [runFromIteration,
    // runUpToIteration]. Only enforced when the caller actually provides an
    // iteration number (icpIteration may be unset outside of the ICP loop,
    // e.g. when a solver is invoked standalone).
    const auto iter = sc.icpIteration;
    if (iter.has_value() && *iter < runFromIteration) return false;
    if (iter.has_value() && runUpToIteration > 0 && *iter > runUpToIteration) return false;

    if (runUntilTranslationCorrectionSmallerThan > 0)
    {
        // already fulfilled in past iters?
        // Once this solver has decided it's converged (flag stored per
        // solver-instance in the shared SolverContext), keep skipping it on
        // every subsequent iteration without re-checking the threshold -
        // this is a one-way latch, not a re-armed condition, so a solver
        // won't "come back" if the correction happens to grow again later.
        auto& myData = sc.perSolverPersistentData[this];
        if (myData.count("finished") != 0) return false;

        // Detect threshold:
        if (sc.lastIcpStepIncrement && sc.lastIcpStepIncrement.value().translation().norm() <
                                           runUntilTranslationCorrectionSmallerThan)
        {
            // Yes, stop using this solver.
            // Store  the condition and quit.
            std::any& value = myData["finished"];
            value           = true;
            return false;
        }
    }

    return impl_optimal_pose(pairings, out, sc);
}
