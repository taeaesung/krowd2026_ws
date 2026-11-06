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
 * @file   Solver.h
 * @brief  Virtual base class for optimal alignment solvers (one step in ICP).
 * @author Jose Luis Blanco Claraco
 * @date   August 3, 2020
 *
 * Defines the common interface implemented by every pose-optimization
 * backend (Solver_GaussNewton, Solver_OLAE, Solver_Horn): given a set of
 * point/line/plane correspondences, compute the relative SE(3)/SE(2) pose
 * that best aligns them. Solver itself is only bookkeeping (enable/disable,
 * iteration-range gating, an early-stop-once-converged option); the actual
 * math lives in the derived classes and in the optimal_tf_*() free functions
 * they call.
 */
#pragma once

#include <mp2p_icp/OptimalTF_Result.h>
#include <mp2p_icp/Pairings.h>
#include <mp2p_icp/Parameterizable.h>
#include <mp2p_icp/WeightParameters.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/poses/CPose3DPDFGaussianInf.h>
#include <mrpt/rtti/CObject.h>
#include <mrpt/system/COutputLogger.h>

#include <any>
#include <optional>

namespace mp2p_icp
{
class Solver;

/** Defines the context of a solver within the successive ICP iterations.
 *
 * \ingroup mp2p_icp_grp
 */
struct SolverContext
{
    SolverContext() = default;

    /** Current pose estimate the pairings were computed against (i.e. the
     * seed for this solve). */
    std::optional<mrpt::poses::CPose3D> guessRelativePose;
    /** Accumulated correction relative to the very first initial guess given
     * to ICP::align(), useful for solvers that want to bound total drift. */
    std::optional<mrpt::poses::CPose3D> currentCorrectionFromInitialGuess;
    /** The pose correction applied in the *previous* ICP iteration; used
     * e.g. by runUntilTranslationCorrectionSmallerThan below to detect that
     * a solver has effectively converged and can be skipped from then on. */
    std::optional<mrpt::poses::CPose3D> lastIcpStepIncrement;

    /** Optional prior guess of the SE(3) solution, including a mean value
     *  and an inverse covariance (information) matrix, i.e. zeros in the
     * diagonal mean that those prior coordinates should be ignored, a large
     * value means the solution must be close to those coordinates.
     */
    std::optional<mrpt::poses::CPose3DPDFGaussianInf> prior;

    // room for optional solver-specific context:
    // Keyed by solver instance so multiple solvers sharing one SolverContext
    // (as happens across ICP::run_solvers()) don't clobber each other's
    // state; used e.g. by Solver::optimal_pose() to remember "this solver
    // already reached its convergence threshold" across iterations (see
    // runUntilTranslationCorrectionSmallerThan handling in Solver.cpp).
    mutable std::map<const Solver*, std::map<std::string, std::any>> perSolverPersistentData;

    std::optional<uint32_t> icpIteration;
};

/** Virtual base class for optimal alignment solvers (one step in ICP).
 *
 * Each "solver" implementation takes a list of correspondences, and returns the
 * optimal (in some sense) relative SE(3) pose minimizing some particular error
 * function.
 *
 * \ingroup mp2p_icp_grp
 */
class Solver : public mrpt::system::COutputLogger,
               public mrpt::rtti::CObject,
               public mp2p_icp::Parameterizable
{
    DEFINE_VIRTUAL_MRPT_OBJECT(Solver, mp2p_icp)

   public:
    /** Check each derived class to see required and optional parameters. */
    virtual void initialize(const mrpt::containers::yaml& params);

    /** Non-virtual entry point called by ICP::run_solvers(): applies the
     * enable/iteration-range/convergence gating below and, if this solver
     * should run this iteration, delegates the actual math to
     * impl_optimal_pose() (the pure-virtual hook derived classes implement).
     * "out" is not cleared, but new pairings added to it.
     *
     * \return true if the method was actually invoked (due to the filter in
     * runFromIteration and runUpToIteration) and valid solution was found.
     */
    virtual bool optimal_pose(
        const Pairings& pairings, OptimalTF_Result& out, const SolverContext& sc) const;

    /** Skip this solver until this ICP iteration is reached (0 = from the
     * start). Lets a pipeline e.g. run a coarse solver first and only bring
     * in a finer one after a few iterations. */
    uint32_t runFromIteration = 0;
    uint32_t runUpToIteration = 0;  //!< 0: no limit

    /** If >0, once the previous iteration's translation correction norm
     * drops below this threshold, this solver reports itself as no longer
     * needed and stops running (see SolverContext::perSolverPersistentData);
     * useful to hand off from a robust-but-slow solver to a cheap one once
     * near convergence. 0 disables this behavior. */
    double runUntilTranslationCorrectionSmallerThan = 0;

    /** Can be used to disable one of a set of solvers in a pipeline */
    bool enabled = true;

   protected:
    /** Derived-class hook implementing the actual pose-optimization
     * algorithm. Must return false (without modifying `out`, ideally) if no
     * solution could be computed - e.g. too few pairings, or a degenerate/
     * unobservable correspondence geometry (see each derived class for its
     * specific failure conditions) - so that ICP::run_solvers() can fall
     * back to the next configured solver. */
    virtual bool impl_optimal_pose(
        const Pairings& pairings, OptimalTF_Result& out, const SolverContext& sc) const = 0;
};

}  // namespace mp2p_icp
