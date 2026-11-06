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
 * @file   Matcher.h
 * @brief  Pointcloud matching generic base class
 * @author Jose Luis Blanco Claraco
 * @date   June 22, 2020
 *
 * This file defines the abstract interface for the "correspondence search"
 * step of ICP: given two point clouds (a "global"/reference map and a
 * "local"/mobile one placed at a hypothesized SE(3) pose) a Matcher decides
 * which geometric entities (points, planes, lines, ...) of one cloud pair up
 * with which entities of the other. Concrete matchers (Matcher_Points_*,
 * Matcher_Point2Plane, Matcher_Point2Line, Matcher_Cov2Cov, ...) implement
 * different pairing criteria and live in their own files; this one only
 * defines the common base class, the per-call context/state objects, and the
 * helper that runs a whole sequence ("pipeline") of matchers. The resulting
 * Pairings are later consumed by the pose-optimization solvers (ICP.cpp,
 * Solver_*) to compute the next pose update.
 */
#pragma once

#include <mp2p_icp/Pairings.h>
#include <mp2p_icp/Parameterizable.h>
#include <mp2p_icp/metricmap.h>
#include <mp2p_icp/pointcloud_bitfield.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/rtti/CObject.h>
#include <mrpt/system/COutputLogger.h>

namespace mp2p_icp
{
/** Defines the static part of a match operation.
 * "Static" here means information that does not change while a single
 * matcher runs (as opposed to MatchState, which is mutated as matchers in a
 * pipeline claim entities). Currently only carries the ICP iteration index,
 * which individual matchers use (via runFromIteration/runUpToIteration) to
 * enable/disable themselves at different stages of the ICP loop (e.g. run a
 * coarse matcher first, then switch to a stricter one once converging).
 *
 * \ingroup mp2p_icp_grp
 */
struct MatchContext
{
    MatchContext() = default;

    /// The ICP iteration number we are in:
    uint32_t icpIteration = 0;
};

/** Mutable, per-matching-pipeline-run bookkeeping shared across all matchers
 * invoked in sequence (see run_matchers()). Its purpose is to let a matcher
 * that runs later in the pipeline know which entities were already paired by
 * an earlier matcher, so that (depending on the matcher's own logic) it can
 * avoid assigning the same point/plane/line a second, conflicting pairing.
 */
struct MatchState
{
    MatchState(const metric_map_t& pcGlobal, const metric_map_t& pcLocal)
        : pcGlobal_(pcGlobal), pcLocal_(pcLocal)
    {
        initialize();
    }

    /// The pairings already assigned by former matches in the pipeline
    /// indexed by the pcLocal entities (true=already have a pairing).
    pointcloud_bitfield_t localPairedBitField;

    /// Like localPairedBitField for the global map
    pointcloud_bitfield_t globalPairedBitField;

    /** Initialize all bit fields to their correct length and default value
     * (false) */
    void initialize()
    {
        localPairedBitField.initialize_from(pcLocal_);
        globalPairedBitField.initialize_from(pcGlobal_);
    }

   private:
    // Kept only to (re)size the bitfields in initialize(); not stored for
    // later per-entity lookups (matchers receive pcGlobal/pcLocal directly).
    const metric_map_t& pcGlobal_;
    const metric_map_t& pcLocal_;
};

/** Pointcloud matching generic base class.
 * Each "matcher" implementation takes a global ("reference") `metric_map_t` and
 * another local ("mobile") `metric_map_t` which is assumed to be placed in a
 * hypothetical SE(3) pose in the global frame, and generates pairings between
 * the geometric entities (points, planes, etc.) of both groups.
 *
 * Derived classes implement impl_match() with the actual correspondence
 * search (typically backed by a KD-tree nearest-neighbor query into
 * pcGlobal). The base class only handles the common enable/disable and
 * iteration-range gating (see match()), plus RTTI/parameter-loading
 * plumbing inherited from CObject/Parameterizable.
 *
 * \ingroup mp2p_icp_grp
 */
class Matcher : public mrpt::system::COutputLogger,
                public mrpt::rtti::CObject,
                public mp2p_icp::Parameterizable
{
    DEFINE_VIRTUAL_MRPT_OBJECT(Matcher, mp2p_icp)

   public:
    /** Loads matcher parameters from a YAML node.
     * Check each derived class to see required and optional parameters. */
    virtual void initialize(const mrpt::containers::yaml& params);

    /** Finds correspondences between the two point clouds.
     * "out" is not cleared, but new pairings added to it.
     * First checks enabled/runFromIteration/runUpToIteration and, if the
     * matcher should run at this ICP iteration, delegates to the derived
     * class' impl_match() for the actual correspondence search.
     * \return false if the matcher is disabled and was not actually run.
     */
    virtual bool match(
        const metric_map_t& pcGlobal, const metric_map_t& pcLocal,
        const mrpt::poses::CPose3D& localPose, const MatchContext& mc, MatchState& ms,
        Pairings& out) const;

    /// Skip this matcher on ICP iterations before this index (0-based).
    /// Lets a pipeline start with a loose/robust matcher and only switch to
    /// e.g. a stricter plane/line matcher once the pose estimate is closer.
    uint32_t runFromIteration = 0;
    /// Skip this matcher on ICP iterations after this index; 0 means no limit.
    uint32_t runUpToIteration = 0;  //!< 0: no limit
    /// Master on/off switch, independent of the iteration range above.
    bool     enabled          = true;

   protected:
    /** Derived-class hook implementing the actual correspondence search
     * between pcGlobal and pcLocal (the latter transformed by localPose),
     * appending new entries to `out`. `ms` carries which entities are
     * already paired by earlier matchers in the same pipeline run, so this
     * matcher's chosen pairing criterion can decide whether to reuse or
     * skip already-paired entities.
     * \return true if the mather is actually invoked, false if disabled.
     */
    virtual bool impl_match(
        const metric_map_t& pcGlobal, const metric_map_t& pcLocal,
        const mrpt::poses::CPose3D& localPose, const MatchContext& mc, MatchState& ms,
        Pairings& out) const = 0;
};

/// An ordered list of matchers, run in sequence to build up one Pairings set
/// (a matching "pipeline"), typically configured from YAML.
using matcher_list_t = std::vector<mp2p_icp::Matcher::Ptr>;

/** Runs a sequence of matcher between two metric_map_t objects.
 *
 * Each matcher in `matchers` is invoked in order against the same pair of
 * point clouds and pose hypothesis, sharing one MatchState (either supplied
 * via `userProvidedMS` or created locally) so later matchers can see which
 * entities earlier ones already paired. All resulting per-matcher Pairings
 * are concatenated into the single Pairings object returned.
 *
 * This is normally invoked by mp2p_icp::ICP, but users can use it as a
 * standalone module as needed.
 *
 * \ingroup mp2p_icp_grp
 */
Pairings run_matchers(
    const matcher_list_t& matchers, const metric_map_t& pcGlobal, const metric_map_t& pcLocal,
    const mrpt::poses::CPose3D& local_wrt_global, const MatchContext& mc,
    const mrpt::optional_ref<MatchState>& userProvidedMS = std::nullopt);

}  // namespace mp2p_icp
