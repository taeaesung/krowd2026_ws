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
 * @file   Matcher.cpp
 * @brief  Pointcloud matching generic base class
 * @author Jose Luis Blanco Claraco
 * @date   June 22, 2020
 *
 * Implements the base-class plumbing shared by all matchers: parameter
 * loading and the enabled/iteration-range gating in match(), plus the
 * run_matchers() helper that drives a whole matcher pipeline. The actual
 * correspondence-search logic lives in the derived-class impl_match()
 * overrides (Matcher_Points_*, Matcher_Point2Plane, etc.), not here.
 */

#include <mp2p_icp/Matcher.h>
#include <mrpt/core/exceptions.h>

IMPLEMENTS_VIRTUAL_MRPT_OBJECT(Matcher, mrpt::rtti::CObject, mp2p_icp)

using namespace mp2p_icp;

void Matcher::initialize(const mrpt::containers::yaml& params)
{
    // MCP_LOAD_OPT: load an optional key from YAML, leaving the C++ default
    // untouched if absent.
    MCP_LOAD_OPT(params, runFromIteration);
    MCP_LOAD_OPT(params, runUpToIteration);
    MCP_LOAD_OPT(params, enabled);
}

bool Matcher::match(
    const metric_map_t& pcGlobal, const metric_map_t& pcLocal,
    const mrpt::poses::CPose3D& localPose, const MatchContext& mc, MatchState& ms,
    Pairings& out) const
{
    // Gate execution: disabled matchers, or matchers outside their
    // configured ICP-iteration window, contribute no pairings this call.
    if (!enabled) return false;
    if (mc.icpIteration < runFromIteration) return false;
    if (runUpToIteration > 0 && mc.icpIteration > runUpToIteration) return false;
    return impl_match(pcGlobal, pcLocal, localPose, mc, ms, out);
}

Pairings mp2p_icp::run_matchers(
    const matcher_list_t& matchers, const metric_map_t& pcGlobal, const metric_map_t& pcLocal,
    const mrpt::poses::CPose3D& local_wrt_global, const MatchContext& mc,
    const mrpt::optional_ref<MatchState>& userProvidedMS)
{
    Pairings pairings;

    MatchState* ms = nullptr;

    std::optional<MatchState> localMS;

    if (userProvidedMS.has_value())
    {
        // Use user-provided memory storage:
        ms = &userProvidedMS.value().get();
    }
    else
    {
        // Reserve here:
        // No caller-provided state (e.g. standalone use outside ICP::run()):
        // allocate one locally, scoped to this call, so all matchers below
        // still share a single "who's already paired" bitfield.
        localMS.emplace(pcGlobal, pcLocal);
        ms = &localMS.value();
    }

    bool anyRun = false;

    // Run every matcher in order against the *same* ms, so each one can see
    // pairings claimed by earlier matchers in the pipeline. Results from all
    // matchers are appended (not merged/deduplicated) into `pairings`.
    for (const auto& matcher : matchers)
    {
        ASSERT_(matcher);
        Pairings pc;
        bool     hasRun = matcher->match(pcGlobal, pcLocal, local_wrt_global, mc, *ms, pc);
        anyRun          = anyRun || hasRun;
        pairings.push_back(pc);
    }

    // If every matcher was disabled or outside its iteration range, this is
    // almost certainly a misconfiguration (ICP would then have zero
    // correspondences to optimize against), so warn loudly.
    if (!anyRun)
    {
        std::cerr << "[mp2p_icp::run_matchers] WARNING: No active matcher "
                     "actually ran on the two maps."
                  << std::endl;
    }

    return pairings;
}
