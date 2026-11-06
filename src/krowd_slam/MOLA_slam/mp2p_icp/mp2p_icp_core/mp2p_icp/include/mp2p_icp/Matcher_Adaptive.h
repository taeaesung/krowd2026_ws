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
 * @file   Matcher_Adaptive.h
 * @brief  Pointcloud matcher: smart adaptive matcher
 * @author Jose Luis Blanco Claraco
 * @date   Nov 11, 2023
 *
 * The most sophisticated of the point matchers. Instead of a user-specified
 * fixed distance threshold, it derives the inlier/outlier cutoff at runtime
 * from the actual distribution of nearest-neighbor distances observed in
 * this ICP iteration (via a histogram + confidence interval), so it adapts
 * automatically as the alignment converges and errors shrink. It can also
 * optionally locally fit a plane to each point's neighborhood (PCA
 * eigenvalue planarity test) and emit a point-to-plane pairing instead of a
 * point-to-point one when the neighborhood is locally flat - see
 * implMatchOneLayer() in the .cpp for the full algorithm.
 */
#pragma once

#include <mp2p_icp/Matcher_Points_Base.h>
#include <mrpt/containers/vector_with_small_size_optimization.h>

namespace mp2p_icp
{
/** Pointcloud matcher: adaptive algorithm automatically discarding outliers,
 *  and creating point-to-plane, point-to-line, or point-to-point pairings.
 *
 * Finds pairings between the `local` and `global` input metric maps.
 *
 * By default, each `local` point layer is matched against the layer with the
 * same name in the `global` map, unless specified otherwise in the base class
 * member `weight_pt2pt_layers`. Refer to example configuration YAML files for
 * example configurations.
 *
 * \ingroup mp2p_icp_grp
 */
class Matcher_Adaptive : public Matcher_Points_Base
{
    DEFINE_MRPT_OBJECT(Matcher_Adaptive, mp2p_icp)

   public:
    Matcher_Adaptive() = default;

    Matcher_Adaptive(
        double ConfidenceInterval, double FirstToSecondDistanceMax,
        double AbsoluteMaxSearchDistance)
        : Matcher_Points_Base(),
          confidenceInterval(ConfidenceInterval),
          firstToSecondDistanceMax(FirstToSecondDistanceMax),
          absoluteMaxSearchDistance(AbsoluteMaxSearchDistance)
    {
    }

    /** Parameters:
     * - `confidenceInterval`: Inliers top-confidence interval. (0-1)
     * - `firstToSecondDistanceMax`:
     * - `absoluteMaxSearchDistance`:
     *
     * Plus: the parameters of Matcher_Points_Base::initialize()
     */
    void initialize(const mrpt::containers::yaml& params) override;

   private:
    /// Fraction (0-1) of the 1st/2nd-nearest-neighbor error histogram to keep
    /// as inliers; the upper bound of this confidence interval becomes the
    /// adaptive distance threshold for this ICP iteration (see ci_high in
    /// the .cpp). Higher = more permissive/adaptive threshold.
    double         confidenceInterval        = 0.80;
    /// Ratio gate for accepting more than one correspondence per local point:
    /// a candidate beyond the closest one is only kept if its squared error
    /// is within this factor of the closest match's squared error.
    double         firstToSecondDistanceMax  = 1.2;
    /// Hard cap (radius, in meters) for the KD-tree neighbor search,
    /// independent of the adaptive threshold; also seeds the histogram
    /// range. `mutable` because a (currently commented-out) update path in
    /// the .cpp would let it self-adjust across iterations.
    mutable double absoluteMaxSearchDistance = 5.0;  // m
    /// If true, attempt local plane fitting per local point (see
    /// planeSearchPoints/planeMinimumFoundPoints/planeEigenThreshold below)
    /// and emit point-to-plane pairings where the neighborhood is flat
    /// enough, instead of plain point-to-point pairings.
    bool           enableDetectPlanes        = false;
    /// Max number of point-to-point correspondences kept per local point
    /// when NOT emitting a point-to-plane pairing for it.
    uint32_t       maxPt2PtCorrespondences   = 1;
    /// Number of nearest global neighbors to fetch per local point when
    /// plane detection is enabled (must be >= planeMinimumFoundPoints).
    uint32_t       planeSearchPoints         = 8;
    /// Minimum number of neighbor points required to attempt a plane fit
    /// (must be >= 3, the minimum to define a plane).
    uint32_t       planeMinimumFoundPoints   = 4;
    /// Max point-to-plane distance [m] for a fitted plane to be accepted as
    /// the pairing for this local point (planarity alone is not enough; the
    /// local point itself must also lie close to the fitted plane).
    double         planeMinimumDistance      = 0.10;
    /// Planarity test threshold: the neighborhood is considered "flat"
    /// only if its smallest PCA eigenvalue is below this fraction of the
    /// other two (see eig.eigVals check in implMatchOneLayer()). Smaller =
    /// stricter (closer to a perfect plane required).
    double         planeEigenThreshold       = 0.01;
    /// Lower bound imposed on the adaptive squared-distance threshold, so a
    /// very tight/degenerate confidence interval never collapses the
    /// acceptance radius to (near) zero.
    double         minimumCorrDist           = 0.1;  // m

    // Declared here to avoid memory reallocations:
    mutable std::vector<uint64_t>              neighborIndices_;
    mutable std::vector<float>                 neighborSqrDists_;
    mutable std::vector<mrpt::math::TPoint3Df> neighborPts_;
    mutable std::vector<float>                 kddXs, kddYs, kddZs;
    mutable std::vector<double>                histXs_, histValues_;

    /// Cap on how many candidate neighbor pairings are buffered per local
    /// point before further ones are silently dropped (see lambdaAddPair()).
    constexpr static size_t MAX_CORRS_PER_LOCAL = 10;

    /// Per-local-point buffer of candidate (local, global) matches found by
    /// the KD-tree search, before the adaptive threshold/plane-fit decision
    /// is applied. Indexed by local point index.
    mutable std::vector<mrpt::containers::vector_with_small_size_optimization<
        mrpt::tfest::TMatchingPair, MAX_CORRS_PER_LOCAL>>
        matchesPerLocal_;

    /** Two-pass matching per layer: (1) gather up to
     * planeSearchPoints/maxPt2PtCorrespondences nearest neighbors for every
     * local point within absoluteMaxSearchDistance, while also building a
     * histogram of 1st/2nd-neighbor squared errors; (2) derive an adaptive
     * distance threshold from that histogram's confidence interval, then for
     * each local point either fit+accept a plane (if enableDetectPlanes and
     * the neighborhood passes the planarity + distance tests) or fall back
     * to point-to-point pairing(s) gated by the adaptive threshold and the
     * firstToSecondDistanceMax ratio. See the .cpp for the full logic. */
    void implMatchOneLayer(
        const mrpt::maps::CMetricMap& pcGlobal, const mrpt::maps::CPointsMap& pcLocal,
        const mrpt::poses::CPose3D& localPose, MatchState& ms, const layer_name_t& globalName,
        const layer_name_t& localName, Pairings& out) const override;
};

}  // namespace mp2p_icp
