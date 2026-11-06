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
 * @file   Matcher_Points_DistanceThreshold.h
 * @brief  Pointcloud matcher: fixed distance thresholds
 * @author Jose Luis Blanco Claraco
 * @date   June 22, 2020
 *
 * The workhorse "vanilla ICP" point matcher: for each local point, finds its
 * nearest neighbor(s) in the global map's KD-tree and accepts the pairing
 * only if within a fixed Euclidean distance threshold (optionally relaxed
 * for far-away points via an angular term). This is typically the first/only
 * matcher in a pipeline for point-to-point ICP variants.
 */
#pragma once

#include <mp2p_icp/Matcher_Points_Base.h>

namespace mp2p_icp
{
/** Pointcloud matcher: fixed distance thresholds.
 *
 * Finds point-to-point pairings between the `local` and `global` input metric
 * maps.
 *
 * By default, each `local` point layer is matched against the layer with the
 * same name in the `global` map, unless especified otherwise in the base class
 * member `weight_pt2pt_layers`. Refer to example configuration YAML files for
 * example configurations.
 *
 * \ingroup mp2p_icp_grp
 */
class Matcher_Points_DistanceThreshold : public Matcher_Points_Base
{
    DEFINE_MRPT_OBJECT(Matcher_Points_DistanceThreshold, mp2p_icp)

   public:
    Matcher_Points_DistanceThreshold();

    Matcher_Points_DistanceThreshold(double distThreshold) : Matcher_Points_DistanceThreshold()
    {
        threshold = distThreshold;
    }

    /** Parameters:
     * - `threshold`: Inliers distance threshold [meters][mandatory]
     * - `pairingsPerPoint`: Number of pairings in "global" for each "local"
     * points. Default=1. If more than one, they will be picked in ascending
     * order of distance, up to `threshold`. [optional].
     *
     * Plus: the parameters of Matcher_Points_Base::initialize()
     */
    void initialize(const mrpt::containers::yaml& params) override;

    /// Maximum accepted Euclidean nearest-neighbor distance [m]. This is the
    /// primary inlier/outlier gate: pairings with squared distance beyond
    /// threshold^2 (as widened by thresholdAngularDeg below) are rejected.
    double   threshold           = 0.50;  // m
    /// Optional extra per-point tolerance, growing with the point's distance
    /// from the coordinate origin, added on top of `threshold` (see
    /// implMatchOneLayer() for the exact formula). 0 disables it (pure fixed
    /// threshold). Intended to loosen the distance gate for points that are
    /// far from the sensor, where angular/ranging noise translates into a
    /// larger Euclidean spread.
    double   thresholdAngularDeg = 0;  // deg
    /// How many nearest global neighbors to pair with each local point
    /// (ascending distance order, all within `threshold`). >1 enables
    /// multi-pairing, e.g. for robust kernels downstream that benefit from
    /// several candidate correspondences per point.
    uint32_t pairingsPerPoint    = 1;

   private:
    /** Per-(global layer, local layer) matching: for every local point,
     * queries the global layer's KD-tree/nearest-neighbor structure for its
     * 1 (or pairingsPerPoint) nearest neighbor(s), accepts those within the
     * distance threshold, and (unless already-matched points are allowed)
     * marks both endpoints as paired in `ms` so later matchers in the
     * pipeline skip them. Parallelized over local points via TBB when
     * available. See Matcher_Points_Base for the per-layer iteration this is
     * called from. */
    void implMatchOneLayer(
        const mrpt::maps::CMetricMap& pcGlobal, const mrpt::maps::CPointsMap& pcLocal,
        const mrpt::poses::CPose3D& localPose, MatchState& ms, const layer_name_t& globalName,
        const layer_name_t& localName, Pairings& out) const override;
};

}  // namespace mp2p_icp
