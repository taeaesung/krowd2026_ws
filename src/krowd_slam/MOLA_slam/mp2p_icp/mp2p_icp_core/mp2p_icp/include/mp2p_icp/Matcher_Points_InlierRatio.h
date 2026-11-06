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
 * @file   Matcher_Points_InlierRatio.h
 * @brief  Pointcloud matcher: fixed ratio of inliers/outliers by distance
 * @author Jose Luis Blanco Claraco
 * @date   June 22, 2020
 *
 * Unlike Matcher_Points_DistanceThreshold (which accepts/rejects based on a
 * fixed metric distance), this matcher always finds a nearest neighbor for
 * every local point and then keeps only the closest `inliersRatio` fraction
 * of them, rejecting the worst (farthest) tail as outliers. This makes it
 * self-scaling to whatever the current alignment error is: useful in early
 * ICP iterations when the pose estimate may still be far off and a fixed
 * distance threshold would either reject almost everything or accept noise.
 */
#pragma once

#include <mp2p_icp/Matcher_Points_Base.h>

namespace mp2p_icp
{
/** Pointcloud matcher: fixed ratio of inliers/outliers by distance
 *
 * Finds point-to-point pairings between the `local` and `global` input metric
 * maps.
 *
 * By default, each `local` point layer is matched against the layer with the
 * same name in the `global` map, unless specified otherwise in the base class
 * member `weight_pt2pt_layers`. Refer to example configuration YAML files for
 * example configurations.
 *
 * \ingroup mp2p_icp_grp
 */
class Matcher_Points_InlierRatio : public Matcher_Points_Base
{
    DEFINE_MRPT_OBJECT(Matcher_Points_InlierRatio, mp2p_icp)

   public:
    Matcher_Points_InlierRatio();
    Matcher_Points_InlierRatio(const double ratio) : Matcher_Points_InlierRatio()
    {
        inliersRatio = ratio;
    }

    /** Parameters:
     * `inliersRatio`: Inliers distance ratio threshold [0-1]
     */
    void initialize(const mrpt::containers::yaml& params) override;

   private:
    /** Inliers distance ratio threshold [0-1]. E.g. 0.80 keeps the 80% of
     * local points whose nearest-neighbor distance is smallest, discarding
     * the farthest 20% regardless of their absolute distance value. */
    double inliersRatio = 0.80;

    /** For every local point, finds its single nearest neighbor in the
     * global layer (no distance gate at search time), sorts all candidate
     * pairings by squared distance, and keeps only the closest
     * `inliersRatio` fraction - see .cpp for the sort/truncate logic. Note
     * this means the matcher's cost and behavior do NOT benefit from an
     * early distance cutoff the way Matcher_Points_DistanceThreshold does:
     * every local point always gets a KD-tree query. */
    void implMatchOneLayer(
        const mrpt::maps::CMetricMap& pcGlobal, const mrpt::maps::CPointsMap& pcLocal,
        const mrpt::poses::CPose3D& localPose, MatchState& ms, const layer_name_t& globalName,
        const layer_name_t& localName, Pairings& out) const override;
};

}  // namespace mp2p_icp
