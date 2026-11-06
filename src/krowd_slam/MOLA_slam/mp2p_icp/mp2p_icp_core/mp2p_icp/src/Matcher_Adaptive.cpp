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
 * @file   Matcher_Adaptive.cpp
 * @brief  Pointcloud matcher: smart adaptive matcher
 * @author Jose Luis Blanco Claraco
 * @date   Nov 11, 2023
 *
 * See the header for the high-level two-pass algorithm description
 * (gather candidates + histogram -> adaptive threshold -> plane-fit-or-
 * point-pairing decision per local point). Comments below walk through the
 * implementation of each stage.
 */

#include <mp2p_icp/Matcher_Adaptive.h>
#include <mp2p_icp/estimate_points_eigen.h>
#include <mrpt/core/exceptions.h>
#include <mrpt/core/round.h>
#include <mrpt/math/CHistogram.h>  // CHistogram
#include <mrpt/math/distributions.h>  // confidenceIntervalsFromHistogram()

IMPLEMENTS_MRPT_OBJECT(Matcher_Adaptive, Matcher, mp2p_icp)

using namespace mp2p_icp;

void Matcher_Adaptive::initialize(const mrpt::containers::yaml& params)
{
    Matcher_Points_Base::initialize(params);

    MCP_LOAD_REQ(params, confidenceInterval);
    MCP_LOAD_REQ(params, firstToSecondDistanceMax);

    MCP_LOAD_REQ(params, absoluteMaxSearchDistance);
    MCP_LOAD_OPT(params, minimumCorrDist);

    MCP_LOAD_REQ(params, enableDetectPlanes);

    MCP_LOAD_OPT(params, planeSearchPoints);
    MCP_LOAD_OPT(params, planeMinimumFoundPoints);
    MCP_LOAD_OPT(params, planeEigenThreshold);
    MCP_LOAD_OPT(params, maxPt2PtCorrespondences);
    MCP_LOAD_OPT(params, planeMinimumDistance);

    ASSERT_LT_(confidenceInterval, 1.0);
    ASSERT_GT_(confidenceInterval, 0.0);

    ASSERT_GE_(planeSearchPoints, planeMinimumFoundPoints);
    ASSERT_GE_(planeMinimumFoundPoints, 3);

    ASSERT_GT_(planeEigenThreshold, 0.0);
}

void Matcher_Adaptive::implMatchOneLayer(
    const mrpt::maps::CMetricMap& pcGlobalMap, const mrpt::maps::CPointsMap& pcLocal,
    const mrpt::poses::CPose3D& localPose, MatchState& ms, const layer_name_t& globalName,
    const layer_name_t& localName, Pairings& out) const
{
    MRPT_START

    const mrpt::maps::NearestNeighborsCapable& nnGlobal =
        *mp2p_icp::MapToNN(pcGlobalMap, true /*throw if cannot convert*/);

    out.potential_pairings += pcLocal.size() * maxPt2PtCorrespondences;

    // Empty maps?  Nothing to do
    if (pcGlobalMap.isEmpty() || pcLocal.empty())
    {
        return;
    }

    const TransformedLocalPointCloud tl = transform_local_to_global(pcLocal, localPose);

    // Try to do matching only if the bounding boxes have some overlap:
    if (!pcGlobalMap.boundingBox().intersection(
            {tl.localMin, tl.localMax},
            /* threshold?? */ +bounding_box_intersection_check_epsilon_))
    {
        return;
    }

    // Prepare output: no correspondences initially:
    out.paired_pt2pt.reserve(out.paired_pt2pt.size() + pcLocal.size());

    // Loop for each point in local map:
    // --------------------------------------------------
    const float absoluteMaxDistSqr = mrpt::square(absoluteMaxSearchDistance);

    const auto& lxs = pcLocal.getPointsBufferRef_x();
    const auto& lys = pcLocal.getPointsBufferRef_y();
    const auto& lzs = pcLocal.getPointsBufferRef_z();

    // In order to find the closest association for each global point, we must
    // first build this temporary list of *potential* associations, indexed by
    // global point indices, and sorted by errSqr:

    // List of all 1st and 2nd closest pairings to each
    // local point (Pass 1 output; consumed by Pass 2 below).
    matchesPerLocal_.clear();
    matchesPerLocal_.resize(tl.x_locals.size());

    // Calculate limits for the histogram:
    // Running min/max of the 1st/2nd-nearest-neighbor squared errors seen so
    // far, used to size the CHistogram range further down.
    std::optional<float> minSqrErrorForHistogram;
    std::optional<float> maxSqrErrorForHistogram;

    // Buffers a candidate correspondence for a given local point (capped at
    // MAX_CORRS_PER_LOCAL to bound memory); final accept/reject happens later
    // in Pass 2, once the adaptive threshold is known.
    const auto lambdaAddPair = [&](const size_t localIdx, const mrpt::math::TPoint3Df& globalPt,
                                   const uint64_t globalIdxOrID, const float errSqr)
    {
        auto& ps = matchesPerLocal_.at(localIdx);
        if (ps.size() >= MAX_CORRS_PER_LOCAL) return;

        mrpt::tfest::TMatchingPair p;
        p.globalIdx                      = globalIdxOrID;
        p.localIdx                       = localIdx;
        p.global                         = globalPt;
        p.local                          = {lxs[localIdx], lys[localIdx], lzs[localIdx]};
        p.errorSquareAfterTransformation = errSqr;

        ps.push_back(p);
    };

    // Plane fitting needs a whole local neighborhood (planeSearchPoints);
    // plain point-to-point only needs as many candidates as we might keep.
    const uint32_t nn_search_max_points =
        enableDetectPlanes ? planeSearchPoints : maxPt2PtCorrespondences;

    for (size_t i = 0; i < tl.x_locals.size(); i++)
    {
        const size_t localIdx = tl.idxs.has_value() ? (*tl.idxs)[i] : i;

        if (!allowMatchAlreadyMatchedPoints_ &&
            ms.localPairedBitField.point_layers.at(localName)[localIdx])
        {
            // skip, already paired, e.g. by another Matcher in the pipeline
            // before me:
            continue;
        }

        const float lx = tl.x_locals[i], ly = tl.y_locals[i], lz = tl.z_locals[i];

        // Use a KD-tree to look for the nearnest neighbor(s) of
        // (x_local, y_local, z_local) in the global map:
        if (nn_search_max_points == 1)
        {
            neighborSqrDists_.resize(1);
            neighborIndices_.resize(1);
            neighborPts_.resize(1);

            if (!nnGlobal.nn_single_search(
                    {lx, ly, lz},  // Look closest to this guy
                    neighborPts_[0], neighborSqrDists_[0], neighborIndices_[0]))
            {
                neighborPts_.clear();
                neighborSqrDists_.clear();
                neighborIndices_.clear();
            }
        }
        else
        {
            nnGlobal.nn_radius_search(
                {lx, ly, lz},  // Look closest to this guy
                absoluteMaxDistSqr, neighborPts_, neighborSqrDists_, neighborIndices_,
                nn_search_max_points);
        }

        for (size_t k = 0; k < neighborIndices_.size(); k++)
        {
            const auto tentativeErrSqr = neighborSqrDists_.at(k);

            if (tentativeErrSqr > absoluteMaxDistSqr) continue;

            if (k <= 1)
            {
                // keep max of 1st and 2nd closest point errors for the
                // histogram:
                // Only the 1st/2nd-nearest errors feed the histogram (not
                // all K candidates): these two are the ones whose
                // distribution across the whole cloud is most informative
                // about "how far is a genuine inlier currently landing",
                // which is exactly what the adaptive threshold needs to
                // capture.
                if (maxSqrErrorForHistogram)
                {
                    mrpt::keep_max(*maxSqrErrorForHistogram, tentativeErrSqr);
                    mrpt::keep_min(*minSqrErrorForHistogram, tentativeErrSqr);
                }
                else
                {
                    maxSqrErrorForHistogram = tentativeErrSqr;
                    minSqrErrorForHistogram = tentativeErrSqr;
                }
            }

            lambdaAddPair(localIdx, neighborPts_.at(k), neighborIndices_.at(k), tentativeErrSqr);
        }

    }  // For each local point

    // --- Pass 2: derive the adaptive threshold, then decide pairings ---
    // Now, estimate the probability distribution (histogram) of the
    // 1st/2nd points:
    mrpt::math::CHistogram hist(*minSqrErrorForHistogram, *maxSqrErrorForHistogram, 50);

    for (const auto& mspl : matchesPerLocal_)
        for (size_t i = 0; i < std::min<size_t>(mspl.size(), 2UL); i++)
            hist.add(mspl[i].errorSquareAfterTransformation);

    hist.getHistogramNormalized(histXs_, histValues_);

    // ci_high: the upper bound of the central `confidenceInterval` fraction
    // of the observed 1st/2nd-neighbor error distribution. E.g. with
    // confidenceInterval=0.80, ci_high is (approximately) the 90th
    // percentile of squared errors - used below as this iteration's
    // effective inlier threshold instead of a fixed, hand-picked value.
    double ci_low = 0, ci_high = 0;
    mrpt::math::confidenceIntervalsFromHistogram(
        histXs_, histValues_, ci_low, ci_high, 1.0 - confidenceInterval);

#if 0
    printf("Histograms:\n");
    for (auto v : histXs_) printf("%.02f ", v);
    printf("\n");
    for (auto v : histValues_) printf("%.02f ", v);
    printf("\n");
    printf(
        "[MatcherAdaptive] CI_HIGH: %.03f => threshold=%.03f m nCorrs=%zu\n",
        ci_high, std::sqrt(ci_high), matchesPerLocal_.size());
#endif

    // Take the confidence interval limit as the definitive maximum squared
    // distance for correspondences:
    // (bounded below by minimumCorrDist so a tight/degenerate distribution
    // never yields a near-zero acceptance radius)
    const double maxCorrDistSqr = std::max(mrpt::square(minimumCorrDist), ci_high);

    const float maxSqr1to2 = mrpt::square(firstToSecondDistanceMax);

    // Now, process candidates pairing and store them in `out.paired_pt2pt`:
    for (const auto& mspl : matchesPerLocal_)
    {
        // Check for a potential plane?
        // minimum: 3 points to be able to fit a plane
        if (enableDetectPlanes && mspl.size() >= planeMinimumFoundPoints)
        {
            kddXs.clear();
            kddYs.clear();
            kddZs.clear();
            for (const auto& p : mspl)
            {
                kddXs.push_back(p.global.x);
                kddYs.push_back(p.global.y);
                kddZs.push_back(p.global.z);
            }

            // PCA over this local point's global-map neighborhood: for a
            // perfectly flat (planar) neighborhood, the covariance has one
            // near-zero eigenvalue (variance across the plane, i.e. along
            // its normal) and two larger, roughly-equal eigenvalues
            // (variance within the plane). eigVals are ascending, so
            // eigVals[0] is that "flatness" eigenvalue and eigVectors[0] is
            // the corresponding (normal) direction.
            const PointCloudEigen& eig = mp2p_icp::estimate_points_eigen(
                kddXs.data(), kddYs.data(), kddZs.data(), std::nullopt, kddXs.size());

            // e0/e2 must be < planeEigenThreshold:
            // Planarity test: smallest eigenvalue must be a small fraction
            // (planeEigenThreshold) of *both* larger ones - i.e. flat in
            // one direction but well-spread in the other two, ruling out
            // degenerate cases like a near-1D line/edge (which would also
            // have one small eigenvalue but so would the other "in-plane"
            // one) or a small/noisy cluster with no clear structure.
            if (eig.eigVals[0] < planeEigenThreshold * eig.eigVals[2] &&
                eig.eigVals[0] < planeEigenThreshold * eig.eigVals[1])
            {
                const auto&                normal        = eig.eigVectors[0];
                const mrpt::math::TPoint3D planeCentroid = {
                    eig.meanCov.mean.x(), eig.meanCov.mean.y(), eig.meanCov.mean.z()};

                const auto   thePlane    = mrpt::math::TPlane(planeCentroid, normal);
                const double ptPlaneDist = std::abs(thePlane.distance(mspl.at(0).local));

                // Beyond the neighborhood being flat, the *local* point
                // itself must actually lie close to the fitted plane
                // (guards against the local point being an outlier relative
                // to a genuinely planar - but displaced - neighborhood).
                if (ptPlaneDist < planeMinimumDistance)
                {
                    const auto localIdx = mspl.at(0).localIdx;

                    // OK, all conditions pass: add the new pairing:
                    auto& p              = out.paired_pt2pl.emplace_back();
                    p.pt_local           = {lxs[localIdx], lys[localIdx], lzs[localIdx]};
                    p.pl_global.centroid = planeCentroid;

                    p.pl_global.plane = thePlane;

                    // Mark local point as already paired:
                    ms.localPairedBitField.point_layers[localName].mark_as_set(localIdx);

                    // all good with this local point:
                    // a plane pairing was emitted, so skip the
                    // point-to-point fallback below for this local point.
                    continue;
                }
            }
        }

        // Fallback (or enableDetectPlanes==false): emit point-to-point
        // pairing(s) from the candidates gathered in Pass 1, which are
        // implicitly in ascending-distance order (nn_single_search/
        // nn_radius_search return neighbors nearest-first).
        for (size_t i = 0; i < std::min<size_t>(mspl.size(), maxPt2PtCorrespondences); i++)
        {
            const auto& p         = mspl.at(i);
            const auto  globalIdx = p.globalIdx;

            if (!allowMatchAlreadyMatchedGlobalPoints_ &&
                ms.globalPairedBitField.point_layers.at(globalName)[globalIdx])
                continue;  // skip, global point already paired.

            // too large error for the adaptive threshold?
            if (p.errorSquareAfterTransformation >= maxCorrDistSqr) continue;

            // Beyond the first (closest) candidate, only accept further
            // ones if they're not much worse than the closest - i.e. all
            // kept candidates for this local point must be near-tied in
            // distance, not just "within the adaptive threshold". Since
            // candidates are distance-sorted, the first failure here means
            // all later ones fail too, so it's safe to stop (`break`).
            if (i != 0 && mspl[i].errorSquareAfterTransformation >
                              mspl[0].errorSquareAfterTransformation * maxSqr1to2)
            {
                break;
            }

            out.paired_pt2pt.emplace_back(p);

            // Mark local & global points as already paired:
            if (!allowMatchAlreadyMatchedGlobalPoints_)
            {
                const auto localIdx = p.localIdx;
                ms.localPairedBitField.point_layers[localName].mark_as_set(localIdx);
            }
        }
    }

    // Update adaptive maximum range:
    // absoluteMaxSearchDistance = std::sqrt(ci_high) + minimumCorrDist;

// Now, mark global idxs as used. It's done after the loop above
// to allow multiple local -> global pairs with the same global.
#if 0
                // Mark local & global points as already paired:
            ms.globalPairedBitField.point_layers[globalName].mark_as_set(
                globalIdx);
#endif

    MRPT_END
}
