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
 * @file   Results.h
 * @brief  Output of ICP::align(): final pose, convergence status, quality.
 *
 * This is the summary handed back to the caller of ICP::align() (as opposed
 * to LogRecord, which optionally captures the full per-iteration trace for
 * offline debugging).
 */
#pragma once

#include <mp2p_icp/Pairings.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>

#include <cstdint>
#include <iosfwd>

#include "IterTermReason.h"

namespace mp2p_icp
{
/** \addtogroup  mp2p_icp_grp
 * @{ */

struct Results
{
    /** The found value (mean + covariance) of the optimal transformation of
     * "local" wrt "global". The covariance is computed by mp2p_icp::covariance()
     * from the geometry of finalPairings, not from ICP's iterative
     * convergence behavior - see covariance.h for what it does and does not
     * capture. */
    mrpt::poses::CPose3DPDFGaussian optimal_tf;

    /** Found scale (if supported by the underlying algorithm) for optimal_tf */
    double optimalScale = 1.0;

    /** The number of executed iterations until convergence.
     * Note: despite the name, this counts iterations actually run before
     * *any* termination reason (converged, stalled, max iterations, error,
     * ...) - check terminationReason to know whether it actually converged. */
    size_t nIterations = 0;

    /** Why the ICP loop stopped; IterTermReason::MaxIterations or
     * IterTermReason::SolverError/NoPairings/QualityCheckpointFailed
     * indicate the result may not represent a good alignment even though a
     * pose value is still populated. */
    IterTermReason terminationReason{IterTermReason::Undefined};

    /** A measure of the 'quality' of the matching, combining all configured
     * QualityEvaluators (see ICP::evaluate_quality()); higher is better
     * (typically normalized so 1.0 ~ perfect match), independent of, and a
     * more direct indicator of alignment goodness than, `terminationReason`
     * alone.
     */
    double quality = 0;

    /** A copy of the pairings found in the last ICP iteration (or from a
     * forced re-match at the final pose if
     * Parameters::force_final_pairings_for_quality is set). */
    Pairings finalPairings;

    void serializeTo(mrpt::serialization::CArchive& out) const;
    void serializeFrom(mrpt::serialization::CArchive& in);

    /** Print all results in human-friendly format to the given output, for
     * example, use `std::cout`.
     */
    void print(std::ostream& o) const;
};

mrpt::serialization::CArchive& operator<<(mrpt::serialization::CArchive& out, const Results& obj);

mrpt::serialization::CArchive& operator>>(mrpt::serialization::CArchive& in, Results& obj);

/** @} */

}  // namespace mp2p_icp
