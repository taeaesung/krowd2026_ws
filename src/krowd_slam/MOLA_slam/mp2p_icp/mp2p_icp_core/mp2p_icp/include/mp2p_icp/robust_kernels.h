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
 * @file   robust_kernels.h
 * @brief  Robust kernel types and functions, for common use in all solvers
 * @author Jose Luis Blanco Claraco
 * @date   Nov 8, 2023
 *
 * Robust kernels down-weight correspondences with a large residual error so
 * that a few bad data associations (wrong matches, dynamic objects, etc.)
 * don't dominate the least-squares solution the way they would under plain
 * (unweighted) least-squares, where the cost grows quadratically with error
 * and outliers can pull the whole solution off. This is essentially one
 * iteration of an Iteratively Reweighted Least Squares (IRLS) scheme: the
 * weight for each pairing is recomputed from its *current* residual, using
 * the previous pose estimate, each ICP iteration.
 */
#pragma once

#include <mrpt/core/bits_math.h>
#include <mrpt/typemeta/TEnumType.h>

#include <cstdint>
#include <functional>

namespace mp2p_icp
{
/** Type to select a robust kernel. Support mrpt::typemeta::TEnumType
 *  to get/set from strings.
 */
enum class RobustKernel : uint8_t
{
    /// None: plain least-squares
    None = 0,

    /// Generalized GemanMcClure kernel (Zhang97ivc, Agarwal15phd).
    GemanMcClure,

    /// Cauchy kernel (Lee2013IROS).
    Cauchy,
};

using robust_sqrt_weight_func_t = std::function<double(double /*errSqr*/)>;

/**
 * Creates a functor with the sqrt of the weight function of a given
 * kernel, or an empty functor if non-robust kernel is selected.
 *
 * Implemented as `inline` to try to make the compiler to optimize.
 *
 * Returning sqrt(w) rather than w itself is a convenience for callers: in
 * (weighted) least-squares, a residual `e` with weight `w` contributes
 * `w*e^2` to the cost, which is equivalent to scaling the residual (and its
 * Jacobian row) by `sqrt(w)` before feeding it into an unweighted
 * normal-equations solve - so solvers can just multiply by this returned
 * value directly instead of computing sqrt() themselves at every use.
 *
 * @param kernel Selected kernel type.
 * @param kernelParam Parameter of the kernel. Roughly, the residual
 * magnitude ("c" in the formulas below) beyond which a correspondence starts
 * being significantly down-weighted; needs to be tuned to the expected
 * inlier noise level.
 * @return A functor.
 */
inline robust_sqrt_weight_func_t create_robust_kernel(
    const RobustKernel kernel, const double kernelParam)
{
    const double kernelParamSqr = mrpt::square(kernelParam);

    switch (kernel)
    {
        case RobustKernel::None:
            return {};  // empty
            break;

        case RobustKernel::GemanMcClure:
            /**
             * We must return the sqrt of the weight function:
             *
             *   sqrt(w(x))=( ∂ρ(x)/∂x )/x = c²/(e²+c)²
             *
             * with the loss function ρ(x) = (x²/2)/(c²+x²)
             */
            return [kernelParamSqr, kernelParam](double errorSqr) -> double
            { return (kernelParamSqr) / mrpt::square(errorSqr + kernelParam); };

        case RobustKernel::Cauchy:
            /**
             * We must return the sqrt of the weight function:
             *
             *   sqrt(w(x))=( ∂ρ(x)/∂x )/x = c²/(e²+c²)
             *
             * with the loss function ρ(x) = 0.5 c² log(1+x²/c²)
             *
             */
            return [kernelParamSqr](double errorSqr) -> double
            { return (kernelParamSqr) / (errorSqr + kernelParamSqr); };

        default:
            throw std::invalid_argument("Unknown kernel type");
    };
};

}  // namespace mp2p_icp

// This allows reading/writing the enum type to strings, e.g. in YAML files.
MRPT_ENUM_TYPE_BEGIN_NAMESPACE(mp2p_icp, mp2p_icp::RobustKernel)
MRPT_FILL_ENUM(RobustKernel::None);
MRPT_FILL_ENUM(RobustKernel::GemanMcClure);
MRPT_FILL_ENUM(RobustKernel::Cauchy);
MRPT_ENUM_TYPE_END()
