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
 * @file   errorTerms.h
 * @brief  Per-correspondence-type residual (error) functions and their
 *         Jacobians, used by the Gauss-Newton solver.
 * @author Francisco José Mañas Álvarez, Jose Luis Blanco Claraco
 * @date   Aug 4, 2020
 *
 * Each error_*() function below computes the residual vector of one
 * correspondence (point-point, point-line, point-plane, line-line,
 * plane-plane) as a function of the candidate relative pose, plus
 * (optionally) its Jacobian with respect to the pose. The Jacobian is *not*
 * expressed wrt the 6 SE(3) tangent-space DOF directly: it is wrt the 12
 * "flattened" entries of the pose's 3x3 rotation matrix (column-major) and
 * 3x1 translation (i.e. treating the pose as an over-parameterized 3x4
 * matrix). optimal_tf_gauss_newton.cpp chain-rules this 3x12 (or 4x12)
 * Jacobian through the 12x6 `dDexpe_de` Jacobian (from MRPT's SE(3) Lie
 * group utilities) to get the 6-column Jacobian actually used in the
 * Gauss-Newton normal equations. This two-step approach lets these
 * functions be written directly in terms of the plain rotate+translate
 * point transformation, without having to hand-derive the SE(3)
 * manifold chain rule for each geometric primitive.
 */
#pragma once

#include <mp2p_icp/OptimalTF_Result.h>
#include <mp2p_icp/Pairings.h>
#include <mp2p_icp/WeightParameters.h>
#include <mp2p_icp/optimal_tf_gauss_newton.h>
#include <mrpt/core/optional_ref.h>
#include <mrpt/math/CVectorFixed.h>

#include <Eigen/Dense>

namespace mp2p_icp
{
/** \addtogroup  mp2p_icp_grp
 * @{ */

/** Residual = (relativePose ⊕ local_point) - global_point, i.e. the
 * Euclidean mismatch after transforming the local point into the global
 * frame. The simplest and best-conditioned of the error terms. */
mrpt::math::CVectorFixedDouble<3> error_point2point(
    const mrpt::math::TPoint3Df& local_point, const mrpt::math::TPoint3Df& global_point,
    const mrpt::poses::CPose3D&                                 relativePose,
    mrpt::optional_ref<mrpt::math::CMatrixFixed<double, 3, 12>> jacobian = std::nullopt);

/** Residual = component of (transformed local point - line base point)
 * orthogonal to the line's director vector, i.e. the perpendicular offset of
 * the transformed point from the target line. Degenerate/uninformative
 * along the line's own direction (a point can slide freely along the line
 * without changing this residual), which is why point-to-line alone cannot
 * constrain all 6 DOF and is normally combined with other pairing types. */
mrpt::math::CVectorFixedDouble<3> error_point2line(
    const mp2p_icp::point_line_pair_t& pairing, const mrpt::poses::CPose3D& relativePose,
    mrpt::optional_ref<mrpt::math::CMatrixFixed<double, 3, 12>> jacobian = std::nullopt);

/** Residual = signed distance of the transformed local point to the global
 * plane, projected back along the plane normal (so it's a 3-vector rather
 * than a scalar, matching the convention used by the other error terms).
 * Like point-to-line, uninformative for translation *within* the plane. */
mrpt::math::CVectorFixedDouble<3> error_point2plane(
    const mp2p_icp::point_plane_pair_t& pairing, const mrpt::poses::CPose3D& relativePose,
    mrpt::optional_ref<mrpt::math::CMatrixFixed<double, 3, 12>> jacobian = std::nullopt);

/** Residual between two matched 3D lines. Handles two distinct geometric
 * cases differently (see errorTerms.cpp): near-parallel lines (measured by
 * the angle between them being below a small tolerance) use a single
 * squared point-to-line distance term to avoid the ill-conditioning of the
 * general formula as the lines become parallel; otherwise a 4-component
 * residual (perpendicular distance + 3 director-vector alignment
 * components) is used. */
mrpt::math::CVectorFixedDouble<4> error_line2line(
    const mp2p_icp::matched_line_t& pairing, const mrpt::poses::CPose3D& relativePose,
    mrpt::optional_ref<mrpt::math::CMatrixFixed<double, 4, 12>> jacobian = std::nullopt);

/** Residual between two matched planes' normal vectors only (rotated local
 * normal minus global normal); intentionally insensitive to translation
 * (see the Jacobian's zero translation columns in errorTerms.cpp), since a
 * plane-to-plane pairing alone cannot observe translation along the plane. */
mrpt::math::CVectorFixedDouble<3> error_plane2plane(
    const mp2p_icp::matched_plane_t& pairing, const mrpt::poses::CPose3D& relativePose,
    mrpt::optional_ref<mrpt::math::CMatrixFixed<double, 3, 12>> jacobian = std::nullopt);

}  // namespace mp2p_icp
