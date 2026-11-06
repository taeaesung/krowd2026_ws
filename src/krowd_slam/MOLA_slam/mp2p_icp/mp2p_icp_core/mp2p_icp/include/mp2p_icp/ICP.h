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
 * @file   ICP.h
 * @brief  Generic ICP algorithm container.
 * @author Jose Luis Blanco Claraco
 * @date   Jun 10, 2019
 *
 * This is the top-level entry point of the whole library: the ICP class
 * drives the classic "iterative closest point" loop (find correspondences,
 * solve for the optimal relative pose, repeat until convergence) while
 * delegating each stage to pluggable, YAML-configurable modules:
 *   - Matcher: finds point/line/plane correspondences between the two clouds
 *     for the current pose estimate (see Matcher.h, not covered here).
 *   - Solver: turns a set of correspondences into an incremental pose update
 *     (Solver_GaussNewton, Solver_OLAE, Solver_Horn - see those headers).
 *   - QualityEvaluator: scores how good the final alignment is (not covered
 *     here, see QualityEvaluator.h).
 * The actual align() loop, iteration/convergence bookkeeping and logging
 * live in ICP.cpp; this header only defines the pipeline container and its
 * configuration/registration API.
 */
#pragma once

#include <mp2p_icp/IterTermReason.h>
#include <mp2p_icp/LogRecord.h>
#include <mp2p_icp/Matcher.h>
#include <mp2p_icp/Parameters.h>
#include <mp2p_icp/QualityEvaluator.h>
#include <mp2p_icp/QualityEvaluator_PairedRatio.h>
#include <mp2p_icp/Results.h>
#include <mp2p_icp/Solver.h>
#include <mp2p_icp/metricmap.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/core/optional_ref.h>
#include <mrpt/math/TPose3D.h>
#include <mrpt/rtti/CObject.h>
#include <mrpt/system/COutputLogger.h>
#include <mrpt/system/CTimeLogger.h>

#include <cstdint>
#include <functional>

namespace mp2p_icp
{
/** Generic ICP algorithm container: builds a custom ICP pipeline by selecting
 * algorithm and parameter for each stage.
 *
 * The main API entry point is align().
 *
 * A convenient way to create an ICP pipeline instance is using a YAML
 * configuration file and calling mp2p_icp::icp_pipeline_from_yaml().
 *
 * \todo Add pipeline picture.
 *
 * Several solvers may exists, but the output from the first one returning
 * `true` will be used. This is by design, to enable different solver algorithms
 * depending on the ICP iteration or the type of geometric entities.
 *
 * \ingroup mp2p_icp_grp
 */
class ICP : public mrpt::system::COutputLogger, public mrpt::rtti::CObject
{
    DEFINE_MRPT_OBJECT(ICP, mp2p_icp)

   public:
    /** Register (align) two point clouds (possibly after having been
     * preprocessed to extract features, etc.) and returns the relative pose of
     * pcLocal with respect to pcGlobal.
     *
     * This runs the full ICP loop: on each iteration, run all configured
     * Matchers to build correspondences ("pairings") between pcLocal (as
     * transformed by the current pose estimate) and pcGlobal, then run the
     * configured Solvers on those pairings to get an updated relative pose,
     * and repeat until a termination criterion in `p` is met (max
     * iterations, small enough pose increment, etc. - see Parameters.h).
     * `initialGuessLocalWrtGlobal` seeds the first iteration; a poor initial
     * guess is the most common cause of ICP converging to a wrong local
     * optimum, since correspondence search only looks at nearby geometry.
     * `prior`, if given, adds a Gaussian prior on the relative pose (e.g.
     * from odometry) that the solver is biased towards. The outcome
     * (converged or not, final pose, quality score, iteration count, ...) is
     * written to `result`; `outputDebugInfo`, if given, additionally records
     * the full per-iteration trace for offline inspection/plotting.
     */
    virtual void align(
        const metric_map_t& pcLocal, const metric_map_t& pcGlobal,
        const mrpt::math::TPose3D& initialGuessLocalWrtGlobal, const Parameters& p, Results& result,
        const std::optional<mrpt::poses::CPose3DPDFGaussianInf>& prior           = std::nullopt,
        const mrpt::optional_ref<LogRecord>&                     outputDebugInfo = std::nullopt);

    /** @name Module: Solver instances
     * @{ */
    using solver_list_t = std::vector<mp2p_icp::Solver::Ptr>;

    /** Create and configure one or more "Solver" modules from YAML-like config
     *   block. Config must be a *sequence* of one or more entries, each with a
     *   `class` and a `params` dictionary entries.
     *
     * Read the comments for ICP on the possible existence of more than one
     * solver.
     *
     * Example:
     *\code
     *- class: mp2p_icp::Solver_Horn
     *  params:
     *   # Parameters depend on the particular class
     *   # none
     *\endcode
     *
     * Alternatively, the objects can be directly created via solvers().
     *
     * \sa mp2p_icp::icp_pipeline_from_yaml()
     */
    void initialize_solvers(const mrpt::containers::yaml& params);

    static void initialize_solvers(const mrpt::containers::yaml& params, ICP::solver_list_t& lst);

    const solver_list_t& solvers() const { return solvers_; }
    solver_list_t&       solvers() { return solvers_; }

    /** Runs a set of solvers in order, returning the result of the first one
     * that successfully produces a solution (returns true). This lets a
     * pipeline configure a primary solver (e.g. OLAE, which is fast but can
     * be degenerate/unobservable for some correspondence geometries, such as
     * all points lying on a single plane) followed by a fallback (e.g.
     * Gauss-Newton) that is only invoked if the first one declines to solve.
     */
    static bool run_solvers(
        const solver_list_t& solvers, const Pairings& pairings, OptimalTF_Result& out,
        const SolverContext& sc = {});

    /** @} */

    /** @name Module: Matcher instances
     * @{ */

    /** Create and configure one or more "Match" modules from YAML-like config
     *block. Config must be a *sequence* of one or more entries, each with a
     *`class` and a `params` dictionary entries.
     *
     * Example:
     *\code
     *- class: mp2p_icp::Matcher_Points_DistanceThreshold
     *  params:
     *   # Parameters depend on the particular class
     *   threshold: 1.0
     *\endcode
     *
     * Alternatively, the objects can be directly created via matchers().
     *
     * \sa mp2p_icp::icp_pipeline_from_yaml()
     */
    void initialize_matchers(const mrpt::containers::yaml& params);

    static void initialize_matchers(const mrpt::containers::yaml& params, matcher_list_t& lst);

    const matcher_list_t& matchers() const { return matchers_; }
    matcher_list_t&       matchers() { return matchers_; }

    /** @} */

    /** @name Module: QualityEvaluator instances
     * @{ */

    /** One configured QualityEvaluator plus its relative weight in the final
     * weighted-average quality score (see evaluate_quality()). */
    struct QualityEvaluatorEntry
    {
        QualityEvaluatorEntry(mp2p_icp::QualityEvaluator::Ptr o, double w)
            : obj(o), relativeWeight(w)
        {
        }

        mp2p_icp::QualityEvaluator::Ptr obj;
        double                          relativeWeight = 1.0;
    };
    using quality_eval_list_t = std::vector<QualityEvaluatorEntry>;

    /** Create and configure one or more  "QualityEvaluator" modules from
     *YAML-like config block. Config must be a *sequence* of one or more
     *entries, each with a `class` and a `params` dictionary entries.
     *
     * Example:
     *\code
     *- class: mp2p_icp::QualityEvaluator_PairedRatio
     *  weight: 1.0  # (Optional if only one quality evaluator is defined)
     *  params:
     *   # Parameters depend on the particular class
     *   # xxx: yyyy
     *\endcode
     *
     * Alternatively, the objects can be directly created via matchers().
     *
     * \sa mp2p_icp::icp_pipeline_from_yaml()
     */
    void initialize_quality_evaluators(const mrpt::containers::yaml& params);

    static void initialize_quality_evaluators(
        const mrpt::containers::yaml& params, quality_eval_list_t& lst);

    const quality_eval_list_t& quality_evaluators() const { return quality_evaluators_; }
    quality_eval_list_t&       quality_evaluators() { return quality_evaluators_; }

    /** Combines all configured QualityEvaluators into a single scalar score
     * (relativeWeight-weighted average) for the final alignment. Used by
     * align() to fill in Results::quality, and by callers to decide whether
     * an ICP result is trustworthy enough to use (e.g. reject a converged
     * but poorly-supported alignment). */
    static double evaluate_quality(
        const quality_eval_list_t& evaluators, const metric_map_t& pcGlobal,
        const metric_map_t& pcLocal, const mrpt::poses::CPose3D& localPose,
        const Pairings& finalPairings);

    /** @} */

    /** Propagates a shared ParameterSource (e.g. for run-time-adjustable
     * parameters driven by external expressions) to every Matcher, Solver,
     * and QualityEvaluator in the pipeline. */
    void attachToParameterSource(ParameterSource& source)
    {
        for (auto& o : matchers()) o->attachToParameterSource(source);
        for (auto& o : solvers()) o->attachToParameterSource(source);
        for (auto& o : quality_evaluators()) o.obj->attachToParameterSource(source);
    }

    /** For whole-ICP overriden classes (e.g. external ICP library wrappers),
     * initialize those external libraries with these parameters.
     * Invoked from mp2p_icp::icp_pipeline_from_yaml().
     *
     * \sa mp2p_icp::icp_pipeline_from_yaml()
     */
    virtual void initialize_derived(  //
        [[maybe_unused]] const mrpt::containers::yaml& p)
    {
        // Default: do nothing
    }

    /** Snapshot of the ICP loop state passed to the optional iteration_hook_t
     * callback after each iteration (correspondences + solved pose so far),
     * so a caller can inspect/log intermediate progress or decide to stop
     * early (see IterationHook_Output::request_stop). Pointers are only
     * valid for the duration of the callback invocation. */
    struct IterationHook_Input
    {
        IterationHook_Input() = default;

        const metric_map_t*     pcGlobal         = nullptr;
        const metric_map_t*     pcLocal          = nullptr;
        const Pairings*         currentPairings  = nullptr;
        const OptimalTF_Result* currentSolution  = nullptr;
        uint32_t                currentIteration = 0;
    };
    /** Return value of the iteration hook: set request_stop=true to make
     * align() terminate the loop early (e.g. from a user-cancel signal or an
     * external convergence check), independently of the built-in
     * termination criteria in Parameters. */
    struct IterationHook_Output
    {
        IterationHook_Output() = default;

        bool request_stop = false;
    };

    using iteration_hook_t = std::function<IterationHook_Output(const IterationHook_Input&)>;

    /** Registers a callback invoked once per ICP iteration; see
     * IterationHook_Input/Output for what it can observe and control. */
    void setIterationHook(const iteration_hook_t& ih) { iteration_hook_ = ih; }

    const mrpt::system::CTimeLogger& profiler() const { return profiler_; }
    mrpt::system::CTimeLogger&       profiler() { return profiler_; }

   protected:
    solver_list_t  solvers_;
    matcher_list_t matchers_;
    // Defaults to a single PairedRatio evaluator so align() always has a
    // quality score to report even if the pipeline config doesn't set one.
    quality_eval_list_t quality_evaluators_ = {{QualityEvaluator_PairedRatio::Create(), 1.0}};

    iteration_hook_t iteration_hook_;

    // Off by default (first ctor arg) to avoid profiling overhead in normal
    // use; enable via profiler().enable() to get per-stage timing stats.
    mrpt::system::CTimeLogger profiler_{false /*disabled*/, "mp2p_icp::ICP"};

    /** Serializes a completed LogRecord to disk if `p` requests it (used for
     * offline debugging of ICP runs, see Parameters::generateDebugFiles). */
    static void save_log_file(const LogRecord& log, const Parameters& p);

    /** Mutable working state threaded through one align() call's iteration
     * loop: which pairings were found and what pose they solved for, so far,
     * plus the iteration counter and an optional pointer to the debug log
     * being filled in. Kept as a separate struct (rather than local
     * variables in align()) so it can be handed to the iteration hook and to
     * helper/derived-class methods without a long parameter list. */
    struct ICP_State
    {
        ICP_State(const metric_map_t& pcsGlobal, const metric_map_t& pcsLocal)
            : pcGlobal(pcsGlobal), pcLocal(pcsLocal)
        {
        }

        const metric_map_t& pcGlobal;
        const metric_map_t& pcLocal;
        Pairings            currentPairings;
        OptimalTF_Result    currentSolution;
        uint32_t            currentIteration = 0;
        LogRecord*          log              = nullptr;
    };

   private:
    // Default ParameterSource used when attachToParameterSource() is never
    // called by the user, so parameter-expression evaluation always has a
    // valid source to bind to.
    ParameterSource ownParamSource_;
};
}  // namespace mp2p_icp
