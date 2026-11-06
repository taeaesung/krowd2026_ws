/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2026 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
 Closed-source licenses available upon request, for this odometry package
 alone or in combination with the complete SLAM system.
*/

/**
 * @file   LidarOdometry.cpp
 * @brief  Main C++ class exposing LIDAR odometry
 * @author Jose Luis Blanco Claraco
 * @date   Sep 16, 2023
 *
 * Implements LidarOdometry::initialize_frontend(): the one-time, YAML-driven
 * setup that runs before any sensor data is processed. It parses the
 * `Parameters` struct (delegating most sub-sections to their own
 * `initialize(const Yaml&)`, see LidarOdometry_Parameters.cpp), builds the
 * mp2p_icp generator/filter pipelines and the two ICP instances (with vs.
 * without a motion-model prior, see AlignKind) declared in the YAML, attaches
 * them all to the shared `state_.parameter_source` for dynamic-variable
 * support, locates the required NavStateFilter module (and optional
 * SharedKeyframeMap sink), and optionally preloads an existing local/simple
 * map (doPreloadLocalMap()) for multi-session SLAM or localization-only runs.
 */

// This module:
#include <mola_lidar_odometry/LidarOdometry.h>

// MOLA:
#include <mola_yaml/yaml_helpers.h>

// MP2P_ICP:
#include <mp2p_icp/IcpPrepareCapable.h>
#include <mp2p_icp/icp_pipeline_from_yaml.h>
#include <mrpt/system/filesystem.h>

// Std:
#include <sstream>

namespace mola
{

namespace
{
// Builds one ICP_case (an mp2p_icp::ICP instance plus its parsed Parameters)
// from a YAML block, via the mp2p_icp library's own YAML pipeline loader.
// Used twice below: once for "icp_settings_with_vel" (AlignKind::RegularOdometry)
// and once for "icp_settings_without_vel" (AlignKind::NoMotionModel).
void load_icp_set_of_params(
  LidarOdometry::Parameters::ICP_case & out, const mrpt::containers::yaml & cfg)
{
  const auto [icp, params] = mp2p_icp::icp_pipeline_from_yaml(cfg);

  out.icp = icp;
  out.icp_parameters = params;
}
}  // namespace

// See file-level comment above for an overview of what this does.
void LidarOdometry::initialize_frontend(const Yaml & c)
{
  MRPT_TRY_START

  this->setLoggerName("LidarOdometry");

  // make a copy of the initialization, for use in reset()
  lastInitConfig_ = c;

  {
    auto lckState = mrpt::lockHelper(state_mtx_);
    // This block builds the generators/pipelines and attaches them to the
    // parameter source, all of which the IMU worker thread reaches without
    // state_mtx_ (sensors may already be feeding by now):
    auto lckImu = mrpt::lockHelper(imu_state_mtx_);

    // One-shot deprecation warning for the legacy *_sensor_* names. Aliases
    // are still honored (dynamic variables are double-published; the *_sensor_*
    // YAML param keys fall back via cfg.getOrDefault below), so old pipelines
    // keep working, but we nudge users to migrate. Scope-limited substring scan
    // over the serialized config covers embedded pipeline blocks
    // (observations_deskew_pass, localmap_generator, etc.) without parsing
    // them again.
    {
      std::stringstream ss;
      ss << c;
      const std::string serialized = ss.str();
      const auto found = [&](const char * needle) {
        return serialized.find(needle) != std::string::npos;
      };
      if (
        found("ESTIMATED_SENSOR_MAX_RANGE") || found("INSTANTANEOUS_SENSOR_MAX_RANGE") ||
        found("max_sensor_range_filter_coefficient") || found("absolute_minimum_sensor_range")) {
        MRPT_LOG_WARN(
          "Your YAML pipeline references one or more deprecated *_sensor_* "
          "names (ESTIMATED_SENSOR_MAX_RANGE, INSTANTANEOUS_SENSOR_MAX_RANGE, "
          "max_sensor_range_filter_coefficient, absolute_minimum_sensor_range). "
          "They are aliases of ESTIMATED_OBSERVATION_RADIUS, "
          "INSTANTANEOUS_OBSERVATION_RADIUS, observation_radius_filter_coefficient, "
          "absolute_minimum_observation_radius respectively (all measured from "
          "base_link, not from the sensor) and will be removed in a future "
          "release. Please rename references in your pipeline files.");
      }
    }

    // Load params:
    const auto cfg = c["params"];
    MRPT_LOG_DEBUG_STREAM("Loading these params:\n" << cfg);

    if (auto pipelineName = cfg.getOrDefault<std::string>("pipeline_name", "");
        !pipelineName.empty()) {
      MRPT_LOG_INFO_FMT(
        "\n"
        "┌──────────────────────────────────────────────────────────────┐\n"
        "│              USING THIS LIDAR-ODOMETRY PIPELINE              │\n"
        "├──────────────────────────────────────────────────────────────┤\n"
        "│  %59s │\n"
        "└──────────────────────────────────────────────────────────────┘\n",
        pipelineName.c_str());
    }

    ENSURE_YAML_ENTRY_EXISTS(cfg, "lidar_sensor_labels");
    if (cfg["lidar_sensor_labels"].isSequence()) {
      const auto lsl = cfg["lidar_sensor_labels"].asSequenceRange();
      for (const auto & sl : lsl) {
        const auto s = sl.as<std::string>();
        MRPT_LOG_DEBUG_STREAM("Adding as input lidar sensor label: " << s);
        params_.lidar_sensor_labels.emplace_back(s);
      }
    } else {
      ASSERT_(cfg["lidar_sensor_labels"].isScalar());
      const auto s = cfg["lidar_sensor_labels"].as<std::string>();
      MRPT_LOG_DEBUG_STREAM("Adding as input lidar sensor label: " << s);
      params_.lidar_sensor_labels.emplace_back(s);
    }
    ASSERT_(!params_.lidar_sensor_labels.empty());

    // Obs2map merge pipeline:
    ASSERT_(c["insert_observation_into_local_map"].isSequence());
    // Create, and copy my own verbosity level:
    state_.obs2map_merge = mp2p_icp_filters::filter_pipeline_from_yaml(
      c["insert_observation_into_local_map"], this->getMinLoggingLevel());

    // Attach to the parameter source for dynamic parameters:
    mp2p_icp::AttachToParameterSource(state_.obs2map_merge, state_.parameter_source);

    ASSERT_(!state_.obs2map_merge.empty());

    // Deskew for visualization in fallback mode (see gicp.yaml comments)
    // If not specified, the "raw" cloud will be shown instead of the de-skewed one.
    if (c.has("observations_filter_deskew_for_visualization")) {
      ASSERT_(c["observations_filter_deskew_for_visualization"].isSequence());
      // Create, and copy my own verbosity level:
      state_.obsDeskewForViz = mp2p_icp_filters::filter_pipeline_from_yaml(
        c["observations_filter_deskew_for_visualization"], this->getMinLoggingLevel());

      // Attach to the parameter source for dynamic parameters and IMU-velocities:
      mp2p_icp::AttachToParameterSource(state_.obsDeskewForViz, state_.parameter_source);

      ASSERT_(!state_.obsDeskewForViz.empty());
    }

    // Other sensors:
    if (cfg.has("imu_sensor_label")) {
      params_.imu_sensor_label = cfg["imu_sensor_label"].as<std::string>();
    }

    if (cfg.has("gnss_sensor_label")) {
      params_.gnss_sensor_label = cfg["gnss_sensor_label"].as<std::string>();
    }

    ASSERT_(cfg.has("local_map_updates"));
    params_.local_map_updates.initialize(cfg["local_map_updates"], params_);

    if (cfg.has("multiple_lidars")) {
      params_.multiple_lidars.initialize(cfg["multiple_lidars"], params_);
    }

    // this one is std::optional
    {
      const std::string key = "write_debug_icp_log_if_quality_under";
      if (cfg.has(key) && !cfg[key].isNullNode() && !cfg[key].as<std::string>().empty()) {
        params_.write_debug_icp_log_if_quality_under.emplace(cfg[key].as<double>());
      }
    }

    YAML_LOAD_OPT(params_, min_time_between_scans, double);
    YAML_LOAD_REQ(params_, min_icp_goodness, double);
    YAML_LOAD_OPT(params_, max_accepted_pose_step_base, double);
    YAML_LOAD_OPT(params_, max_accepted_linear_speed, double);
    YAML_LOAD_OPT(params_, max_accepted_rotation_step_base_deg, double);
    YAML_LOAD_OPT(params_, max_accepted_angular_speed_deg_s, double);
    ASSERT_GE_(params_.max_accepted_pose_step_base, .0);
    ASSERT_GE_(params_.max_accepted_linear_speed, .0);
    ASSERT_GE_(params_.max_accepted_rotation_step_base_deg, .0);
    ASSERT_GE_(params_.max_accepted_angular_speed_deg_s, .0);
    // Accept the deprecated *_sensor_* keys as fallbacks before reading the new
    // canonical names, so a YAML that only sets the old key keeps working and a
    // YAML that sets both lets the new key win.
    params_.observation_radius_filter_coefficient = cfg.getOrDefault<double>(
      "max_sensor_range_filter_coefficient", params_.observation_radius_filter_coefficient);
    params_.absolute_minimum_observation_radius = cfg.getOrDefault<double>(
      "absolute_minimum_sensor_range", params_.absolute_minimum_observation_radius);
    YAML_LOAD_OPT(params_, observation_radius_filter_coefficient, double);
    YAML_LOAD_OPT(params_, absolute_minimum_observation_radius, double);
    YAML_LOAD_OPT(params_, start_active, bool);

    YAML_LOAD_OPT(params_, max_lidar_queue_before_drop, uint32_t);
    YAML_LOAD_OPT(params_, gnss_queue_max_size, uint32_t);
    YAML_LOAD_OPT(params_, min_motion_model_xyz_cov_inv, double);

    YAML_LOAD_OPT(params_, optimize_twist, bool);
    YAML_LOAD_OPT(params_, optimize_twist_rerun_min_trans, double);
    YAML_LOAD_OPT(params_, optimize_twist_rerun_min_rot_deg, double);
    YAML_LOAD_OPT(params_, optimize_twist_max_corrections, size_t);

    YAML_LOAD_OPT(params_, publish_reference_frame, std::string);
    YAML_LOAD_OPT(params_, publish_vehicle_frame, std::string);
    YAML_LOAD_OPT(params_, publish_deskewed_scans, bool);

    if (cfg.has("adaptive_threshold")) {
      params_.adaptive_threshold.initialize(cfg["adaptive_threshold"]);
    }

    if (cfg.has("diagnostics")) {
      params_.diagnostics.initialize(cfg["diagnostics"]);
    }

    if (cfg.has("visualization")) {
      params_.visualization.initialize(cfg["visualization"]);
    }

    YAML_LOAD_OPT(params_, pipeline_profiler_enabled, bool);
    YAML_LOAD_OPT(params_, icp_profiler_enabled, bool);
    YAML_LOAD_OPT(params_, icp_profiler_full_history, bool);

    if (cfg.has("simplemap")) {
      params_.simplemap.initialize(cfg["simplemap"], params_);
    }

    if (cfg.has("estimated_trajectory")) {
      params_.estimated_trajectory.initialize(cfg["estimated_trajectory"]);
    }

    if (cfg.has("debug_traces")) {
      params_.debug_traces.initialize(cfg["debug_traces"]);
    }

    if (cfg.has("observation_validity_checks")) {
      params_.observation_validity_checks.initialize(cfg["observation_validity_checks"]);
    }

    if (cfg.has("imu_gravity_correction")) {
      params_.imu_gravity_correction.initialize(cfg["imu_gravity_correction"]);

#if defined(MOLA_LO_HAS_MAP_GRAVITY_ESTIMATOR)
      if (params_.imu_gravity_correction.map_gravity.enabled) {
        state_.map_gravity.estimator.parameters.load_from(
          params_.imu_gravity_correction.map_gravity.estimator_params);
        if (params_.imu_gravity_correction.map_gravity.log_only) {
          MRPT_LOG_INFO(
            "imu_gravity_correction.map_gravity enabled in LOG-ONLY mode: the "
            "estimate is computed and logged but does not reach the verticality "
            "reference, so the trajectory matches that of a disabled run.");
        } else {
          MRPT_LOG_INFO(
            "imu_gravity_correction.map_gravity enabled: the verticality "
            "reference will be estimated online instead of frozen at the first "
            "keyframe.");
        }
      }
#else
      if (params_.imu_gravity_correction.map_gravity.enabled) {
        MRPT_LOG_WARN(
          "imu_gravity_correction.map_gravity is enabled but this build "
          "lacks mola::imu::MapGravityEstimator; the verticality reference "
          "will remain frozen at the first keyframe.");
      }
#endif
    }

#if !defined(MOLA_LO_HAS_MP2P_GRAVITY_PRIOR)
    // Built against an mp2p_icp without the rank-2 gravity prior: degrade to
    // the legacy path instead of failing, so older setups keep working.
    if (params_.imu_gravity_correction.enabled && params_.imu_gravity_correction.use_rank2_prior) {
      MRPT_LOG_WARN(
        "imu_gravity_correction.use_rank2_prior requires a newer mp2p_icp "
        "providing mp2p_icp::GravityPrior; falling back to the legacy "
        "pose-prior path.");
      params_.imu_gravity_correction.use_rank2_prior = false;
    }
#endif

    if (c.has("initial_localization")) {
      params_.initial_localization.initialize(c["initial_localization"]);
    }

    // Watch for legacy (mola_lidar_odometry version <0.5.0) organization:
    if (c.has("navstate_fuse_params")) {
      THROW_EXCEPTION(
        "It seems you are using a legacy mola_lo pipeline config file. Please, refer to release "
        "notes for mola_lidar_odometry 0.5.0");
    }

    ENSURE_YAML_ENTRY_EXISTS(c, "icp_settings_with_vel");
    load_icp_set_of_params(params_.icp[AlignKind::RegularOdometry], c["icp_settings_with_vel"]);

    if (c.has("icp_settings_without_vel")) {
      load_icp_set_of_params(params_.icp[AlignKind::NoMotionModel], c["icp_settings_without_vel"]);
    } else {
      // Default: use the regular ICP settings:
      params_.icp[AlignKind::NoMotionModel] = params_.icp[AlignKind::RegularOdometry];
    }

    for (auto & [kind, icpCase] : params_.icp) {
      icpCase.icp->profiler().enable(params_.icp_profiler_enabled);
      icpCase.icp->profiler().enableKeepWholeHistory(params_.icp_profiler_full_history);

      // Attach all ICP instances to the parameter source for dynamic
      // parameters:
      icpCase.icp->attachToParameterSource(state_.parameter_source);

      // Attach final filter pipeline:
      // (mostly to save space & CPU when logging to disk)
      icpCase.icp_parameters.functor_before_logging_local = [this](mp2p_icp::metric_map_t & m) {
        const ProfilerEntry tle(profiler_, "icp_functor_before_logging");

        mp2p_icp_filters::apply_filter_pipeline(state_.pc_filter3, m, profiler_);
      };
    }
    // system-wide profiler:
    profiler_.enable(params_.pipeline_profiler_enabled);

    // Create lidar segmentation algorithm:
    {
      const ProfilerEntry tle(profiler_, "filterPointCloud_initialize");

      // Observation -> map generator:
      if (c.has("observations_generator") && !c["observations_generator"].isNullNode()) {
        // Create, and copy my own verbosity level:
        state_.obs_generators = mp2p_icp_filters::generators_from_yaml(
          c["observations_generator"], this->getMinLoggingLevel());
      } else {
        MRPT_LOG_WARN(
          "Using default mp2p_icp_filters::Generator for "
          "observations since no YAML 'observations_generator' entry "
          "was given");

        auto defaultGen = mp2p_icp_filters::Generator::Create();
        defaultGen->initialize({});
        state_.obs_generators.push_back(defaultGen);
      }

      // Attach to the parameter source for dynamic parameters:
      mp2p_icp::AttachToParameterSource(state_.obs_generators, state_.parameter_source);

      if (c.has("observations_filter_adjust_timestamps")) {
        // Create, and copy my own verbosity level:
        state_.pc_filterAdjustTimes = mp2p_icp_filters::filter_pipeline_from_yaml(
          c["observations_filter_adjust_timestamps"], this->getMinLoggingLevel());

        // Attach to the parameter source for dynamic parameters:
        mp2p_icp::AttachToParameterSource(state_.pc_filterAdjustTimes, state_.parameter_source);
      } else {
        MRPT_LOG_WARN(
          "No YAML entry 'observations_filter_adjust_timestamps', this "
          "filter stage will have no effect.");
      }

      if (c.has("observations_prefilter_file")) {
        if (const auto prefilterFile = c["observations_prefilter_file"].as<std::string>();
            !prefilterFile.empty()) {
          ASSERT_FILE_EXISTS_(prefilterFile);

          const auto prefilterYaml = mrpt::containers::yaml::FromFile(prefilterFile);

          // Create, and copy my own verbosity level:
          state_.pc_prefilter =
            mp2p_icp_filters::filter_pipeline_from_yaml(prefilterYaml, this->getMinLoggingLevel());

          // Attach to the parameter source for dynamic parameters:
          mp2p_icp::AttachToParameterSource(state_.pc_prefilter, state_.parameter_source);
        }
      }

      // Early deskew pass:
      if (c.has("observations_deskew_pass")) {
        state_.pc_deskew = mp2p_icp_filters::filter_pipeline_from_yaml(
          c["observations_deskew_pass"], this->getMinLoggingLevel());
        mp2p_icp::AttachToParameterSource(state_.pc_deskew, state_.parameter_source);
      } else {
        MRPT_LOG_WARN(
          "No YAML entry 'observations_deskew_pass', this "
          "filter stage will have no effect.");
      }

      if (c.has("observations_filter_1st_pass")) {
        // Create, and copy my own verbosity level:
        state_.pc_filter1 = mp2p_icp_filters::filter_pipeline_from_yaml(
          c["observations_filter_1st_pass"], this->getMinLoggingLevel());

        // Attach to the parameter source for dynamic parameters:
        mp2p_icp::AttachToParameterSource(state_.pc_filter1, state_.parameter_source);
      } else {
        MRPT_LOG_WARN(
          "No YAML entry 'observations_filter_1st_pass', this "
          "filter stage will have no effect.");
      }

      if (c.has("observations_filter_2nd_pass")) {
        // Create, and copy my own verbosity level:
        state_.pc_filter2 = mp2p_icp_filters::filter_pipeline_from_yaml(
          c["observations_filter_2nd_pass"], this->getMinLoggingLevel());

        // Attach to the parameter source for dynamic parameters:
        mp2p_icp::AttachToParameterSource(state_.pc_filter2, state_.parameter_source);
      } else {
        MRPT_LOG_WARN(
          "No YAML entry 'observations_filter_2nd_pass', this "
          "filter stage will have no effect.");
      }

      if (c.has("observations_filter_final_pass")) {
        // Create, and copy my own verbosity level:
        state_.pc_filter3 = mp2p_icp_filters::filter_pipeline_from_yaml(
          c["observations_filter_final_pass"], this->getMinLoggingLevel());

        // Attach to the parameter source for dynamic parameters:
        mp2p_icp::AttachToParameterSource(state_.pc_filter3, state_.parameter_source);
      } else {
        MRPT_LOG_WARN(
          "No YAML entry 'observations_filter_final_pass', this "
          "filter stage will have no effect.");
      }

      // Local map generator:
      if (c.has("localmap_generator") && !c["localmap_generator"].isNullNode()) {
        // Create, and copy my own verbosity level:
        state_.local_map_generators = mp2p_icp_filters::generators_from_yaml(
          c["localmap_generator"], this->getMinLoggingLevel());
      } else {
        std::cout << "[warning] Using default mp2p_icp_filters::Generator "
                     "for the local map since no YAML 'localmap_generator' "
                     "entry was given\n";

        auto defaultGen = mp2p_icp_filters::Generator::Create();
        defaultGen->initialize({});
        state_.local_map_generators.push_back(defaultGen);
      }
      // Attach to the parameter source for dynamic parameters:
      mp2p_icp::AttachToParameterSource(state_.local_map_generators, state_.parameter_source);
    }

    // Parameterizable values in params_:
    params_.attachToParameterSource(state_.parameter_source);

  }  // end of state_mtx_ scope -- release before file I/O below

  // Preload maps (multisession SLAM or localization-only):
  if (
    !params_.local_map_updates.load_existing_local_map.empty() ||
    !params_.simplemap.load_existing_simple_map.empty()) {
    if (params_.local_map_updates.load_map_after_gui_init) {
      MRPT_LOG_INFO(
        "Map loading deferred until after GUI is initialized "
        "(load_map_after_gui_init=true).");
      pending_preload_map_ = true;
    } else {
      doPreloadLocalMap();
    }
  } else if (!params_.local_map_updates.enabled) {
    // Mapping disabled (localization-only) but no preexisting map was configured
    // to load: the local map will remain empty, which will make the first ICP
    // attempt run against an unprepared, empty map instead of building one online.
    MRPT_LOG_ERROR(
      "Mapping is disabled (mapping_enabled=false, localization-only mode) but no "
      "'local_map_updates.load_existing_local_map' (nor a simplemap) was configured to "
      "preload. The local map will remain empty: localization will very likely fail or "
      "crash on the first LiDAR scan. Either provide a map to load, or enable mapping.");
  }

  // Attach to the state estimation module, which since MOLA-LO v0.5.0,
  // must run as a separate MOLA module:
  {
    auto mods = findService<mola::NavStateFilter>();
    ASSERTMSG_(
      mods.size() == 1,
      "No state estimation MOLA module (mola::NavStateFilter) was found. Please, check your MOLA "
      "system .yaml file");
    state_.navstate_fuse = std::dynamic_pointer_cast<NavStateFilter>(mods[0]);
    ASSERT_(state_.navstate_fuse);
    MRPT_LOG_DEBUG("Attached to the state estimation module");
  }
  applyLoadedMapGeorefToStateEstimator();

#if defined(MOLA_HAS_SHARED_KEYFRAME_MAP_SINK)
  // Optional: a central-map backend (e.g. mola_mapper_3d) accepting
  // keyframe-insertion requests. Unlike navstate_fuse, this is NOT required:
  // most systems still write their own local .simplemap only.
  {
    auto sinks = findService<mola::SharedKeyframeMap>();
    if (!sinks.empty()) {
      state_.shared_keyframe_map_sink = std::dynamic_pointer_cast<SharedKeyframeMap>(sinks[0]);
      MRPT_LOG_DEBUG("Detected a SharedKeyframeMap sink: will push central-map keyframes to it.");
    }
  }
#endif

#if defined(MOLA_HAS_TRANSFORM_TREE_SOURCE)
  // Optional: a data source exposing a /tf tree, used only by the (opt-in)
  // tf-tree visualization. Absent in datasets without /tf, which is fine.
  {
    auto srcs = findService<mola::TransformTreeSource>();
    if (!srcs.empty()) {
      state_.transform_tree_source = std::dynamic_pointer_cast<TransformTreeSource>(srcs[0]);
      MRPT_LOG_DEBUG("Detected a TransformTreeSource: /tf tree visualization is available.");
    } else if (params_.visualization.show_tf_tree) {
      MRPT_LOG_WARN(
        "visualization.show_tf_tree is enabled, but no module in this system provides a "
        "mola::TransformTreeSource: nothing will be drawn.");
    }
  }
#endif

  // If using FromStateEstimator initialization, also subscribe to map updates
  // from the state estimator to receive geo-referencing information:
  if (params_.initial_localization.method == InitLocalization::FromStateEstimator) {
    // The state estimator may also be a MapSourceBase (for geo-ref publishing)
    if (auto mapSrc = std::dynamic_pointer_cast<MapSourceBase>(state_.navstate_fuse); mapSrc) {
      mapSrc->subscribeToMapUpdates(
        [this](const MapSourceBase::MapUpdate & mu) { onExternalMapUpdate(mu); });
      MRPT_LOG_DEBUG("Subscribed to state estimator's map updates for geo-referencing");
    }

    // Also subscribe to localization updates for pose convergence checking
    if (auto locSrc = std::dynamic_pointer_cast<LocalizationSourceBase>(state_.navstate_fuse);
        locSrc) {
      locSrc->subscribeToLocalizationUpdates(
        [this](const LocalizationSourceBase::LocalizationUpdate & lu) {
          onExternalLocalizationUpdate(lu);
        });
      MRPT_LOG_DEBUG("Subscribed to state estimator's localization updates");
    }
  }

  // end of initialization:
  {
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);

    state_.initialized = true;
    state_.active = params_.start_active;
  }

  // Make runtime params exposed:
  onExposeParameters();

  // Handle any optional persistent state/settings:
  onInitializePersistentState();

  MRPT_TRY_END
}

void LidarOdometry::doPreloadLocalMap()
{
  if (!params_.local_map_updates.load_existing_local_map.empty()) {
    MRPT_LOG_INFO_STREAM(
      "Loading map from file: '" << params_.local_map_updates.load_existing_local_map << "'...");

    auto lckState = mrpt::lockHelper(state_mtx_);
    // Guards the map contents against a concurrent visualization render,
    // which runs without state_mtx_ (see local_map_content_mtx_ docs):
    auto lckMapContents = mrpt::lockHelper(local_map_content_mtx_);

    const bool loadOk =
      state_.local_map->load_from_file(params_.local_map_updates.load_existing_local_map);
    ASSERTMSG_(
      loadOk, mrpt::format(
                "Error loading local map: '%s'",
                params_.local_map_updates.load_existing_local_map.c_str()));

    state_.loaded_map_georef_synced_from_navstate = false;
    state_.mark_local_map_as_updated(true);
    state_.mark_local_map_georef_as_updated();

    // Pre-warm ICP search structures for each layer that supports it.
    //
    // Without this, the very first call to mp2p_icp::ICP::align() for a freshly loaded
    // map does the prepare-global work (per-keyframe global-frame cloud materialization,
    // merged submap construction, KD-tree build) on the lidar worker thread. For a
    // non-trivial map this can stall the first scan for many seconds, during which the
    // bag-feed backs up and the GUI stops updating. Doing it here moves the cost to
    // mola-cli startup, alongside the cost of load_from_file the user already pays.
    //
    // The ref point used for selection is the configured initial pose (defaults to
    // origin). If the actual first ICP estimate ends up being far away, the search
    // submap will be transparently rebuilt then, but the per-keyframe heavy data is
    // already materialized and that rebuild is cheap.
    {
      const ProfilerEntry tle(profiler_, "initialize.prewarm_icp_search");
      const auto initial_pose =
        mrpt::poses::CPose3D(params_.initial_localization.fixed_initial_pose);
      std::size_t prepared_layers = 0;
      for (const auto & [layer_name, layer_map] : state_.local_map->layers) {
        const auto * prep = dynamic_cast<const mp2p_icp::IcpPrepareCapable *>(layer_map.get());
        if (prep == nullptr) {
          continue;
        }
        prep->icp_get_prepared_as_global(initial_pose);
        ++prepared_layers;
      }
      MRPT_LOG_INFO_STREAM(
        "Pre-warmed ICP search structures for "
        << prepared_layers << " of " << state_.local_map->layers.size() << " local-map layer(s).");
    }

    MRPT_LOG_INFO("Map loaded successfully.");
  }

  if (!params_.simplemap.load_existing_simple_map.empty()) {
    MRPT_LOG_INFO_STREAM(
      "Loading simple map from file: '" << params_.simplemap.load_existing_simple_map << "'...");

    auto lckSM = mrpt::lockHelper(state_simplemap_mtx_);

    const bool loadOk =
      state_.reconstructed_simplemap.loadFromFile(params_.simplemap.load_existing_simple_map);
    ASSERTMSG_(
      loadOk,
      mrpt::format(
        "Error loading simple map: '%s'", params_.simplemap.load_existing_simple_map.c_str()));

    MRPT_LOG_INFO("Simple map loaded successfully.");
  }

  state_.map_has_been_loaded = true;
  pending_preload_map_ = false;

  // Same guard as above, but here it also catches a configured map file that
  // turned out to load empty (e.g. an empty or corrupt map), which the checks
  // in initialize() above cannot see since they run before loading happens.
  if (!params_.local_map_updates.enabled && state_.local_map->empty()) {
    MRPT_LOG_ERROR(
      "Mapping is disabled (mapping_enabled=false, localization-only mode) but the "
      "local map is empty after the preload step. Localization will very likely fail "
      "or crash on the first LiDAR scan. Check that "
      "'local_map_updates.load_existing_local_map' points to a valid, non-empty map.");
  }

  // Covers the deferred-load path. For immediate preload this is initially a
  // no-op because NavStateFilter is attached just afterwards, where the helper
  // is called again.
  applyLoadedMapGeorefToStateEstimator();
}

void LidarOdometry::applyLoadedMapGeorefToStateEstimator()
{
#if defined(MOLA_KERNEL_NAVSTATE_FILTER_HAS_GEO_REFERENCE)
  std::shared_ptr<NavStateFilter> navstate;
  std::optional<mola::Georeferencing> georef;
  {
    auto lck = mrpt::lockHelper(state_mtx_);
    navstate = state_.navstate_fuse;
    if (state_.local_map && state_.local_map->georeferencing.has_value()) {
      const auto & mapGeoref = *state_.local_map->georeferencing;
      georef.emplace();
      georef->geo_coord = mapGeoref.geo_coord;
      georef->T_enu_to_map = mapGeoref.T_enu_to_map;
    }
  }
  if (navstate && georef.has_value()) {
    // A launch-configured fixed georeference (the run-directory sidecar) is
    // authoritative. It is written independently of the .mm shutdown save, so
    // it may be newer than metadata embedded in the metric map. Only fall back
    // to the embedded value when the estimator has no reference of its own.
    if (navstate->get_geo_reference().has_value()) {
      MRPT_LOG_INFO(
        "State estimator already has a geo-reference; keeping it instead of "
        "overwriting it with loaded-map metadata.");
    } else {
      navstate->set_geo_reference(*georef);
      MRPT_LOG_INFO("Applied loaded-map geo-reference to state estimator.");
    }
  }
#endif
}

}  // namespace mola
