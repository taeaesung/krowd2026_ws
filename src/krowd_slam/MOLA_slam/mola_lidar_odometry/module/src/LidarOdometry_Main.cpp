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
 * Implements the module lifecycle and cross-cutting utilities of
 * mola::LidarOdometry that don't belong to a more specific file: ctor/dtor,
 * spinOnce() (called at a fixed rate by the MOLA executor, independent of
 * sensor data arrival), shutdown/save-to-disk logic, the thread-safe
 * accessors used by other MOLA modules/GUIs to query the latest estimated
 * state, and small persistent-config/version-check helpers. The actual
 * per-scan processing lives in LidarOdometry_ProcessScan.cpp and is invoked
 * from the sensor callbacks in LidarOdometry_SensorCallbacks.cpp.
 */

// This module:
#include <mola_lidar_odometry/LidarOdometry.h>

// MP2P_ICP:
#include <mp2p_icp/Solver_GaussNewton.h>
#include <mp2p_icp_filters/sm2mm.h>
#include <mrpt/config/CConfigFile.h>
#include <mrpt/config/CConfigFileMemory.h>
#include <mrpt/containers/yaml.h>
#include <mrpt/core/initializer.h>
#include <mrpt/io/lazy_load_path.h>
#include <mrpt/maps/CColouredPointsMap.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/obs/CRawlog.h>
#include <mrpt/opengl/CAssimpModel.h>
#include <mrpt/opengl/CGridPlaneXY.h>
#include <mrpt/opengl/COpenGLScene.h>
#include <mrpt/opengl/CText.h>
#include <mrpt/opengl/stock_objects.h>
#include <mrpt/poses/CPose3DPDFGaussian.h>
#include <mrpt/random.h>
#include <mrpt/system/datetime.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/string_utils.h>
#include <mrpt/topography/conversions.h>

// STD:
#include <chrono>
#include <mutex>
#include <thread>

// Portable config path
#include "libcfgpath/cfgpath.h"

// Portable online version check:
#if defined(MOLA_LO_HAS_ONLINE_VERSION_CHECK)
#include <arpa/inet.h>  // for sockaddr_in
#include <netdb.h>      // for gethostbyname(), hostent
#include <unistd.h>     // for close()

#include <cstring>  // for memset
#include <iostream>
#include <sstream>
#include <string>
#endif

namespace mola
{

// arguments: class_name, parent_class, class namespace
IMPLEMENTS_MRPT_OBJECT(LidarOdometry, FrontEndBase, mola)

LidarOdometry::LidarOdometry() = default;

// Belt-and-braces: normally onQuit() (below) has already run shutdownCleanup()
// by the time the destructor executes; shutdown_cleanup_done_ makes the call
// here a no-op in that (expected) case, but still cleans up correctly for
// callers that destroy the object without going through MolaLauncherApp.
LidarOdometry::~LidarOdometry() { shutdownCleanup(); }

void LidarOdometry::onQuit() { shutdownCleanup(); }

// Drains the worker thread pools and saves outputs to disk. Called from
// onQuit() (while all other MOLA modules are still alive) and again from the
// destructor (idempotent, no-op the second time) for callers that do not go
// through MolaLauncherApp.
//
// onQuit() is invoked by MolaLauncherApp::executor_thread() for every module,
// before any module is destroyed (see ExecutableBase::onQuit() docs). This
// matters because worker_lidar_/worker_others_/worker_viz_/
// worker_viz_local_map_ may have an in-flight task (e.g. onLidar(), or the
// local-map render) that calls back into other modules, e.g.
// BridgeROS2's TF broadcaster via VizInterface/LocalizationSourceBase. If
// that task is still running when another module's destructor runs (which
// previously only happened from ~LidarOdometry(), after other modules may
// have already been destroyed), it can use-after-free that module's
// resources.
void LidarOdometry::shutdownCleanup()
{
  using namespace std::chrono_literals;

  if (shutdown_cleanup_done_.exchange(true)) {
    return;
  }

  try  // must never throw
  {
    {
      auto lck = mrpt::lockHelper(is_busy_mtx_);
      destructor_called_ = true;
    }

    // clear() drops queued work and joins the currently running callback. Do
    // this immediately: worker_others_ is FIFO and can otherwise contain
    // thousands of 100 Hz IMU callbacks, making Ctrl-C wait minutes before the
    // save path is ever reached. destructor_called_ above prevents any new
    // LiDAR callback from starting while the pools are being drained.
    worker_lidar_.clear();
    worker_others_.clear();
    worker_viz_.clear();
    worker_viz_local_map_.clear();
    {
      auto lck = mrpt::lockHelper(is_busy_mtx_);
      state_.worker_tasks_lidar = 0;
      state_.worker_tasks_others = 0;
    }

    if (params_.simplemap.generate) {
      saveReconstructedMapToFile();
    }

    if (params_.estimated_trajectory.save_to_file) {
      saveEstimatedTrajectoryToFile();
    }
    if (!params_.local_map_updates.save_final_local_map.empty()) {
      saveLocalMapToFile();
    }

    // This must come after save map calls above:
    while (worker_disk_io_.pendingTasks() > 0) {
      MRPT_LOG_THROTTLE_WARN(
        2.0, "shutdownCleanup(): waiting for pending tasks on the lazy-load write thread...");
      std::this_thread::sleep_for(100ms);
    }
    worker_disk_io_.clear();

    if (!params_.local_map_updates.save_final_global_map.empty()) {
      saveGlobalMapFromSimpleMapToFile();
    }

  } catch (const std::exception & e) {
    std::cerr << "[LidarOdometry::shutdownCleanup] Exception: " << e.what();
  }
}

// Called by the MOLA executor at a fixed rate (independent of sensor data
// arrival). Used here for bookkeeping that must happen even when no LiDAR
// scan is being processed: draining enqueue_request()'d work, keeping the
// GUI alive/updated even while idle or inactive, deferred map loading, and
// periodic diagnostics/local-map republishing.
void LidarOdometry::spinOnce()
{
  MRPT_TRY_START

  const ProfilerEntry tle(profiler_, "spinOnce");

  processPendingUserRequests();

  // Force a refresh of the GUI?
  // Executed here since
  // otherwise the GUI would never show up if inactive, or if the LIDAR
  // observations are misconfigured and are not been fed in.
  // Note the explicit "!gui_.gui_created" check: in localization mode
  // (start_active=true, start_mapping_enabled=false) a prior map is loaded, so
  // the local map is not empty and the module is active. Without this check
  // neither of the other two conditions would hold and the GUI (3D scene and
  // UI controls, both built lazily on the first updateVisualization() call)
  // would never show up until the first LiDAR scan arrives.
  bool guiCreated = false;
  {
    auto lckGuiMtx = mrpt::lockHelper(state_gui_mtx_);
    guiCreated = gui_.gui_created;
  }
  // Evaluated before taking state_mtx_, since it uses state_flags_mtx_:
  const bool isNowActive = isActive();

  if (visualizer_) {
    // updateVisualization() reads state_, so it must be called with state_mtx_
    // held:
    std::unique_lock<std::mutex> lckState(state_mtx_);

    if (
      ((state_.local_map && state_.local_map->empty()) || !isNowActive || !guiCreated) &&
      mrpt::Clock::nowDouble() - gui_.timestampLastUpdateUI > 1.0) {
      updateVisualization({}, {}, lckState);
    }
  }

  // If map loading was deferred (load_map_after_gui_init=true), run it now,
  // after the GUI has had a chance to render at least once:
  if (pending_preload_map_ && guiCreated) {
    doPreloadLocalMap();
  }

  // If SLAM/Localization is disabled, refresh the current map
  // here if needed, since it won't be published until observations arrive.
  {
    auto lckState = mrpt::lockHelper(state_mtx_);

    const auto mapStamp =
      state_.last_obs_timestamp ? *state_.last_obs_timestamp : mrpt::Clock::now();

    doPublishUpdatedLocalMap(mapStamp);
  }

  // Publish optional regular diagnostics:
  if (module_is_time_to_publish_diagnostics()) {
    onPublishDiagnostics();
  }

  MRPT_TRY_END
}

void LidarOdometry::reset()
{
  ASSERTMSG_(!lastInitConfig_.empty(), "initialize() must be called first.");

  // Reset the state under the lock, then re-initialize without holding it:
  // initialize() (via initialize_frontend()) takes state_mtx_ itself, and
  // state_mtx_ is a plain (non-recursive) mutex.
  {
    auto lck = mrpt::lockHelper(state_mtx_);
    // Replacing state_ wholesale also destroys the parameter source, the
    // gravity estimator and the rest of the IMU-fed state, which the IMU worker
    // thread reaches without state_mtx_:
    auto lckImu = mrpt::lockHelper(imu_state_mtx_);
    state_ = MethodState();
  }
  initialize(lastInitConfig_);
}

// True if any worker task is currently running or queued (used by
// shutdownCleanup() to wait for in-flight work before tearing down state).
bool LidarOdometry::isBusy() const
{
  bool b = false;
  is_busy_mtx_.lock();
  b = (state_.worker_tasks_lidar != 0) || (state_.worker_tasks_others != 0);
  is_busy_mtx_.unlock();
  return b || worker_lidar_.pendingTasks() != 0 || worker_others_.pendingTasks() != 0;
}

bool LidarOdometry::isActive() const
{
  auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
  return state_.active;
}

// Toggles processing of incoming sensor data on/off (see Parameters::start_active).
void LidarOdometry::setActive(const bool active)
{
  auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
  state_.active = active;
}

// Returns a copy (not a reference) of the trajectory under its own mutex, so
// callers get a consistent snapshot without blocking the worker thread for
// longer than the copy itself takes.
mrpt::poses::CPose3DInterpolator LidarOdometry::estimatedTrajectory() const
{
  auto lck = mrpt::lockHelper(state_trajectory_mtx_);
  return state_.estimated_trajectory;
}

mrpt::maps::CSimpleMap LidarOdometry::reconstructedMap() const
{
  auto lck = mrpt::lockHelper(state_simplemap_mtx_);
  return state_.reconstructed_simplemap;
}

// Two-stage locking: first check the (cheap) initialized/fatal_error flags
// under state_flags_mtx_ and bail out early, only then take the heavier
// state_mtx_ to read the pose/twist. Avoids holding state_mtx_ (which the
// processing loop needs) any longer than necessary.
std::optional<std::tuple<mrpt::poses::CPose3DPDFGaussian, mrpt::math::TTwist3D>>
LidarOdometry::lastEstimatedState() const
{
  {
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
    if (!state_.initialized || state_.fatal_error) {
      return {};
    }
  }

  auto lck = mrpt::lockHelper(state_mtx_);

  if (!state_.last_motion_model_output) {
    return {};
  }
  const auto & tw = state_.last_motion_model_output->twist;
  const auto & pose = state_.last_lidar_pose;
  return {{pose, tw}};
}

// Same two-stage locking pattern as lastEstimatedState(), for the ICP
// goodness score alone.
double LidarOdometry::lastIcpQuality() const
{
  {
    auto lckStateFlags = mrpt::lockHelper(state_flags_mtx_);
    if (!state_.initialized || state_.fatal_error) {
      return 0;
    }
  }

  auto lck = mrpt::lockHelper(state_mtx_);

  return state_.last_icp_quality;
}

// Dumps the full estimated trajectory (one pose per processed LiDAR scan) to
// disk in TUM format. Called from shutdownCleanup() if
// Parameters::estimated_trajectory.save_to_file is set.
void LidarOdometry::saveEstimatedTrajectoryToFile() const
{
  if (params_.estimated_trajectory.output_file.empty()) {
    return;
  }

  auto lck = mrpt::lockHelper(state_trajectory_mtx_);

  const auto fil = params_.estimated_trajectory.output_file;

  MRPT_LOG_INFO_STREAM(
    "Saving estimated trajectory with " << state_.estimated_trajectory.size()
                                        << " keyframes to file '" << fil << "' in TUM format...");

  state_.estimated_trajectory.saveToTextFile_TUM(fil);

  MRPT_LOG_INFO("Final trajectory saved.");
}

// Dumps the accumulated CSimpleMap (keyframes + observations) to a
// ".simplemap" file. Called from shutdownCleanup() if
// Parameters::simplemap.save_final_map_to_file is set.
void LidarOdometry::saveReconstructedMapToFile() const
{
  if (params_.simplemap.save_final_map_to_file.empty()) {
    return;
  }

  // make sure the unload queue is empty first,
  // so if we have this feature enabled, all SF entries have been
  // "externalized" to make reading them much faster and less RAM
  // intensive:
  unloadPastSimplemapObservations(0 /* unload until queue is empty */);

  auto lck = mrpt::lockHelper(state_simplemap_mtx_);

  const auto fil = params_.simplemap.save_final_map_to_file;

  MRPT_LOG_INFO_STREAM(
    "Saving final simplemap with " << state_.reconstructed_simplemap.size()
                                   << " keyframes to file '" << fil << "'...");
  std::cout.flush();

  state_.reconstructed_simplemap.saveToFile(fil);

  MRPT_LOG_INFO("Final simplemap saved.");
}

// Dumps the current local metric map (the map used for ICP registration,
// distinct from the simplemap) to an ".mm" file. Called from
// shutdownCleanup() if Parameters::local_map_updates.save_final_local_map is set.
void LidarOdometry::saveLocalMapToFile() const
{
  if (params_.local_map_updates.save_final_local_map.empty()) {
    return;
  }

  auto lck = mrpt::lockHelper(state_mtx_);
  auto lckMapContents = mrpt::lockHelper(local_map_content_mtx_);

  const auto fil = params_.local_map_updates.save_final_local_map;
  const auto georefFile =
    params_.local_map_updates.save_final_local_map_georeference;

  if (!georefFile.empty()) {
    bool installedRequestedGeoref = false;
    if (!mrpt::system::fileExists(georefFile)) {
      MRPT_LOG_WARN_STREAM(
        "Metric-map georeference sidecar does not exist; saving without "
        "embedded global alignment: '" << georefFile << "'");
    } else {
      try {
        const auto yamlData = mrpt::containers::yaml::FromFile(georefFile);
        const auto georef = mp2p_icp::FromYAML(yamlData);
        if (georef.has_value()) {
          state_.local_map->georeferencing = georef;
          installedRequestedGeoref = true;
          MRPT_LOG_INFO_STREAM(
            "Embedded metric-map georeference from '" << georefFile << "'.");
        } else {
          MRPT_LOG_WARN_STREAM(
            "Georeference sidecar is explicitly undefined: '" << georefFile << "'.");
        }
      } catch (const std::exception & e) {
        MRPT_LOG_ERROR_STREAM(
          "Could not parse metric-map georeference sidecar '" << georefFile
                                                               << "': " << e.what());
      }
    }
    if (!installedRequestedGeoref) {
      // Never serialize a pre-existing georeference under saved-odom point
      // coordinates: it may describe the smoother's map frame and omit the
      // map<-odom part. An absent reference is detectable by localization;
      // a plausible but frame-inconsistent one is much more dangerous.
      state_.local_map->georeferencing.reset();
    }
  }

  MRPT_LOG_INFO_STREAM("Saving final metric map to file '" << fil << "'...");
  std::cout.flush();

  const bool saved_ok = state_.local_map->save_to_file(fil);
  if (!saved_ok) {
    MRPT_LOG_ERROR_STREAM("Error saving map to: " << fil);
  } else {
    MRPT_LOG_INFO("Final local metric map saved.");
  }
}

// Rebuilds an unbounded, full-route metric map from accepted simplemap
// keyframes. This intentionally happens after the rolling local map has been
// saved and all lazy-load writes have completed, so live distance eviction
// cannot truncate the final artifact.
void LidarOdometry::saveGlobalMapFromSimpleMapToFile() const
{
  const auto & outputFile = params_.local_map_updates.save_final_global_map;
  const auto & pipelineFile = params_.local_map_updates.save_final_global_map_pipeline;
  if (outputFile.empty()) {
    return;
  }
  if (pipelineFile.empty()) {
    MRPT_LOG_ERROR(
      "Cannot rebuild global metric map: save_final_global_map_pipeline is empty.");
    return;
  }
  if (!mrpt::system::fileExists(pipelineFile)) {
    MRPT_LOG_ERROR_STREAM(
      "Cannot rebuild global metric map: pipeline does not exist: '" << pipelineFile << "'.");
    return;
  }

  auto lckSimpleMap = mrpt::lockHelper(state_simplemap_mtx_);
  if (state_.reconstructed_simplemap.empty()) {
    MRPT_LOG_ERROR(
      "Cannot rebuild global metric map: the reconstructed simplemap is empty.");
    return;
  }

  MRPT_LOG_INFO_STREAM(
    "Rebuilding full global metric map from " << state_.reconstructed_simplemap.size()
                                                << " accepted simplemap keyframes using '"
                                                << pipelineFile << "'...");
  std::cout.flush();

  const auto pipeline = mrpt::containers::yaml::FromFile(pipelineFile);
  mp2p_icp::metric_map_t globalMap;
  mp2p_icp_filters::sm2mm_options_t options;
  options.verbosity = mrpt::system::LVL_INFO;
  options.showProgressBar = false;
  options.throw_on_missing_external_files = true;
  mp2p_icp_filters::simplemap_to_metricmap(
    state_.reconstructed_simplemap, globalMap, pipeline, options);

  // Keep point coordinates in the small odometry frame. Attach metadata only;
  // passing georeferencing through sm2mm options would transform every point
  // into large ENU coordinates and reintroduce RViz float precision jitter.
  const auto & georefFile =
    params_.local_map_updates.save_final_global_map_georeference;
  if (!georefFile.empty()) {
    if (!mrpt::system::fileExists(georefFile)) {
      MRPT_LOG_WARN_STREAM(
        "Global-map georeference sidecar does not exist; saving the point map "
        "without embedded global alignment: '" << georefFile << "'.");
      globalMap.georeferencing.reset();
    } else {
      const auto georef = mp2p_icp::FromYAML(mrpt::containers::yaml::FromFile(georefFile));
      if (georef.has_value()) {
        globalMap.georeferencing = georef;
        MRPT_LOG_INFO_STREAM(
          "Attached global-map georeference metadata from '" << georefFile << "'.");
      } else {
        MRPT_LOG_WARN_STREAM(
          "Global-map georeference sidecar is undefined: '" << georefFile << "'.");
        globalMap.georeferencing.reset();
      }
    }
  }

  MRPT_LOG_INFO_STREAM(
    "Saving rebuilt global metric map (" << globalMap.contents_summary() << ") to '"
                                          << outputFile << "'...");
  if (!globalMap.save_to_file(outputFile)) {
    MRPT_LOG_ERROR_STREAM("Error saving rebuilt global metric map to: " << outputFile);
    return;
  }
  MRPT_LOG_INFO("Full global metric map saved.");
}

// Trims state_.past_simplemaps_observations down to maxSizeUnloadQueue
// entries, oldest first, freeing their RAM via unload() (see
// handleUnloadSinglePastObservation()). Called periodically while building a
// simplemap with lazy-load enabled to bound peak memory, and with
// maxSizeUnloadQueue=0 before final save to flush everything.
void LidarOdometry::unloadPastSimplemapObservations(const size_t maxSizeUnloadQueue) const
{
  auto lck = mrpt::lockHelper(state_simplemap_mtx_);

  auto & pso = state_.past_simplemaps_observations;

  while (pso.size() > maxSizeUnloadQueue) {
    for (auto & o : *pso.begin()->second) {
      handleUnloadSinglePastObservation(o);
    }

    pso.erase(pso.begin());
  }
}

// Frees an observation's in-RAM payload, externalizing point clouds to disk
// first if lazy-load simplemap generation is enabled (so a later reload of
// the .simplemap is fast and light on RAM). No-op paths below bail out early
// for observations that are already external, or when the feature isn't
// requested; the actual disk write is offloaded to worker_disk_io_ since
// serialization can be slow and must not stall the caller.
void LidarOdometry::handleUnloadSinglePastObservation(mrpt::obs::CObservation::Ptr & o) const
{
  using mrpt::obs::CObservationPointCloud;

  // Generic method first: it will work with datasets providing input
  // observations *already* in lazy-load format:
  o->unload();

  // special case: point cloud
  auto oPts = std::dynamic_pointer_cast<CObservationPointCloud>(o);
  if (!oPts) {
    return;
  }

  if (oPts->isExternallyStored()) {
    return;  // already external, do nothing.
  }

  if (params_.simplemap.save_final_map_to_file.empty()) {
    return;  // no generation of simplemap requested by the user
  }

  if (!params_.simplemap.generate_lazy_load_scan_files) {
    return;  // feature is disabled
  }

  ASSERT_(oPts->pointcloud);

  const std::string filename = mrpt::format(
    "%s_%.09f.bin", mrpt::system::fileNameStripInvalidChars(oPts->sensorLabel).c_str(),
    mrpt::Clock::toDouble(oPts->timestamp));

  // Create the default "/Images" directory.
  const auto & smFile = params_.simplemap.save_final_map_to_file;

  const std::string out_basedir = mrpt::system::pathJoin(
    {mrpt::system::extractFileDirectory(smFile),
     mrpt::system::extractFileName(smFile) + std::string("_Images")});

  if (!mrpt::system::directoryExists(out_basedir)) {
    const bool dirCreatedOk = mrpt::system::createDirectory(out_basedir);
    ASSERTMSG_(
      dirCreatedOk,
      mrpt::format(
        "Error creating lazy-load directory for output simplemap: '%s'", out_basedir.c_str()));

    MRPT_LOG_INFO_STREAM("Creating lazy-load directory for output .simplemap: " << out_basedir);
  }

  // Establish as reference external path base:
  mrpt::io::setLazyLoadPathBase(out_basedir);

  oPts->setAsExternalStorage(
    filename, CObservationPointCloud::ExternalStorageFormat::MRPT_Serialization);

  // Since unload() does the actual saving to disk and it might take some time, let's run it in its own thread.
  const auto fut = this->worker_disk_io_.enqueue([oPts]() {
    try {
      oPts->unload();
    } catch (const std::exception & e) {
      std::cerr << "[LidarOdometry] handleUnloadSinglePastObservation(): Error saving "
                   "observation to disk: "
                << e.what() << "\n";
    }
  });
  (void)fut;
}

void LidarOdometry::enqueue_request(const std::function<void()> & userRequest)
{
  auto lck = mrpt::lockHelper(requests_mtx_);
  requests_.push_back(userRequest);
}

void LidarOdometry::processPendingUserRequests()
{
  // Swap-and-release pattern:
  //  1) Atomically take ownership of the pending list by swapping it out
  //     under requests_mtx_.
  //  2) Release requests_mtx_ before executing any user lambda.
  // This is required because a request body may legitimately call
  // enqueue_request() (which would self-deadlock on a non-recursive mutex),
  // or take state_mtx_ (a previous version of this function held it here,
  // stalling ROS service threads that called e.g. relocalize_near_pose_pdf).
  std::vector<std::function<void()>> pending;
  {
    auto lck = mrpt::lockHelper(requests_mtx_);
    pending.swap(requests_);
  }

  for (const auto & r : pending) {
    try {
      r();
    } catch (const std::exception & e) {
      MRPT_LOG_ERROR_STREAM("Error processing asynchronous enqueue_request(): " << e.what());
    }
  }
}

#if defined(MOLA_LO_HAS_ONLINE_VERSION_CHECK)
// This whole block (compiled only when MOLA_LO_HAS_ONLINE_VERSION_CHECK is
// defined) implements a minimal, dependency-free HTTP/1.0 GET using raw
// BSD sockets, purely to fetch a one-line "latest version" text file and
// notify the user (via a log message only, never blocking) if a newer
// release exists. It deliberately avoids pulling in a full HTTP/TLS client
// library for this optional, best-effort feature.
namespace
{
// Opens a raw TCP socket to host:80 and issues a bare HTTP/1.0 GET for
// `path`, returning the full raw response (headers + body) as text.
// Throws std::runtime_error on any resolution/socket/connect failure.
std::string http_get(const std::string & host, const std::string & path = "/")
{
  const int port = 80;

  // Resolve hostname
  hostent * server = gethostbyname(host.c_str());  // NOLINT
  if (server == nullptr) {
    throw std::runtime_error("Error: no such host");
  }

  // Create socket
  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    throw std::runtime_error("Error: opening socket");
  }

  // Build server address struct
  sockaddr_in serv_addr{};
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  std::memcpy(&serv_addr.sin_addr.s_addr, server->h_addr, server->h_length);

  // Connect
  if (connect(sock, reinterpret_cast<sockaddr *>(&serv_addr), sizeof(serv_addr)) < 0) {  // NOLINT
    close(sock);
    throw std::runtime_error("Error: connecting");
  }

  // Send HTTP request
  std::ostringstream request;
  request << "GET " << path << " HTTP/1.0\r\n"
          << "Host: " << host << "\r\n"
          << "Connection: close\r\n\r\n";

  std::string req_str = request.str();
  send(sock, req_str.c_str(), req_str.size(), 0);

  // Read response
  std::string response;
  std::array<char, 1024> buffer = {};
  ssize_t bytes = 0;
  while ((bytes = recv(sock, buffer.data(), buffer.size() - 1, 0)) > 0) {
    buffer[bytes] = '\0';
    response += std::string(buffer.data());
  }

  close(sock);
  return response;
}
// Extracts the body out of a raw HTTP response, checking for a "200" status
// line and trimming surrounding whitespace. Returns false if the response
// doesn't look like a 200 OK.
bool parse_http_response(const std::string & response, std::string & body_out)
{
  std::istringstream ss(response);
  std::string line;

  // Parse status line
  if (!std::getline(ss, line)) {
    return false;
  }

  if (line.find("200") == std::string::npos) {
    return false;  // not 200 OK
  }

  // Skip headers
  while (std::getline(ss, line)) {
    if (line == "\r" || line.empty()) {
      break;
    }
  }

  // Read remaining lines as body
  std::ostringstream body;
  while (std::getline(ss, line)) {
    body << line;
    if (!ss.eof()) {
      body << '\n';
    }
  }

  // Trim whitespace
  std::string raw = body.str();

  // Remove leading whitespace
  size_t start = raw.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    body_out.clear();
    return true;
  }

  // Remove trailing whitespace
  size_t end = raw.find_last_not_of(" \t\r\n");

  body_out = raw.substr(start, end - start + 1);
  return true;
}

// Compares two dot-separated version strings (e.g. "3.10.1") numerically
// component-by-component (not lexicographically, so "3.10" > "3.9").
// Missing/non-numeric components are treated as 0.
bool version_greater(const std::string & a, const std::string & b)  // NOLINT
{
  std::istringstream sa(a);
  std::istringstream sb(b);
  int va = 0;
  int vb = 0;
  char dot = 0;

  while (true) {
    if (!(sa >> va)) {
      va = 0;  // default 0 if missing
    }
    if (!(sb >> vb)) {
      vb = 0;
    }

    if (va > vb) {
      return true;
    }
    if (va < vb) {
      return false;
    }

    // consume '.' if present
    if (!(sa >> dot) && !(sb >> dot)) {
      break;  // both ended
    }
  }
  return false;  // equal or smaller
}

// Rate-limited (once per 24h, tracked in the user config file) check for a
// newer release, run fully asynchronously so it can never delay startup or
// block on a slow/unreachable network.
void check_new_version(mrpt::config::CConfigFile & cfg, mrpt::system::COutputLogger * logger)
{
  // Do we need to check today?
  const auto lastTim = cfg.read_double("config", "last_version_check", 0);
  const auto now = mrpt::Clock::nowDouble();
  if (now < lastTim + 24 * 60 * 60) {
    return;
  }
  // Check in a detached thread: fire-and-forget, we don't join it, so
  // startup/shutdown never wait on network I/O.
  std::thread([logger]() {
    try {
      auto response = http_get("docs.mola-slam.org", "/latest/mola_lidar_odometry.version");
      std::string body;
      if (parse_http_response(response, body)) {
        if (version_greater(MOLA_LO_VERSION, body)) {
          logger->logFmt(
            mrpt::system::LVL_DEBUG,
            "[mola-lidar-odometry module] There is a newer version of mola_lidar_odometry "
            "available online: '%s' vs current '%s'",
            body.c_str(), MOLA_LO_VERSION);
        }

      } else {
        std::cout << "[FAIL] Invalid response\n";
      }
    } catch (const std::exception & e) {
      logger->logFmt(mrpt::system::LVL_DEBUG, "[check_new_version] Error: %s", e.what());
    }
  }).detach();

  cfg.write("config", "last_version_check", mrpt::format("%f", now));
}
}  // namespace
#endif

// Reads/writes the small per-user, cross-run persistent config file (via the
// vendored libcfgpath helper for a portable "user config dir" path) used to
// track the last-run timestamp and throttle the online version check.
void LidarOdometry::onInitializePersistentState()
{
  try {
    // Get user app config file
    char appCfgFile[2048];
    ::get_user_config_file(appCfgFile, sizeof(appCfgFile), "mola-lidar-odometry");
    mrpt::config::CConfigFile appCfg(appCfgFile);

    // Handle persistent values:
    appCfg.write("config", "last_run", mrpt::format("%f", mrpt::Clock::nowDouble()));

    // Handle printing a notice of newer versions available:
#if defined(MOLA_LO_HAS_ONLINE_VERSION_CHECK)
    check_new_version(appCfg, this);
#endif

  } catch (const std::exception & e) {
    MRPT_LOG_WARN_STREAM("Error initializing persistent config file:\n" << e.what());
  }
}

// Appends one CSV row per processed scan with every "dynamic variable"
// exposed via state_.parameter_source (ICP pipeline tunables, adaptive
// threshold, etc.) plus timing info. Lazily opens/writes a CSV header on the
// first call. Purely a debugging aid (Parameters::debug_traces).
void LidarOdometry::doWriteDebugTracesFile(const mrpt::Clock::time_point & scan_ref_time)
{
  if (!params_.debug_traces.save_to_file) {
    return;  // disabled
  }

  if (debug_traces_of_ && !debug_traces_of_->is_open()) {
    return;  // apparently, an error creating the file
  }

  bool firstLine = false;
  if (!debug_traces_of_) {
    debug_traces_of_.emplace();
    debug_traces_of_->open(params_.debug_traces.output_file);
    if (debug_traces_of_->is_open()) {
      MRPT_LOG_INFO_STREAM("Writing debug traces to: " << params_.debug_traces.output_file);
      firstLine = true;
    } else {
      MRPT_LOG_ERROR_STREAM(
        "Could not create debug traces file: " << params_.debug_traces.output_file);
      return;
    }
  }

  auto & of = debug_traces_of_.value();

  auto vars = [this]() {
    auto lckImu = mrpt::lockHelper(imu_state_mtx_);
    return state_.parameter_source.getVariableValues();
  }();
  vars["timestamp"] = mrpt::Clock::toDouble(scan_ref_time);
  vars["time_onLidar"] = profiler_.getLastTime("onLidar");

  if (firstLine) {
    for (const auto & [name, value] : vars) {
      of << "\"" << name << "\",";
    }
    of << "\n";
  }
  for (const auto & [name, value] : vars) {
    of << mrpt::format("%f,", value);
  }
  of << "\n";
}

// Records one outcome (kept/dropped) into the fixed-size circular window
// used to compute the recent frame-drop ratio (see getDropStats()).
void LidarOdometry::addDropStats(bool frame_is_dropped)
{
  auto lck = mrpt::lockHelper(drop_stats_mtx_);
  state_.drop_frames_stats_good[state_.drop_frames_stats_next_index] = !frame_is_dropped;
  state_.drop_frames_stats_dropped[state_.drop_frames_stats_next_index] = frame_is_dropped;
  if (++state_.drop_frames_stats_next_index >= MethodState::DROP_STATS_WINDOW_LENGTH) {
    state_.drop_frames_stats_next_index = 0;
  }
}

// Fraction of frames marked as dropped within the last DROP_STATS_WINDOW_LENGTH
// recorded outcomes; used for the "dropped_ratio_warn/error" diagnostics thresholds.
double LidarOdometry::getDropStats() const
{
  auto lck = mrpt::lockHelper(drop_stats_mtx_);
  const auto good =
    std::count(state_.drop_frames_stats_good.begin(), state_.drop_frames_stats_good.end(), true);
  const auto bad = std::count(
    state_.drop_frames_stats_dropped.begin(), state_.drop_frames_stats_dropped.end(), true);
  const auto total = static_cast<double>(good + bad);
  return total != 0 ? static_cast<double>(bad) / total : .0;
}

}  // namespace mola
