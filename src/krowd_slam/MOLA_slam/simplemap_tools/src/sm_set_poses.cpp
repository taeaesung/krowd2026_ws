// sm-set-poses: replace every keyframe pose of a .simplemap with the poses of
// a TUM trajectory file (t x y z qx qy qz qw), matched by timestamp.
//
// Companion to mola_slam_mapping's map_retraject: that tool optimizes the
// keyframe trajectory (Python/scipy); this one only writes the result back,
// because MRPT's Python bindings are not installable in this environment.
// Keyframe timestamps are taken exactly as `sm-cli export-keyframes` takes
// them (first observation carrying a valid stamp), so the TUM file that tool
// exports and the one this tool reads describe the same keyframes.
//
// Strict on purpose: a keyframe without a timestamp, or without a TUM entry
// within --max-time-diff, aborts. A partially re-trajectoried map would be
// geometrically inconsistent and there is no sensible default for it.

#include <mrpt/core/exceptions.h>
#include <mrpt/maps/CSimpleMap.h>
#include <mrpt/maps/registerAllClasses.h>
#include <mrpt/math/CQuaternion.h>
#include <mrpt/obs/CSensoryFrame.h>
#include <mrpt/obs/registerAllClasses.h>
#include <mrpt/poses/CPose3D.h>
#include <mrpt/system/datetime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>

namespace
{

std::map<double, mrpt::poses::CPose3D> loadTum(const std::string& path)
{
    std::ifstream file(path);
    if (!file) THROW_EXCEPTION_FMT("cannot open TUM file '%s'", path.c_str());

    std::map<double, mrpt::poses::CPose3D> poses;
    std::string                             line;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        double             t, x, y, z, qx, qy, qz, qw;
        if (!(fields >> t >> x >> y >> z >> qx >> qy >> qz >> qw))
            THROW_EXCEPTION_FMT("malformed TUM line: '%s'", line.c_str());
        mrpt::math::CQuaternionDouble q(qw, qx, qy, qz);
        q.normalize();
        poses.emplace(t, mrpt::poses::CPose3D(q, x, y, z));
    }
    if (poses.empty()) THROW_EXCEPTION_FMT("no poses in TUM file '%s'", path.c_str());
    return poses;
}

std::optional<double> keyframeTimestamp(const mrpt::obs::CSensoryFrame& sf)
{
    for (const auto& obs : sf)
    {
        if (!obs) continue;
        const auto t = obs->getTimeStamp();
        if (t == mrpt::system::InvalidTimeStamp()) continue;
        return mrpt::Clock::toDouble(t);
    }
    return std::nullopt;
}

// Nearest TUM entry to `stamp`, with its time offset.
std::pair<const mrpt::poses::CPose3D*, double> nearestPose(
    const std::map<double, mrpt::poses::CPose3D>& poses, double stamp)
{
    auto after = poses.lower_bound(stamp);
    if (after == poses.end()) --after;
    auto best = after;
    if (after != poses.begin())
    {
        auto before = std::prev(after);
        if (std::abs(before->first - stamp) < std::abs(after->first - stamp)) best = before;
    }
    return {&best->second, best->first - stamp};
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr << "usage: sm-set-poses <input.simplemap> <poses.tum> <output.simplemap> "
                     "[max_time_diff_sec=0.005]\n";
        return 2;
    }
    try
    {
        // Deserializing the keyframes' observations (CObservationPointCloud,
        // ...) needs their classes registered; the linker keeps no reference
        // to them otherwise. Same as sm-cli-main.cpp.
        mrpt::maps::registerAllClasses_mrpt_maps();
        mrpt::obs::registerAllClasses_mrpt_obs();

        const std::string inputPath  = argv[1];
        const std::string tumPath    = argv[2];
        const std::string outputPath = argv[3];
        const double      maxTimeDiff = argc == 5 ? std::atof(argv[4]) : 0.005;

        const auto poses = loadTum(tumPath);

        mrpt::maps::CSimpleMap sm;
        if (!sm.loadFromFile(inputPath))
            THROW_EXCEPTION_FMT("cannot load simplemap '%s'", inputPath.c_str());

        size_t index = 0;
        double worstOffset = 0.0;
        for (auto& [pose, sf, twist] : sm)
        {
            ASSERT_(pose);
            ASSERT_(sf);
            const auto stamp = keyframeTimestamp(*sf);
            if (!stamp) THROW_EXCEPTION_FMT("keyframe #%zu has no timestamped observation", index);

            const auto [target, offset] = nearestPose(poses, *stamp);
            if (std::abs(offset) > maxTimeDiff)
                THROW_EXCEPTION_FMT(
                    "keyframe #%zu at t=%.6f has no TUM pose within %.3f s (nearest %.3f s away)",
                    index, *stamp, maxTimeDiff, offset);
            worstOffset = std::max(worstOffset, std::abs(offset));

            // changeCoordinatesReference(delta) sets pose := delta (+) pose, for
            // both the mean and the covariance, whatever CPose3DPDF subclass
            // the keyframe stores. delta (+) old == target  =>  delta = target (+) old^-1.
            const auto old   = pose->getMeanVal();
            const auto delta = *target + (-old);
            pose->changeCoordinatesReference(delta);
            const double check = (pose->getMeanVal() - *target).norm();
            ASSERTMSG_(check < 1e-6, "pose update did not land on the target pose");
            ++index;
        }

        if (!sm.saveToFile(outputPath))
            THROW_EXCEPTION_FMT("cannot write simplemap '%s'", outputPath.c_str());
        std::cout << "sm-set-poses: replaced " << index << " keyframe poses (worst timestamp offset "
                  << worstOffset * 1e3 << " ms) -> '" << outputPath << "'\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "sm-set-poses: " << e.what() << "\n";
        return 1;
    }
}
