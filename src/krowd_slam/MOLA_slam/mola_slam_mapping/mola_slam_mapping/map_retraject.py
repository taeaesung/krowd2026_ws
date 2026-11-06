"""Re-trajectory a mapping run's simplemap keyframes onto the RTK reference.

Why: the point-cloud map (map.simplemap -> global_map.mm) is assembled from
LiDAR odometry's own keyframe poses, which drift - in position, and on flat
sites visibly as a growing pitch/roll tilt of the odom frame. The single rigid
placement waypoint_saver_node fits (pointcloud_map_placement) removes only the
constant part. The route already follows the RTK pose (/gnss_imu/pose); this
tool brings the map onto that same trajectory so LiDAR localization against it
agrees with the saved route.

How: a small pose graph over the keyframes, solved with scipy.

  between   LiDAR odometry's relative pose between consecutive keyframes -
            locally excellent, and what keeps the map locally rigid so that
            ICP still matches it afterwards.
  position  the reference pose (/gnss_imu/pose: RTK antenna minus lever arm,
            in map) interpolated at each keyframe stamp.
  gravity   the up direction from the reference orientation. Its roll/pitch
            come from the ESKF IMU (gnss_imu_pose_node), its yaw is a GNSS
            course and is deliberately NOT used - yaw follows from the
            positions and the relative rotations.

Soft-L1 loss: an isolated ICP jump or RTK glitch bends the chain locally
instead of steering the whole solve. The optimized poses are only slightly
different from the rigidly placed LiDAR chain wherever LiDAR was healthy; a
stretch where LiDAR odometry stopped tracking is moved to where RTK says it
was, but the scans recorded there stay as smeared as they were.

Inputs, all inside one run directory (~/krowd_maps/<run_id>):
  keyframes_lo.tum          sm-cli export-keyframes map.simplemap  (odom)
  trajectory_reference.csv  trajectory_recorder_node               (map)
  georeference.yaml         waypoint_saver_node (enu<-map, WGS84 origin)
Outputs:
  keyframes_corrected.tum   keyframe poses in map, for sm-set-poses
  retraject.yaml            residual statistics before / after
  <map>.georef.yaml         native georeference of the rebuilt map (enu<-map)
scripts/retraject_map.sh chains this with sm-cli, sm-set-poses and sm2mm.
"""

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation, Slerp

from mola_slam_mapping.trajectory_io import read_trajectory_csv
from mola_slam_mapping.waypoint_saver_node import (
    _mrpt_diagonal_covariance,
    _yaml_builtin_scalars,
    euler_from_quaternion,
    fit_rigid_transform,
)


@dataclass
class Sigmas:
    """1-sigma weights of the graph's factors.

    THESE CONTROL A REAL TRADE-OFF, measured on testweek_0905_3 (333 keyframes,
    516 m, 74 m of lost LiDAR odometry to repair). Loosening the chain lets the
    trajectory sit closer to RTK, but every centimetre of per-step chain
    deformation is a centimetre of scan-to-scan disagreement, and that smears
    the rebuilt map's surfaces - which is what ICP localization actually needs.
    Surface thickness was measured by sampling the rebuilt cloud and taking the
    smallest eigenvalue of each neighbourhood's covariance:

      between (m, /m, deg)   position (xy, z)   traj err   chain def   thickness
      0.020 0.050 0.05       0.10 0.15           1.29 cm    6.01 cm     0.146 m   <- default
      0.005 0.010 0.02       0.10 0.15           4.40 cm    3.48 cm     0.135 m
      0.005 0.010 0.02       0.30 0.30          10.03 cm    3.16 cm     0.133 m
      0.002 0.005 0.01       0.50 0.50          27.84 cm    1.09 cm     0.094 m

    For reference on the same run: rebuilding from the UNTOUCHED poses through
    the same pipeline gives 0.071 m, and the live map LiDAR odometry built
    incrementally from every scan (not just the 333 keyframes) gives 0.022 m.
    So the rebuild path itself costs most of the sharpness and the
    re-trajectory roughly doubles what is left; see AI_CONTEXT.md.

    THE DEFAULT IS THE FIRST ROW, because of what the corrected map is FOR:
    RTK is the reference frame and this map is a product placed inside it -
    for visualization, obstacle context and checking the recorded corridor -
    not a source the vehicle localizes against (race day localizes with RTK).
    Surface thickness only hurts ICP registration, so 0.146 m costs that use
    nothing, while the 1.3 cm trajectory is the whole point. The middle rows
    are a bad trade for this purpose: row 2 gives up 3.4x of trajectory
    accuracy to gain 7% of thickness. Move to the LAST row only if the map has
    to be localized against - and note that even 0.094 m did not make ICP work
    on testweek_0905_3, so that move is not currently worth making.
    """

    between_translation_m: float = 0.02
    between_translation_per_m: float = 0.05
    between_rotation_deg: float = 0.05
    # RTK is ~1.5 cm; the lever arm rotated by a slightly wrong course heading
    # is what dominates (2.5 m x 2 deg = 9 cm).
    position_xy_m: float = 0.10
    position_z_m: float = 0.15
    gravity_deg: float = 0.5


@dataclass
class Options:
    sigmas: Sigmas
    # A reference sample is only interpolated between two neighbours closer
    # than this; a longer GNSS gap must not be bridged by a straight line.
    reference_max_gap_s: float = 0.5
    # Same rule as pose_fusion_node: >5 m between two 5 Hz samples is a spike.
    reference_max_speed_mps: float = 25.0
    # A long interval is not itself an odometry failure (the vehicle may have
    # been stationary and no keyframe was needed).  Break the LiDAR between
    # factor only when a long interval also disagrees with the reference
    # displacement.  Otherwise a multi-second LO outage is represented as one
    # false short step and forces the optimizer either to tear the map or miss
    # the global trajectory by many metres.
    lidar_chain_break_min_gap_s: float = 1.0
    lidar_chain_break_disagreement_m: float = 2.0
    placement_outlier_m: float = 3.0
    # Refuse to write a trajectory that the reference could not pin down.
    min_matched_fraction: float = 0.5
    # Down-weight residuals beyond this many sigma (scipy soft_l1 f_scale).
    robust_scale_sigma: float = 3.0
    # Measured on testweek_0905_3 (333 keyframes = 1998 unknowns, 74 m of lost
    # LiDAR odometry to undo). scipy's default ftol of 1e-8 never terminates
    # here: soft_l1 keeps shaving thousandths off the one huge residual, so the
    # solve ran the whole 2000-evaluation ceiling in 75 s and reported "maximum
    # number of function evaluations exceeded" - a result that is at the answer
    # but cannot say so. 1e-6 stops at the same answer (median keyframe error
    # 1.288 cm vs 1.289 cm) in 163 evaluations and 6 s.
    function_tolerance: float = 1.0e-6
    max_iterations: int = 500


def read_tum(path):
    """(stamps s, positions (N,3), quaternions (N,4) xyzw) from a TUM file."""

    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != 8:
        raise ValueError(f"'{path}' is not a TUM trajectory (8 columns expected)")
    order = np.argsort(data[:, 0], kind="stable")
    data = data[order]
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def write_tum(path, stamps, positions, quaternions) -> None:
    with open(path, "w") as stream:
        stream.write("# timestamp x y z qx qy qz qw\n")
        for stamp, p, q in zip(stamps, positions, quaternions):
            stream.write(
                f"{stamp:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{q[0]:.7f} {q[1]:.7f} {q[2]:.7f} {q[3]:.7f}\n"
            )


def drop_reference_spikes(stamps_s, positions, max_speed_mps: float):
    """Mask of samples whose speed from the previous kept sample is plausible."""

    keep = np.ones(len(stamps_s), dtype=bool)
    last = 0
    for index in range(1, len(stamps_s)):
        dt = stamps_s[index] - stamps_s[last]
        if dt <= 0.0:
            keep[index] = False
            continue
        if np.linalg.norm(positions[index] - positions[last]) / dt > max_speed_mps:
            keep[index] = False
            continue
        last = index
    return keep


def interpolate_reference(ref_stamps_s, ref_positions, ref_quaternions, query_s, max_gap_s):
    """Reference pose at each query stamp: (positions, quaternions, valid mask).

    Linear in position, SLERP in orientation, only between two reference
    samples that bracket the query and are at most max_gap_s apart.
    """

    query_s = np.asarray(query_s, dtype=float)
    count = len(ref_stamps_s)
    index = np.searchsorted(ref_stamps_s, query_s)
    valid = (index > 0) & (index < count)
    before = np.clip(index - 1, 0, count - 1)
    after = np.clip(index, 0, count - 1)
    gap = ref_stamps_s[after] - ref_stamps_s[before]
    valid &= (gap > 0.0) & (gap <= max_gap_s)

    positions = np.full((len(query_s), 3), np.nan)
    quaternions = np.full((len(query_s), 4), np.nan)
    if valid.any():
        fraction = np.zeros(len(query_s))
        fraction[valid] = (query_s[valid] - ref_stamps_s[before[valid]]) / gap[valid]
        positions[valid] = ref_positions[before[valid]] + fraction[valid, None] * (
            ref_positions[after[valid]] - ref_positions[before[valid]]
        )
        slerp = Slerp(ref_stamps_s, Rotation.from_quat(ref_quaternions))
        quaternions[valid] = slerp(query_s[valid]).as_quat()
    return positions, quaternions, valid


def up_vector_in_map(enu_from_map_quaternion_xyzw) -> np.ndarray:
    """Gravity-opposite direction expressed in the map frame."""

    return Rotation.from_quat(enu_from_map_quaternion_xyzw).inv().apply([0.0, 0.0, 1.0])


def relative_poses(positions, rotations: Rotation):
    """Body-frame translation and rotation from each pose to the next."""

    inverse = rotations[:-1].inv()
    return inverse.apply(positions[1:] - positions[:-1]), inverse * rotations[1:]


class KeyframeGraph:
    """Residuals of the keyframe pose graph, parametrized as local perturbations.

    Keyframe k is p0_k + dp_k, exp(dr_k) * R0_k, so the unknowns start at zero
    and rotation vectors never approach the pi singularity whatever the
    vehicle heading.
    """

    def __init__(self, p0, r0: Rotation, lo_positions, lo_rotations: Rotation,
                 ref_index, ref_positions, ref_rotations: Rotation, up, sigmas: Sigmas,
                 active_edges=None):
        self.count = len(p0)
        self.p0 = np.asarray(p0, dtype=float)
        self.r0 = r0
        self.ref_index = np.asarray(ref_index, dtype=int)
        self.ref_positions = np.asarray(ref_positions, dtype=float)
        self.up = np.asarray(up, dtype=float)
        # What the reference says the up direction is, seen from the body.
        self.gravity_body = ref_rotations.inv().apply(self.up)

        lo_rel_t, lo_rel_rot = relative_poses(lo_positions, lo_rotations)
        if active_edges is None:
            active_edges = np.ones(self.count - 1, dtype=bool)
        self.active_edges = np.asarray(active_edges, dtype=bool)
        if self.active_edges.shape != (self.count - 1,):
            raise ValueError("active_edges must have one entry per keyframe edge")
        self.active_edge_index = np.flatnonzero(self.active_edges)
        self.lo_rel_t = lo_rel_t[self.active_edges]
        self.lo_rel_rot = lo_rel_rot[self.active_edges]
        self.lo_rel_rot_inv = self.lo_rel_rot.inv()
        step = np.linalg.norm(self.lo_rel_t, axis=1)
        self.sigma_t = sigmas.between_translation_m + sigmas.between_translation_per_m * step
        self.sigma_r = math.radians(sigmas.between_rotation_deg)
        self.sigma_p = np.array([sigmas.position_xy_m, sigmas.position_xy_m, sigmas.position_z_m])
        self.sigma_g = math.sin(math.radians(sigmas.gravity_deg))

    def poses(self, x):
        delta = x.reshape(self.count, 6)
        return self.p0 + delta[:, :3], Rotation.from_rotvec(delta[:, 3:]) * self.r0

    def residuals(self, x):
        p, r = self.poses(x)
        rel_t, rel_rot = relative_poses(p, r)
        rel_t = rel_t[self.active_edges]
        rel_rot = rel_rot[self.active_edges]
        e_t = (rel_t - self.lo_rel_t) / self.sigma_t[:, None]
        e_r = (self.lo_rel_rot_inv * rel_rot).as_rotvec() / self.sigma_r
        e_p = (p[self.ref_index] - self.ref_positions) / self.sigma_p
        e_g = (r[self.ref_index].inv().apply(self.up) - self.gravity_body) / self.sigma_g
        return np.concatenate([e_t.ravel(), e_r.ravel(), e_p.ravel(), e_g.ravel()])

    def sparsity(self):
        n, m, edge_count = self.count, len(self.ref_index), len(self.active_edge_index)
        rows = 6 * edge_count + 6 * m
        pattern = lil_matrix((rows, 6 * n), dtype=np.int8)
        for row, k in enumerate(self.active_edge_index):
            columns = slice(6 * k, 6 * k + 12)
            pattern[3 * row: 3 * row + 3, columns] = 1
            pattern[3 * edge_count + 3 * row: 3 * edge_count + 3 * row + 3, columns] = 1
        offset = 6 * edge_count
        for i, k in enumerate(self.ref_index):
            pattern[offset + 3 * i: offset + 3 * i + 3, 6 * k: 6 * k + 3] = 1
            pattern[offset + 3 * m + 3 * i: offset + 3 * m + 3 * i + 3, 6 * k + 3: 6 * k + 6] = 1
        return pattern

    def solve(self, options: Options):
        result = least_squares(
            self.residuals,
            np.zeros(6 * self.count),
            jac="2-point",
            jac_sparsity=self.sparsity(),
            method="trf",
            loss="soft_l1",
            f_scale=options.robust_scale_sigma,
            x_scale="jac",
            ftol=options.function_tolerance,
            max_nfev=options.max_iterations,
        )
        p, r = self.poses(result.x)
        return p, r, result


def _percentiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _fit_statistics(graph: KeyframeGraph, p, r: Rotation) -> dict:
    """Position / attitude agreement with the reference and chain deformation."""

    error = p[graph.ref_index] - graph.ref_positions
    gravity = r[graph.ref_index].inv().apply(graph.up)
    cosine = np.clip(np.einsum("ij,ij->i", gravity, graph.gravity_body), -1.0, 1.0)
    rel_t, rel_rot = relative_poses(p, r)
    rel_t = rel_t[graph.active_edges]
    rel_rot = rel_rot[graph.active_edges]
    return {
        "position_error_m": _percentiles(np.linalg.norm(error, axis=1)),
        "z_error_m": _percentiles(np.abs(error[:, 2])),
        "attitude_error_deg": _percentiles(np.degrees(np.arccos(cosine))),
        "chain_deformation_m": _percentiles(np.linalg.norm(rel_t - graph.lo_rel_t, axis=1)),
        "chain_deformation_deg": _percentiles(
            np.degrees(np.linalg.norm((graph.lo_rel_rot_inv * rel_rot).as_rotvec(), axis=1))
        ),
    }


def chain_edges_to_keep(
    stamps, placed_positions, ref_positions, matched, min_gap_s, disagreement_m
):
    """Keep healthy LiDAR relative-pose factors across keyframe intervals."""

    keep = np.ones(len(stamps) - 1, dtype=bool)
    gaps = np.diff(stamps)
    comparable = matched[:-1] & matched[1:] & (gaps > min_gap_s)
    placed_delta = np.diff(placed_positions, axis=0)
    reference_delta = np.diff(ref_positions, axis=0)
    disagreement = np.linalg.norm(placed_delta - reference_delta, axis=1)
    keep[comparable & (disagreement > disagreement_m)] = False
    return keep, disagreement


def _tilt_diagnosis(p_rigid, lo_positions, ref_index, ref_positions) -> dict:
    """Is the rigid-placement z error constant, or growing along the run?

    A slope means the odom frame's tilt is not one value over the run and a
    rigid placement cannot absorb it - the reason the graph exists.
    """

    distance = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(lo_positions, axis=0), axis=1))])
    z_error = p_rigid[ref_index, 2] - ref_positions[:, 2]
    along = distance[ref_index]
    if len(along) < 2 or along.max() - along.min() < 1.0:
        return {"z_slope_deg": None}
    slope = np.polyfit(along, z_error, 1)[0]
    quarter = max(1, len(along) // 4)
    return {
        "z_slope_deg": float(math.degrees(math.atan(slope))),
        "z_error_first_quarter_m": float(np.mean(z_error[:quarter])),
        "z_error_last_quarter_m": float(np.mean(z_error[-quarter:])),
    }


def retraject(lo_stamps, lo_positions, lo_quaternions, ref_stamps, ref_positions,
              ref_quaternions, up, options: Options):
    """Optimize keyframe poses in the reference frame.

    Returns (positions, quaternions xyzw, statistics dict); raises ValueError
    when the reference cannot pin down enough keyframes.
    """

    keep = drop_reference_spikes(ref_stamps, ref_positions, options.reference_max_speed_mps)
    ref_stamps, ref_positions, ref_quaternions = (
        ref_stamps[keep], ref_positions[keep], ref_quaternions[keep]
    )
    ref_p, ref_q, matched = interpolate_reference(
        ref_stamps, ref_positions, ref_quaternions, lo_stamps, options.reference_max_gap_s
    )
    statistics = {
        "keyframes": int(len(lo_stamps)),
        "reference_samples": int(len(ref_stamps)),
        "reference_spikes_dropped": int((~keep).sum()),
        "matched_keyframes": int(matched.sum()),
    }
    if matched.sum() < max(3, options.min_matched_fraction * len(lo_stamps)):
        raise ValueError(
            f"only {int(matched.sum())} of {len(lo_stamps)} keyframes have a "
            "bracketing reference pose; refusing to re-trajectory the map"
        )
    ref_index = np.flatnonzero(matched)

    placement = fit_rigid_transform(
        lo_positions[ref_index], ref_p[ref_index], options.placement_outlier_m
    )
    if placement is None:
        raise ValueError("too few keyframe/reference pairs for a rigid placement")
    matrix, placement_stats = placement
    r_fit = Rotation.from_matrix(matrix[:3, :3])
    yaw, pitch, roll = r_fit.as_euler("ZYX", degrees=True)
    statistics["rigid_placement"] = {
        **placement_stats, "yaw_deg": float(yaw), "pitch_deg": float(pitch), "roll_deg": float(roll),
    }

    lo_rotations = Rotation.from_quat(lo_quaternions)
    p0 = r_fit.apply(lo_positions) + matrix[:3, 3]
    r0 = r_fit * lo_rotations
    active_edges, edge_disagreement = chain_edges_to_keep(
        lo_stamps, p0, ref_p, matched,
        options.lidar_chain_break_min_gap_s,
        options.lidar_chain_break_disagreement_m,
    )
    broken = np.flatnonzero(~active_edges)
    statistics["lidar_chain_breaks"] = [
        {
            "edge": int(index),
            "from_stamp": float(lo_stamps[index]),
            "to_stamp": float(lo_stamps[index + 1]),
            "gap_s": float(lo_stamps[index + 1] - lo_stamps[index]),
            "placement_reference_disagreement_m": float(edge_disagreement[index]),
        }
        for index in broken
    ]
    graph = KeyframeGraph(
        p0, r0, lo_positions, lo_rotations, ref_index, ref_p[ref_index],
        Rotation.from_quat(ref_q[ref_index]), up, options.sigmas, active_edges,
    )
    statistics["rigid_placement_tilt"] = _tilt_diagnosis(p0, lo_positions, ref_index, ref_p[ref_index])
    statistics["before"] = _fit_statistics(graph, p0, r0)

    p, r, result = graph.solve(options)
    statistics["after"] = _fit_statistics(graph, p, r)
    correction = np.linalg.norm(p - p0, axis=1)
    statistics["correction_from_rigid_placement_m"] = _percentiles(correction)
    statistics["solver"] = {
        # status 0 is scipy's "ran out of evaluations", i.e. the answer is
        # wherever the optimizer happened to be, not a minimum.
        "converged": bool(result.status > 0),
        "status": int(result.status),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "cost_before": float(0.5 * np.sum(graph.residuals(np.zeros(6 * graph.count)) ** 2)),
        "cost_after": float(result.cost),
    }
    statistics["sigmas"] = asdict(options.sigmas)
    return p, r.as_quat(), statistics


def native_georeference(manifest: dict) -> dict:
    """mp2p_icp::Georeferencing YAML for a map expressed in the manifest's map frame."""

    origin = manifest["geodetic_origin_wgs84"]
    enu_from_map = manifest["enu_to_localization_frame"]
    yaw, pitch, roll = euler_from_quaternion(SimpleNamespace(**enu_from_map["rotation"]))
    translation = enu_from_map["translation"]
    return {
        "type": "mp2p_icp::Georeferencing",
        "defined": True,
        "geo_coord": {
            "lon": origin["longitude"],
            "lat": origin["latitude"],
            "altitude": origin["height"],
        },
        "T_enu_to_map": {
            "mean": {
                "x": translation["x"],
                "y": translation["y"],
                "z": translation["z"],
                "yaw_deg": math.degrees(yaw),
                "pitch_deg": math.degrees(pitch),
                "roll_deg": math.degrees(roll),
            },
            "cov": _mrpt_diagonal_covariance(1e-6),
        },
    }


def _parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--keyframes", default="keyframes_lo.tum")
    parser.add_argument("--reference", default="trajectory_reference.csv")
    parser.add_argument("--output", default="keyframes_corrected.tum")
    parser.add_argument("--statistics", default="retraject.yaml")
    parser.add_argument(
        "--map-name", default="global_map_corrected.mm",
        help="rebuilt map file name; its .georef.yaml sidecar is written here",
    )
    defaults = Sigmas()
    for field, value in asdict(defaults).items():
        parser.add_argument(f"--{field.replace('_', '-')}", type=float, default=value)
    parser.add_argument("--reference-max-gap-s", type=float, default=Options.reference_max_gap_s)
    parser.add_argument(
        "--lidar-chain-break-min-gap-s", type=float,
        default=Options.lidar_chain_break_min_gap_s,
    )
    parser.add_argument(
        "--lidar-chain-break-disagreement-m", type=float,
        default=Options.lidar_chain_break_disagreement_m,
    )
    parser.add_argument("--min-matched-fraction", type=float, default=Options.min_matched_fraction)
    parser.add_argument("--max-iterations", type=int, default=Options.max_iterations)
    parser.add_argument("--function-tolerance", type=float, default=Options.function_tolerance)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_arguments(argv)
    run_dir = args.run_dir.expanduser()
    sigmas = Sigmas(**{field: getattr(args, field) for field in asdict(Sigmas())})
    options = Options(
        sigmas=sigmas,
        reference_max_gap_s=args.reference_max_gap_s,
        lidar_chain_break_min_gap_s=args.lidar_chain_break_min_gap_s,
        lidar_chain_break_disagreement_m=args.lidar_chain_break_disagreement_m,
        min_matched_fraction=args.min_matched_fraction,
        max_iterations=args.max_iterations,
        function_tolerance=args.function_tolerance,
    )

    lo_stamps, lo_positions, lo_quaternions = read_tum(run_dir / args.keyframes)
    ref_stamps_ns, frames, ref_positions, ref_quaternions = read_trajectory_csv(
        run_dir / args.reference
    )
    with open(run_dir / "georeference.yaml") as stream:
        manifest = yaml.safe_load(stream)
    route_frame = manifest.get("artifacts", {}).get("route", {}).get("source_frame", "map")
    if set(frames) != {route_frame}:
        print(
            f"error: '{args.reference}' is in frames {sorted(set(frames))}, "
            f"but the run's map frame is '{route_frame}'",
            file=sys.stderr,
        )
        return 1
    up = up_vector_in_map(
        [manifest["enu_to_localization_frame"]["rotation"][key] for key in ("x", "y", "z", "w")]
    )

    try:
        positions, quaternions, statistics = retraject(
            lo_stamps, lo_positions, lo_quaternions, ref_stamps_ns * 1e-9,
            ref_positions, ref_quaternions, up, options,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    statistics["frame"] = route_frame
    statistics["map_file"] = args.map_name
    write_tum(run_dir / args.output, lo_stamps, positions, quaternions)
    with open(run_dir / args.statistics, "w") as stream:
        yaml.safe_dump(_yaml_builtin_scalars(statistics), stream, sort_keys=False)
    georef_path = run_dir / Path(args.map_name).with_suffix(".georef.yaml").name
    with open(georef_path, "w") as stream:
        yaml.safe_dump(_yaml_builtin_scalars(native_georeference(manifest)), stream, sort_keys=False)

    # Per-step chain deformation is the number that predicts map SHARPNESS: it
    # is scan-to-scan disagreement, and it smears the surfaces ICP matches
    # against. A few cm is expected and harmless for a map that is a product
    # rather than a localization source (see Sigmas); this only shouts when it
    # is large enough to mean something else is wrong.
    deformation = statistics["after"]["chain_deformation_m"]["median"]
    if deformation is not None and deformation > 0.15:
        print(
            f"warning: the keyframe chain was deformed by {deformation * 100:.1f} cm per step "
            "(median) - far more than the few cm RTK noise alone produces. The reference is "
            "probably unreliable over this run, and the rebuilt map will be visibly smeared.",
            file=sys.stderr,
        )
    if not statistics["solver"]["converged"]:
        print(
            f"warning: the solver stopped without converging "
            f"({statistics['solver']['message']}). The poses below are wherever it had "
            "reached; raise --max-iterations and compare retraject.yaml's cost_after.",
            file=sys.stderr,
        )
    before, after = statistics["before"], statistics["after"]
    print(
        f"{statistics['matched_keyframes']}/{statistics['keyframes']} keyframes matched; "
        f"position error vs reference median {before['position_error_m']['median']:.2f} -> "
        f"{after['position_error_m']['median']:.2f} m, p95 {before['position_error_m']['p95']:.2f} -> "
        f"{after['position_error_m']['p95']:.2f} m; attitude {before['attitude_error_deg']['median']:.2f} -> "
        f"{after['attitude_error_deg']['median']:.2f} deg; chain deformed "
        f"{after['chain_deformation_m']['median'] * 100:.1f} cm per step (median), at most "
        f"{after['chain_deformation_m']['max']:.3f} m -- that per-step figure is what smears "
        "the rebuilt map's surfaces"
    )
    print(f"wrote {run_dir / args.output}, {run_dir / args.statistics}, {georef_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
