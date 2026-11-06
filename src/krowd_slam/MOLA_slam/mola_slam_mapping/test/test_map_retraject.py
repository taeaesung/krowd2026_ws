import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mola_slam_mapping.map_retraject import (
    Options,
    Sigmas,
    chain_edges_to_keep,
    drop_reference_spikes,
    interpolate_reference,
    native_georeference,
    read_tum,
    retraject,
    up_vector_in_map,
    write_tum,
)

SPEED_MPS = 4.0


def _truth_pose(s):
    """Analytic ground-truth pose along a gently curving, undulating course."""

    s = np.asarray(s, dtype=float)
    x, y, z = s, 20.0 * np.sin(s / 60.0), 0.5 * np.sin(s / 80.0)
    dy, dz = 20.0 / 60.0 * np.cos(s / 60.0), 0.5 / 80.0 * np.cos(s / 80.0)
    yaw = np.arctan2(dy, 1.0)
    pitch = -np.arctan2(dz, np.hypot(1.0, dy))
    rotations = Rotation.from_euler("ZYX", np.column_stack([yaw, pitch, np.zeros_like(s)]))
    return np.column_stack([x, y, z]), rotations


def _drifting_lidar_odometry(truth_p, truth_r, pitch_drift_deg_per_step, seed=0):
    """Integrate the true relative poses with a slow pitch bias, in an odom frame."""

    rng = np.random.default_rng(seed)
    inverse = truth_r[:-1].inv()
    rel_t = inverse.apply(truth_p[1:] - truth_p[:-1])
    rel_r = inverse * truth_r[1:]
    bias = Rotation.from_euler("y", pitch_drift_deg_per_step, degrees=True)

    origin_r = Rotation.from_euler("z", 40.0, degrees=True)
    positions = [np.array([5.0, -3.0, 0.0])]
    rotations = [origin_r]
    for t, r in zip(rel_t, rel_r):
        noisy_t = t + rng.normal(0.0, 0.01, 3)
        positions.append(positions[-1] + rotations[-1].apply(noisy_t))
        rotations.append(rotations[-1] * r * bias)
    return np.asarray(positions), Rotation.concatenate(rotations)


def _reference(duration_s, seed=1):
    """5 Hz reference: true position + 2 cm noise, true roll/pitch, noisy yaw."""

    rng = np.random.default_rng(seed)
    # Starts before and ends after the keyframes so every one is bracketed.
    stamps = np.arange(-0.1, duration_s + 0.3, 0.2)
    positions, rotations = _truth_pose(np.clip(stamps, 0.0, None) * SPEED_MPS)
    positions = positions + rng.normal(0.0, 0.02, positions.shape)
    yaw, pitch, roll = rotations.as_euler("ZYX").T
    yaw = yaw + np.radians(rng.normal(0.0, 3.0, len(yaw)))
    return stamps, positions, Rotation.from_euler("ZYX", np.column_stack([yaw, pitch, roll])).as_quat()


def _scenario(keyframes=300, pitch_drift_deg_per_step=0.01):
    stamps = np.arange(keyframes) * (1.0 / SPEED_MPS)
    truth_p, truth_r = _truth_pose(stamps * SPEED_MPS)
    lo_p, lo_r = _drifting_lidar_odometry(truth_p, truth_r, pitch_drift_deg_per_step)
    ref_t, ref_p, ref_q = _reference(stamps[-1])
    return stamps, truth_p, truth_r, lo_p, lo_r.as_quat(), ref_t, ref_p, ref_q


def test_growing_tilt_is_removed_while_the_chain_stays_locally_rigid():
    """The case a rigid placement cannot fix: pitch drifting over the run."""

    stamps, truth_p, truth_r, lo_p, lo_q, ref_t, ref_p, ref_q = _scenario()

    positions, quaternions, statistics = retraject(
        stamps, lo_p, lo_q, ref_t, ref_p, ref_q, [0.0, 0.0, 1.0], Options(Sigmas())
    )

    # 3 deg of accumulated pitch leaves the rigidly placed chain off by well
    # over RTK noise; the graph brings every keyframe back within it.
    assert statistics["before"]["position_error_m"]["max"] > 0.5
    assert statistics["before"]["attitude_error_deg"]["median"] > 1.0
    error = np.linalg.norm(positions - truth_p, axis=1)
    assert np.median(error) < 0.05
    assert error.max() < 0.2
    attitude = (Rotation.from_quat(quaternions) * truth_r.inv()).magnitude()
    assert np.degrees(np.median(attitude)) < 0.3
    # ... by bending it gently (the synthetic LiDAR steps carry 1 cm of noise),
    # not by tearing consecutive keyframes apart. Per-step deformation is what
    # smears the rebuilt map, so it is the number that must stay small.
    assert statistics["after"]["chain_deformation_m"]["median"] < 0.05
    assert statistics["after"]["chain_deformation_m"]["max"] < 0.1
    # The default weighting is the one tuned for an RTK-frame product: follow
    # the reference closely. A change here would silently alter every map.
    assert Sigmas().between_translation_m == 0.02
    assert Sigmas().position_xy_m == 0.10
    assert statistics["after"]["chain_deformation_deg"]["max"] < 0.1
    assert statistics["matched_keyframes"] == len(stamps)
    assert statistics["solver"]["cost_after"] < statistics["solver"]["cost_before"]
    # A result the optimizer merely ran out of budget on is not a result.
    assert statistics["solver"]["converged"] is True


def test_yaw_of_the_reference_is_not_trusted():
    """The reference yaw is a GNSS course; corrupting it must change nothing."""

    stamps, truth_p, _, lo_p, lo_q, ref_t, ref_p, ref_q = _scenario(keyframes=120)
    yaw, pitch, roll = Rotation.from_quat(ref_q).as_euler("ZYX").T
    spoiled = Rotation.from_euler("ZYX", np.column_stack([yaw + 1.0, pitch, roll])).as_quat()

    clean, _, _ = retraject(stamps, lo_p, lo_q, ref_t, ref_p, ref_q, [0.0, 0.0, 1.0], Options(Sigmas()))
    bad_yaw, _, _ = retraject(stamps, lo_p, lo_q, ref_t, ref_p, spoiled, [0.0, 0.0, 1.0], Options(Sigmas()))

    assert np.allclose(clean, bad_yaw, atol=1e-3)


def test_refuses_when_the_reference_does_not_cover_the_run():
    stamps, _, _, lo_p, lo_q, ref_t, ref_p, ref_q = _scenario(keyframes=120)
    covered = ref_t < stamps[len(stamps) // 4]

    with pytest.raises(ValueError, match="bracketing reference"):
        retraject(
            stamps, lo_p, lo_q, ref_t[covered], ref_p[covered], ref_q[covered],
            [0.0, 0.0, 1.0], Options(Sigmas()),
        )


def test_reference_interpolation_only_bridges_short_gaps():
    stamps = np.array([0.0, 0.2, 0.4, 2.0])
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [10.0, 0, 0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))

    p, _, valid = interpolate_reference(stamps, positions, quaternions, [0.1, 0.3, 1.0, -0.1, 3.0], 0.5)

    assert valid.tolist() == [True, True, False, False, False]
    assert np.allclose(p[:2], [[0.5, 0, 0], [1.5, 0, 0]])


def test_reference_spikes_are_dropped_against_the_last_kept_sample():
    stamps = np.array([0.0, 0.2, 0.4, 0.6])
    positions = np.array([[0.0, 0, 0], [0.8, 0, 0], [30.0, 0, 0], [2.4, 0, 0]])

    assert drop_reference_spikes(stamps, positions, 25.0).tolist() == [True, True, False, True]


def test_lidar_chain_breaks_only_a_long_inconsistent_gap():
    stamps = np.array([0.0, 5.0, 5.3, 8.8])
    # The first long interval is a legitimate sparse-keyframe interval.  The
    # last is an LO outage: reference travelled 10 m while LO advanced 1 m.
    placed = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    reference = np.array([[0.0, 0, 0], [1.1, 0, 0], [2.1, 0, 0], [12.1, 0, 0]])

    keep, disagreement = chain_edges_to_keep(
        stamps, placed, reference, np.ones(4, dtype=bool), 1.0, 2.0
    )

    assert keep.tolist() == [True, True, False]
    assert disagreement[0] == pytest.approx(0.1)
    assert disagreement[2] == pytest.approx(9.0)


def test_tum_round_trip(tmp_path):
    path = tmp_path / "keyframes.tum"
    stamps = np.array([1.5, 1.25])
    positions = np.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]])
    quaternions = np.array([[0.0, 0.0, 0.7071068, 0.7071068], [0.0, 0.0, 0.0, 1.0]])

    write_tum(path, stamps, positions, quaternions)
    read_stamps, read_positions, read_quaternions = read_tum(path)

    # Sorted by stamp on read.
    assert read_stamps.tolist() == [1.25, 1.5]
    assert np.allclose(read_positions, positions[::-1])
    assert np.allclose(read_quaternions, quaternions[::-1])


def test_up_vector_follows_the_map_tilt():
    tilted = Rotation.from_euler("y", 5.0, degrees=True).as_quat()  # enu <- map

    up = up_vector_in_map(tilted)

    assert np.allclose(Rotation.from_quat(tilted).apply(up), [0.0, 0.0, 1.0])
    assert up[2] == pytest.approx(math.cos(math.radians(5.0)))


def test_native_georeference_describes_the_map_frame():
    manifest = {
        "geodetic_origin_wgs84": {"latitude": 37.2, "longitude": 126.7, "height": 28.8},
        "enu_to_localization_frame": {
            "translation": {"x": -1.25, "y": -2.17, "z": -1.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": math.sin(math.radians(45.0)), "w": math.cos(math.radians(45.0))},
        },
    }

    native = native_georeference(manifest)

    assert native["type"] == "mp2p_icp::Georeferencing" and native["defined"] is True
    assert native["geo_coord"] == {"lon": 126.7, "lat": 37.2, "altitude": 28.8}
    mean = native["T_enu_to_map"]["mean"]
    assert (mean["x"], mean["y"], mean["z"]) == (-1.25, -2.17, -1.0)
    assert mean["yaw_deg"] == pytest.approx(90.0)
    assert mean["pitch_deg"] == pytest.approx(0.0, abs=1e-9)
    assert native["T_enu_to_map"]["cov"]["rows"] == 6
