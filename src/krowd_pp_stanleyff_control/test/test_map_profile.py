import hashlib
import os

import pytest

from krowd_pp_stanleyff_control.map_profile import (
    find_profile,
    list_profiles,
    load_profile,
    route_only_profile,
    run_name_of,
)

WAYPOINTS = "idx,utm_zone,utm_x,utm_y,utm_z,yaw_rad\n" + "".join(
    f"{i},52N,{300000 + i},{4100000.0},0.0,0.0\n" for i in range(20)
)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_maps(tmp_path, sha=None, missions="missions:\n  list: []\n", lanes=False):
    maps = tmp_path / "maps"
    run = maps / "run_a"
    profile = "name: test\nrun_name: run_a\nmissions: missions.yaml\n"
    if sha:
        profile += f"waypoints_sha256: {sha}\nexpected_waypoint_count: 20\n"
    if lanes:
        profile += "lane_boundaries: lanes.csv\n"
    _write(str(run / "profile.yaml"), profile)
    _write(str(run / "missions.yaml"), missions)
    map_dir = tmp_path / "krowd_maps" / "run_a"
    _write(str(map_dir / "waypoints.csv"), WAYPOINTS)
    return str(maps), str(map_dir)


def test_run_name_resolves_symlinks(tmp_path):
    real = tmp_path / "20260913_103732"
    real.mkdir()
    link = tmp_path / "latest"
    try:
        os.symlink(real, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert run_name_of(str(link)) == "20260913_103732"
    assert run_name_of(str(real) + "/") == "20260913_103732"
    assert run_name_of("") == ""


def test_unknown_run_is_route_only(tmp_path):
    maps, map_dir = make_maps(tmp_path)
    p = find_profile(maps, "run_b", None)
    assert p.route_only and p.problem is None and not p.has_lane_map and not p.has_missions
    cfg = p.mission_config()
    assert cfg["lane_boundaries"] is None and cfg["missions"]["list"] == []
    assert "route_only" in p.label()


def test_profile_loads_and_checks_the_waypoints_sha(tmp_path):
    maps, map_dir = make_maps(tmp_path, sha=_sha(WAYPOINTS))
    p = find_profile(maps, "run_a", map_dir)
    assert p.problem is None and not p.route_only and p.has_missions
    assert p.waypoints_csv.endswith("waypoints.csv")
    cfg = p.mission_config()
    assert cfg["route"]["expected_trajectory_id_prefix"] == "run_a:20wp"
    assert list_profiles(maps) == ["run_a"]


def test_a_different_waypoints_file_is_a_blocking_problem_not_route_only(tmp_path):
    maps, map_dir = make_maps(tmp_path, sha="ab" * 32)
    p = find_profile(maps, "run_a", map_dir)
    assert not p.route_only
    assert p.problem is not None and "SHA-256 mismatch" in p.problem
    with pytest.raises(ValueError):
        load_profile(os.path.join(maps, "run_a"), "run_a", map_dir)
    # An all-digit digest is read by YAML as integer 0: it must be REFUSED,
    # never swallowed as "no SHA configured".
    maps, map_dir = make_maps(tmp_path / "digits", sha="0" * 64)
    p = find_profile(maps, "run_a", map_dir)
    assert p.problem is not None and "SHA-256" in p.problem and not p.route_only
    maps, map_dir = make_maps(tmp_path / "short", sha="abc")
    assert "SHA-256" in find_profile(maps, "run_a", map_dir).problem


def test_an_unquoted_numeric_run_name_is_refused_with_advice(tmp_path):
    maps = tmp_path / "maps"
    run = maps / "20260913_103732"
    _write(str(run / "profile.yaml"), "name: x\nrun_name: 20260913_103732\n")
    p = find_profile(str(maps), "20260913_103732", None)
    assert p.problem is not None and "QUOTED" in p.problem
    _write(str(run / "profile.yaml"), 'name: x\nrun_name: "20260913_103732"\n')
    assert find_profile(str(maps), "20260913_103732", None).problem is None


def test_declared_lane_map_that_is_missing_is_a_problem(tmp_path):
    maps, map_dir = make_maps(tmp_path, lanes=True)
    p = find_profile(maps, "run_a", map_dir)
    assert p.problem is not None and "lane_boundaries" in p.problem


def test_route_only_profile_uses_the_map_dir_waypoints_when_present(tmp_path):
    maps, map_dir = make_maps(tmp_path)
    p = route_only_profile("run_a", map_dir)
    assert p.waypoints_csv == os.path.join(map_dir, "waypoints.csv")
    assert route_only_profile("x", None).waypoints_csv is None
