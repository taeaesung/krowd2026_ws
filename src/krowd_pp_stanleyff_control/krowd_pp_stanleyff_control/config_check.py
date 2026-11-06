"""CLI: print the configuration readiness report without ROS.

    ros2 run krowd_pp_stanleyff_control config_check [--control PATH] [--map RUN_NAME|DIR] [--map-dir DIR]

Without --map, every profile under config/maps is checked; a run without a
profile directory is what route_only mode handles and needs no check.
"""

from __future__ import annotations

import argparse
import os
import sys

from .config import CONTROL_CONFIG, default_config_dir, default_maps_dir, load_control_config, readiness
from .map_profile import find_profile, list_profiles, load_profile, route_only_profile
from .steering_calibration import SteeringCalibration


def resolve_profile(maps_dir: str, spec: str | None, map_dir: str | None):
    if not spec:
        return None
    if os.path.isdir(os.path.expanduser(spec)):
        directory = os.path.abspath(os.path.expanduser(spec))
        return load_profile(directory, os.path.basename(directory.rstrip("/")), map_dir)
    return find_profile(maps_dir, spec, map_dir)


def check_one(control: dict, profile, label: str) -> bool:
    mission = profile.mission_config() if profile.problem is None else route_only_profile(profile.run_name, profile.map_dir).mission_config()
    report = readiness(control, mission, {"segments": mission.get("segments", {"list": []})}, check_files=True)
    if profile.problem:
        report.blocking.insert(0, profile.problem)
    for w in profile.warnings:
        report.warnings.append(w)
    print(f"\n=== map {label}: {profile.label()} ===")
    print(report.summary())
    return report.ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", default=os.path.join(default_config_dir(), CONTROL_CONFIG))
    parser.add_argument("--overrides", default="")
    parser.add_argument("--maps-dir", default=default_maps_dir())
    parser.add_argument("--map", default=None, help="run name under maps-dir, or a profile directory")
    parser.add_argument("--map-dir", default=None, help="localization map directory holding waypoints.csv (for the SHA check)")
    parser.add_argument("--route-only", action="store_true", help="check the route_only configuration only")
    args = parser.parse_args(argv)
    control = load_control_config(args.control, args.overrides or None)
    print(f"control config : {args.control}" + (f" + {args.overrides}" if args.overrides else ""))
    print(f"maps dir       : {args.maps_dir}")
    try:
        print(SteeringCalibration.from_config(control["steering"]).describe())
    except (KeyError, ValueError) as error:
        print(f"steering table invalid: {error}")

    ok = True
    if args.route_only:
        ok &= check_one(control, route_only_profile("(any)", args.map_dir), "route_only")
    elif args.map:
        try:
            profile = resolve_profile(args.maps_dir, args.map, args.map_dir)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"map profile '{args.map}' unusable: {error}")
            return 1
        ok &= check_one(control, profile, args.map)
    else:
        names = list_profiles(args.maps_dir)
        if not names:
            print("(no map profiles found)")
        for name in names:
            ok &= check_one(control, find_profile(args.maps_dir, name, args.map_dir), name)
        ok &= check_one(control, route_only_profile("(any)", None), "route_only")
    print("\nOVERALL:", "READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
