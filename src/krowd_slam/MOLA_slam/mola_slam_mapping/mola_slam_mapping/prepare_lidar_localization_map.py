"""Export an existing MOLA metric-map point layer to a PLY for GICP."""

import argparse
from pathlib import Path
import shutil
import subprocess

from ament_index_python.packages import get_package_prefix


def exported_layer_path(metric_map: Path, layer_name: str) -> Path:
    return metric_map.with_suffix("").with_name(
        f"{metric_map.stem}_{layer_name}.ply"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export the point layer of an already-built .mm map; this does not "
            "run mapping or modify the source map."
        )
    )
    parser.add_argument("metric_map", type=Path)
    parser.add_argument("--layer", default="localmap")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    metric_map = args.metric_map.expanduser().resolve()
    if not metric_map.is_file():
        parser.error(f"metric map does not exist: '{metric_map}'")
    if not args.layer or "/" in args.layer:
        parser.error(f"invalid layer name: '{args.layer}'")

    output = exported_layer_path(metric_map, args.layer)
    if output.exists() and not args.force:
        print(f"already exists: {output}")
        return

    executable = shutil.which("mm2ply")
    if executable is None:
        raise RuntimeError("mm2ply is not on PATH; source the MOLA workspace")
    plugin = (
        Path(get_package_prefix("mola_metric_maps"))
        / "lib"
        / "libmola_metric_maps.so"
    )
    if not plugin.exists():
        raise RuntimeError(f"mola_metric_maps plugin does not exist: '{plugin}'")

    subprocess.run(
        [
            executable,
            "-i",
            str(metric_map),
            "-o",
            str(metric_map.with_suffix("")),
            "--binary",
            "--export-fields",
            "x,y,z",
            "--ignore-missing-fields",
            "--load-plugins",
            str(plugin),
        ],
        check=True,
    )
    if not output.is_file():
        raise RuntimeError(
            f"MM layer '{args.layer}' was not exported; expected '{output}'"
        )
    print(f"ready: {output}")


if __name__ == "__main__":
    main()
