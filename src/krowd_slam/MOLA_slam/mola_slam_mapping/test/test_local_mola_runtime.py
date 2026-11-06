import os

from mola_slam_mapping.local_mola_runtime import build_local_mola_environment


def test_local_overlay_precedes_reference_and_stale_environment(tmp_path):
    overlay = tmp_path / "overlay"
    reference = tmp_path / "reference"
    local_package = overlay / "mola_launcher"
    reference_package = reference / "mola_launcher"
    stale_package = tmp_path / "stale" / "mola_launcher"
    for package in (local_package, reference_package, stale_package):
        (package / "lib").mkdir(parents=True)
        (package / "bin").mkdir()

    current = {
        "AMENT_PREFIX_PATH": os.pathsep.join(
            [str(reference_package), str(stale_package)]
        ),
        "CMAKE_PREFIX_PATH": str(reference_package),
        "COLCON_PREFIX_PATH": str(reference),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [str(reference_package / "lib"), str(stale_package / "lib")]
        ),
        "PATH": str(reference_package / "bin"),
    }

    environment = build_local_mola_environment(overlay, reference, current)

    assert environment["AMENT_PREFIX_PATH"].split(os.pathsep)[0] == str(
        local_package
    )
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(
        local_package / "lib"
    )
    assert environment["PATH"].split(os.pathsep)[0] == str(
        local_package / "bin"
    )
    assert environment["AMENT_PREFIX_PATH"].split(os.pathsep).count(
        str(reference_package)
    ) == 1
