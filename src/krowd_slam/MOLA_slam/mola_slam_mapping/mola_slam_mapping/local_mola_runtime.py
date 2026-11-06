import os
from pathlib import Path
from typing import Mapping

from ament_index_python.packages import get_package_prefix


def _package_prefixes(install_root: Path) -> list[Path]:
    if not install_root.is_dir():
        return []
    return [
        child
        for child in sorted(install_root.iterdir())
        if child.is_dir()
        and ((child / "lib").is_dir() or (child / "share").is_dir())
    ]


def _prepend_unique(paths: list[Path | str], existing: str = "") -> str:
    ordered: list[str] = []
    for path in [str(item) for item in paths] + existing.split(os.pathsep):
        if path and path not in ordered:
            ordered.append(path)
    return os.pathsep.join(ordered)


def build_local_mola_environment(
    overlay_install_root: Path,
    reference_install_root: Path,
    current_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment with locally built MOLA ahead of prebuilts."""

    current = os.environ if current_environment is None else current_environment
    overlay_prefixes = _package_prefixes(overlay_install_root)
    reference_prefixes = _package_prefixes(reference_install_root)
    prefixes = overlay_prefixes + reference_prefixes
    library_paths = [prefix / "lib" for prefix in prefixes if (prefix / "lib").is_dir()]
    binary_paths = [prefix / "bin" for prefix in prefixes if (prefix / "bin").is_dir()]

    return {
        "AMENT_PREFIX_PATH": _prepend_unique(
            prefixes, current.get("AMENT_PREFIX_PATH", "")
        ),
        "CMAKE_PREFIX_PATH": _prepend_unique(
            prefixes, current.get("CMAKE_PREFIX_PATH", "")
        ),
        "COLCON_PREFIX_PATH": _prepend_unique(
            [overlay_install_root, reference_install_root],
            current.get("COLCON_PREFIX_PATH", ""),
        ),
        "LD_LIBRARY_PATH": _prepend_unique(
            library_paths, current.get("LD_LIBRARY_PATH", "")
        ),
        "PATH": _prepend_unique(binary_paths, current.get("PATH", "")),
    }


def local_mola_runtime() -> tuple[str, dict[str, str]]:
    """Return the local mola-cli path and a launch-safe runtime environment."""

    package_prefix = Path(get_package_prefix("mola_slam_mapping"))
    overlay_install_root = package_prefix.parent
    workspace_root = overlay_install_root.parent
    reference_install_root = (
        workspace_root
        / "src"
        / "mola_3_1_0_ws"
        / "mola_3_1_0_ws"
        / "install"
    )
    executable = (
        overlay_install_root
        / "mola_launcher"
        / "lib"
        / "mola_launcher"
        / "mola-cli"
    )
    if not executable.is_file():
        raise FileNotFoundError(
            "Locally built mola-cli is missing; expected " + str(executable)
        )

    environment = build_local_mola_environment(
        overlay_install_root, reference_install_root
    )
    return str(executable), environment
