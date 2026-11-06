from glob import glob
import os

from setuptools import find_packages, setup

package_name = "krowd_pp_stanleyff_control"


def map_data_files():
    """Every file under config/maps/<run>/ keeps its relative path in share/."""

    out = []
    for root, _dirs, files in os.walk(os.path.join("config", "maps")):
        if files:
            out.append(("share/" + package_name + "/" + root, [os.path.join(root, f) for f in files]))
    return out


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config/speed_modes", glob("config/speed_modes/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ] + map_data_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="krowd",
    maintainer_email="krowd2026@gmail.com",
    description="KROWD PP + Stanley blended controller (with curvature feedforward) with map profiles.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "control_node = krowd_pp_stanleyff_control.control_node:main",
            "route_publisher_node = krowd_pp_stanleyff_control.route_publisher_node:main",
            "vehicle_dynamics_sim_node = krowd_pp_stanleyff_control.vehicle_dynamics_sim_node:main",
            "config_check = krowd_pp_stanleyff_control.config_check:main",
            "static_input_report = krowd_pp_stanleyff_control.static_input_report:main",
        ],
    },
)
