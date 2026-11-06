from glob import glob
from setuptools import find_packages, setup


package_name = "krowd_path_tracking"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="krowd",
    maintainer_email="krowd2026@gmail.com",
    description="Pure Pursuit and Stanley path tracking against a mapped route.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "route_publisher_node = krowd_path_tracking.route_publisher_node:main",
            "vehicle_state_node = krowd_path_tracking.vehicle_state_node:main",
            "pure_pursuit_node = krowd_path_tracking.pure_pursuit_node:main",
            "stanley_node = krowd_path_tracking.stanley_node:main",
        ],
    },
)
