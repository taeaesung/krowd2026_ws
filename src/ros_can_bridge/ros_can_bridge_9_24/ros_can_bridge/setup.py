import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ros_can_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='klu5109',
    maintainer_email='klu5109@todo.todo',
    description='Bidirectional ROS2 and SocketCAN bridge',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'can_to_ros_node = ros_can_bridge.can_to_ros_node:main',
            'ros_to_can_node = ros_can_bridge.ros_to_can_node:main',
        ],
    },
)
