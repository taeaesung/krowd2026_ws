"""Synthetic routes shared by the geometry and speed-profile tests.

Factory fixtures rather than an importable helper module: a ROS 2 ament_python
package's `test/` directory is not itself a package, so a plain
`from test.helpers import ...` works only from certain working directories.
"""

import math

import pytest


@pytest.fixture
def straight_route():
    def build(count: int, spacing: float = 1.0) -> list[tuple[float, float]]:
        return [(index * spacing, 0.0) for index in range(count)]

    return build


@pytest.fixture
def circular_route():
    def build(
        radius: float, count: int, spacing: float = 1.0
    ) -> list[tuple[float, float]]:
        """Counter-clockwise arc - a left turn - sampled every `spacing` metres."""

        step = spacing / radius
        return [
            (
                radius * math.sin(index * step),
                radius * (1.0 - math.cos(index * step)),
            )
            for index in range(count)
        ]

    return build
