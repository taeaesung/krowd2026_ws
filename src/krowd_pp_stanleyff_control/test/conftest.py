"""Test setup: import the package from the source tree, plus the synthetic
route fixtures the geometry / speed-profile tests (from krowd_stanley_control)
use. helpers.py holds the RouteModel factories the controller tests use."""

import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


@pytest.fixture
def straight_route():
    def build(count: int, spacing: float = 1.0) -> list[tuple[float, float]]:
        return [(index * spacing, 0.0) for index in range(count)]

    return build


@pytest.fixture
def circular_route():
    def build(radius: float, count: int, spacing: float = 1.0) -> list[tuple[float, float]]:
        """Counter-clockwise arc - a left turn - sampled every `spacing` metres."""

        step = spacing / radius
        return [
            (radius * math.sin(index * step), radius * (1.0 - math.cos(index * step)))
            for index in range(count)
        ]

    return build
