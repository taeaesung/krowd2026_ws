import math

import numpy as np

from krowd_pp_stanleyff_control.route import RouteModel


def straight_route(length_m=200.0, spacing=1.0, speed=5.0):
    n = int(length_m / spacing) + 1
    xs = np.arange(n) * spacing
    xy = np.column_stack([xs, np.zeros(n)])
    return RouteModel.from_arrays(xy, np.zeros(n), np.zeros(n), np.full(n, speed), "map", "test:%dwp" % n, 10**18)


def arc_route(radius=20.0, straight=40.0, spacing=1.0, speed=5.0):
    """Straight along +x, then a left quarter circle, then straight along +y."""

    pts = [(x, 0.0) for x in np.arange(0.0, straight, spacing)]
    n_arc = int(0.5 * math.pi * radius / spacing)
    for k in range(n_arc + 1):
        ang = k / n_arc * 0.5 * math.pi
        pts.append((straight + radius * math.sin(ang), radius - radius * math.cos(ang)))
    end = pts[-1]
    for k in range(1, int(straight / spacing) + 1):
        pts.append((end[0], end[1] + k * spacing))
    xy = np.array(pts)
    n = len(xy)
    curv = np.zeros(n)
    curv[len(range(int(straight / spacing))) : len(range(int(straight / spacing))) + n_arc + 1] = 1.0 / radius
    return RouteModel.from_arrays(xy, np.zeros(n), curv, np.full(n, speed), "map", "arc:%dwp" % n, 10**18)
