"""Bit-exact port of osu!'s slider geometry (osu-framework PathApproximator +
osu.Game SliderPath), so slider-derived fruit/droplet x positions match the
game to the pixel.

Ported from:
  ppy/osu-framework  osu.Framework/Utils/PathApproximator.cs, CircularArcProperties.cs
  ppy/osu            osu.Game/Rulesets/Objects/SliderPath.cs (calculateLength)
"""
from __future__ import annotations

import bisect
import math

Vec = tuple

BEZIER_TOLERANCE = 0.25
CATMULL_DETAIL = 50
CIRCULAR_ARC_TOLERANCE = 0.1


def _hyp(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


# --- PathApproximator ---------------------------------------------------------

def approximate_linear(points):
    return list(points)


def _catmull_point(v1, v2, v3, v4, t):
    t2 = t * t
    t3 = t * t2
    x = 0.5 * (2 * v2[0] + (-v1[0] + v3[0]) * t
               + (2 * v1[0] - 5 * v2[0] + 4 * v3[0] - v4[0]) * t2
               + (-v1[0] + 3 * v2[0] - 3 * v3[0] + v4[0]) * t3)
    y = 0.5 * (2 * v2[1] + (-v1[1] + v3[1]) * t
               + (2 * v1[1] - 5 * v2[1] + 4 * v3[1] - v4[1]) * t2
               + (-v1[1] + 3 * v2[1] - 3 * v3[1] + v4[1]) * t3)
    return (x, y)


def approximate_catmull(points):
    out = []
    n = len(points)
    for i in range(n - 1):
        v1 = points[i - 1] if i > 0 else points[i]
        v2 = points[i]
        v3 = points[i + 1] if i < n - 1 else (2 * v2[0] - v1[0], 2 * v2[1] - v1[1])
        v4 = points[i + 2] if i < n - 2 else (2 * v3[0] - v2[0], 2 * v3[1] - v2[1])
        for c in range(CATMULL_DETAIL):
            out.append(_catmull_point(v1, v2, v3, v4, c / CATMULL_DETAIL))
            out.append(_catmull_point(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))
    return out


def _bezier_flat_enough(cps):
    for i in range(1, len(cps) - 1):
        dx = cps[i - 1][0] - 2 * cps[i][0] + cps[i + 1][0]
        dy = cps[i - 1][1] - 2 * cps[i][1] + cps[i + 1][1]
        if dx * dx + dy * dy > BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4:
            return False
    return True


def _bezier_subdivide(cps, left, right, count):
    mid = list(cps[:count])
    for i in range(count):
        left[i] = mid[0]
        right[count - i - 1] = mid[count - i - 1]
        for j in range(count - i - 1):
            mid[j] = ((mid[j][0] + mid[j + 1][0]) / 2, (mid[j][1] + mid[j + 1][1]) / 2)


def _bezier_approximate(cps, output, count):
    left = [(0.0, 0.0)] * (count * 2)
    right = [(0.0, 0.0)] * count
    _bezier_subdivide(cps, left, right, count)
    for i in range(count - 1):
        left[count + i] = right[i + 1]
    output.append(cps[0])
    for i in range(1, count - 1):
        idx = 2 * i
        output.append((0.25 * (left[idx - 1][0] + 2 * left[idx][0] + left[idx + 1][0]),
                       0.25 * (left[idx - 1][1] + 2 * left[idx][1] + left[idx + 1][1])))


def approximate_bezier(control_points):
    n = len(control_points)
    if n < 2:
        return list(control_points)
    output: list = []
    count = n
    to_flatten = [list(control_points)]
    while to_flatten:
        parent = to_flatten.pop()
        if _bezier_flat_enough(parent):
            _bezier_approximate(parent, output, count)
            continue
        left = [(0.0, 0.0)] * count
        right = [(0.0, 0.0)] * count
        _bezier_subdivide(parent, left, right, count)
        for i in range(count):
            parent[i] = left[i]
        to_flatten.append(right)
        to_flatten.append(parent)
    output.append(control_points[-1])
    return output


def _circular_arc_properties(cps):
    a, b, c = cps[0], cps[1], cps[2]
    if abs((b[1] - a[1]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[1] - a[1])) < 1e-3:
        return None  # degenerate (collinear)
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    a_sq = a[0] ** 2 + a[1] ** 2
    b_sq = b[0] ** 2 + b[1] ** 2
    c_sq = c[0] ** 2 + c[1] ** 2
    centre = ((a_sq * (b[1] - c[1]) + b_sq * (c[1] - a[1]) + c_sq * (a[1] - b[1])) / d,
              (a_sq * (c[0] - b[0]) + b_sq * (a[0] - c[0]) + c_sq * (b[0] - a[0])) / d)
    d_a = (a[0] - centre[0], a[1] - centre[1])
    d_c = (c[0] - centre[0], c[1] - centre[1])
    radius = math.hypot(*d_a)
    theta_start = math.atan2(d_a[1], d_a[0])
    theta_end = math.atan2(d_c[1], d_c[0])
    while theta_end < theta_start:
        theta_end += 2 * math.pi
    direction = 1.0
    theta_range = theta_end - theta_start
    ortho = (c[1] - a[1], -(c[0] - a[0]))
    if ortho[0] * (b[0] - a[0]) + ortho[1] * (b[1] - a[1]) < 0:
        direction = -1.0
        theta_range = 2 * math.pi - theta_range
    return (theta_start, theta_range, direction, radius, centre)


def approximate_circular_arc(cps):
    pr = _circular_arc_properties(cps)
    if pr is None:
        return approximate_bezier(cps)
    theta_start, theta_range, direction, radius, centre = pr
    if 2 * radius <= CIRCULAR_ARC_TOLERANCE:
        amount = 2
    else:
        amount = max(2, math.ceil(theta_range / (2 * math.acos(1 - CIRCULAR_ARC_TOLERANCE / radius))))
    out = []
    for i in range(amount):
        fract = i / (amount - 1)
        theta = theta_start + direction * fract * theta_range
        out.append((centre[0] + math.cos(theta) * radius, centre[1] + math.sin(theta) * radius))
    return out


# --- SliderPath ---------------------------------------------------------------

class SliderPath:
    """Legacy slider path: type char + control points -> calculated polyline,
    truncated/extended to the expected (pixel) length, with position_at()."""

    def __init__(self, curve_str: str, x0: float, hr: bool, expected_distance: float):
        parts = curve_str.split("|")
        type_char = parts[0] if parts else "B"
        points = [(float(x0), 0.0)]
        for p in parts[1:]:
            if ":" in p:
                px, py = p.split(":")
                fx = 512.0 - float(px) if hr else float(px)
                points.append((fx, float(py)))
        self.path = self._calc_path(type_char, points)
        self.cum = [0.0]
        self.distance = 0.0
        self._calc_length(expected_distance)

    def _calc_path(self, type_char, points):
        path: list = []

        def emit(seg):
            for pt in seg:
                if not path or path[-1] != pt:
                    path.append(pt)

        if len(points) < 2:
            return list(points)
        if type_char == "L":
            emit(approximate_linear(points))
        elif type_char == "P":
            emit(approximate_circular_arc(points) if len(points) == 3
                 else approximate_bezier(points))
        elif type_char == "C":
            emit(approximate_catmull(points))
        else:  # Bezier: split into sub-segments at duplicate control points
            start = 0
            for i in range(1, len(points)):
                if points[i] == points[i - 1]:
                    emit(approximate_bezier(points[start:i]))
                    start = i
            emit(approximate_bezier(points[start:]))
        return path

    def _calc_length(self, expected):
        path = self.path
        cum = [0.0]
        total = 0.0
        for i in range(len(path) - 1):
            total += _hyp(path[i + 1][0], path[i + 1][1], path[i][0], path[i][1])
            cum.append(total)

        if expected is not None and total != expected:
            # stable: if last two points equal and expected longer, no extension
            if len(path) >= 2 and path[-1] == path[-2] and expected > total:
                cum.append(total)
                self.cum, self.distance = cum, cum[-1]
                return
            cum.pop()  # the last length is always incorrect
            path_end = len(path) - 1
            if total > expected:
                while cum and cum[-1] >= expected:
                    cum.pop()
                    path.pop()
                    path_end -= 1
            if path_end <= 0:
                cum.append(0.0)
                self.cum, self.distance = cum, cum[-1]
                return
            ax, ay = path[path_end]
            bx, by = path[path_end - 1]
            dx, dy = ax - bx, ay - by
            dlen = math.hypot(dx, dy) or 1.0
            scale = (expected - cum[-1]) / dlen
            path[path_end] = (bx + dx * scale, by + dy * scale)
            cum.append(expected)

        self.cum = cum
        self.distance = cum[-1]

    def position_at(self, progress: float):
        d = max(0.0, min(1.0, progress)) * self.distance
        cum = self.cum
        path = self.path
        i = bisect.bisect_left(cum, d)
        if i <= 0:
            return path[0]
        if i >= len(path):
            return path[-1]
        d0, d1 = cum[i - 1], cum[i]
        if d1 - d0 <= 1e-9:
            return path[i - 1]
        w = (d - d0) / (d1 - d0)
        a, b = path[i - 1], path[i]
        return (a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w)

    def x_at(self, progress: float) -> float:
        return self.position_at(progress)[0]
