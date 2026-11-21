"""Collision clearance checks for robot-arm links.

Links are treated as capsules (a segment plus a radius) and obstacles as
spheres (3D) or circles (planar 2D).  All functions are pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SphereObstacle:
    """A spherical obstacle in 3D.

    Attributes:
        center: ``(x, y, z)`` centre in metres.
        radius: sphere radius in metres.
        name: optional label.
    """

    center: tuple
    radius: float
    name: str = ""


@dataclass
class CircleObstacle:
    """A circular obstacle in the 2D plane.

    Attributes:
        center: ``(x, y)`` centre in metres.
        radius: circle radius in metres.
        name: optional label.
    """

    center: tuple
    radius: float
    name: str = ""


def _segment_param(point, a, b, dims):
    """Return the clamped projection parameter of ``point`` on segment ``a-b``.

    Args:
        point: query point (tuple of ``dims`` floats).
        a: segment start.
        b: segment end.
        dims: number of coordinates to use.
    """
    ab = [b[i] - a[i] for i in range(dims)]
    denom = sum(v * v for v in ab)
    if denom == 0.0:
        return 0.0
    ap = [point[i] - a[i] for i in range(dims)]
    t = sum(ap[i] * ab[i] for i in range(dims)) / denom
    return max(0.0, min(1.0, t))


def segment_point_distance(point, a, b):
    """Return the 3D distance from ``point`` to segment ``a``-``b``.

    The projection parameter is clamped to ``[0, 1]``; a degenerate
    segment (``a == b``) returns the distance to ``a``.
    """
    t = _segment_param(point, a, b, 3)
    closest = tuple(a[i] + t * (b[i] - a[i]) for i in range(3))
    return math.sqrt(sum((point[i] - closest[i]) ** 2 for i in range(3)))


def segment_point_distance_2d(point, a, b):
    """Return the 2D distance from ``point`` to segment ``a``-``b``.

    Same logic as :func:`segment_point_distance` in the plane.
    """
    t = _segment_param(point, a, b, 2)
    closest = tuple(a[i] + t * (b[i] - a[i]) for i in range(2))
    return math.sqrt(sum((point[i] - closest[i]) ** 2 for i in range(2)))


def link_clearance_3d(joint_positions, link_radii, obstacles):
    """Return the minimum capsule-to-obstacle clearance in 3D.

    For each link ``i`` (segment ``positions[i]`` to ``positions[i+1]``)
    and each obstacle, the margin is
    ``segment_point_distance(center, p_i, p_{i+1}) - radius - link_radii[i]``.

    Args:
        joint_positions: list of ``(x, y, z)`` joint positions.
        link_radii: capsule radius per link.
        obstacles: list of :class:`SphereObstacle`.

    Returns:
        The minimum margin, or ``float('inf')`` when there are no
        obstacles (or no links).
    """
    if not obstacles or len(joint_positions) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(joint_positions) - 1):
        r_link = link_radii[i] if i < len(link_radii) else 0.0
        for obs in obstacles:
            d = segment_point_distance(
                obs.center, joint_positions[i], joint_positions[i + 1]
            )
            margin = d - obs.radius - r_link
            if margin < best:
                best = margin
    return best


def link_clearance_2d(joint_positions_xy, link_radii, obstacles):
    """Return the minimum capsule-to-obstacle clearance in 2D.

    Same as :func:`link_clearance_3d` in the plane, with
    :class:`CircleObstacle` obstacles.
    """
    if not obstacles or len(joint_positions_xy) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(joint_positions_xy) - 1):
        r_link = link_radii[i] if i < len(link_radii) else 0.0
        for obs in obstacles:
            d = segment_point_distance_2d(
                obs.center, joint_positions_xy[i], joint_positions_xy[i + 1]
            )
            margin = d - obs.radius - r_link
            if margin < best:
                best = margin
    return best


def check_trajectory_clearance(positions_sequence, link_radii, obstacles,
                               planar=False):
    """Check clearance of a whole trajectory of joint positions.

    Args:
        positions_sequence: list of joint-position lists, one per time
            sample.
        link_radii: capsule radius per link.
        obstacles: list of :class:`SphereObstacle` (or
            :class:`CircleObstacle` when ``planar`` is True).
        planar: use the 2D clearance when True, 3D otherwise.

    Returns:
        ``(min_margin, worst_index)``: the smallest clearance over all
        samples and the index of the sample attaining it; ``worst_index``
        is ``-1`` (and the margin ``inf``) when the sequence is empty.
    """
    if not positions_sequence:
        return (float("inf"), -1)
    clearance = link_clearance_2d if planar else link_clearance_3d
    best = float("inf")
    worst = -1
    for idx, positions in enumerate(positions_sequence):
        margin = clearance(positions, link_radii, obstacles)
        if margin < best:
            best = margin
            worst = idx
    return (best, worst)
