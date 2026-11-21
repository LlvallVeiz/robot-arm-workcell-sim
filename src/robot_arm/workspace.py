"""Workspace analysis for ``robot-arm-workcell-sim``.

Reachability, manipulability / singularity checks, and workspace
sampling (uniform joint sampling for 6-DOF arms, a joint grid for
planar arms).  Pure stdlib; all angles in radians, lengths in metres.
"""

from __future__ import annotations

import itertools
import math
import random

from robot_arm.fk import (
    end_effector_pose,
    fk_planar,
    geometric_jacobian,
    jacobian_planar,
)
from robot_arm.vecmath import mat_det, mat_mul, mat_transpose, vnorm


def planar_reach_2r(l1, l2):
    """Return the reachable annulus ``(r_min, r_max)`` of a planar 2R arm.

    Args:
        l1: first link length (metres).
        l2: second link length (metres).

    Returns:
        ``(abs(l1 - l2), l1 + l2)``.
    """
    return (abs(l1 - l2), l1 + l2)


def check_reachable_2r(x, y, l1, l2):
    """Return True if ``(x, y)`` lies in the 2R reachable annulus.

    A small ``1e-9`` tolerance is allowed on both boundaries.

    Args:
        x: target x position (metres).
        y: target y position (metres).
        l1: first link length (metres).
        l2: second link length (metres).
    """
    r_min, r_max = planar_reach_2r(l1, l2)
    r = math.hypot(x, y)
    return r_min - 1e-9 <= r <= r_max + 1e-9


def manipulability_planar(lengths, thetas):
    """Return the planar manipulability at ``thetas``.

    For 2R arms this is ``|l1 * l2 * sin(t2)|``; for 3R arms it is
    ``|det(J)|`` of the 3x3 planar Jacobian.

    Args:
        lengths: link lengths, one per joint.
        thetas: joint angles in radians, one per joint.

    Returns:
        A non-negative float (0 at a singularity).

    Raises:
        ValueError: for arms with a joint count other than 2 or 3.
    """
    if len(lengths) != len(thetas):
        raise ValueError(
            f"lengths ({len(lengths)}) and angles ({len(thetas)}) must match"
        )
    if len(thetas) == 2:
        return abs(lengths[0] * lengths[1] * math.sin(thetas[1]))
    if len(thetas) == 3:
        return abs(mat_det(jacobian_planar(lengths, thetas)))
    raise ValueError(
        f"manipulability_planar supports 2R/3R arms, got {len(thetas)} joints"
    )


def manipulability_6dof(arm, thetas):
    """Return the 6-DOF manipulability ``sqrt(det(J @ J^T))``.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        thetas: joint angles in radians, one per joint.

    Returns:
        A non-negative float (0 at a singularity).
    """
    J = geometric_jacobian(arm, thetas)
    JJt = mat_mul(J, mat_transpose(J))
    return math.sqrt(max(0.0, mat_det(JJt)))


def is_singular(arm, thetas, threshold=1e-6):
    """Return True if the arm is (near-)singular at ``thetas``.

    Planar arms use :func:`manipulability_planar`; all other arms use
    :func:`manipulability_6dof`.  The manipulability is compared
    against ``threshold``.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        thetas: joint angles in radians, one per joint.
        threshold: manipulability below this counts as singular.
    """
    if arm.planar:
        return manipulability_planar(arm.lengths, thetas) < threshold
    return manipulability_6dof(arm, thetas) < threshold


def sample_workspace_6dof(arm, n_samples=2000, seed=12345):
    """Sample the 6-DOF reachable workspace by uniform joint sampling.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        n_samples: number of joint configurations to sample.
        seed: seed for the ``random.Random`` sampler (reproducible).

    Returns:
        ``(positions, bbox)`` where ``positions`` is a list of
        ``(x, y, z)`` end-effector positions and ``bbox`` is
        ``{"min": (x, y, z), "max": (x, y, z)}``.
    """
    rng = random.Random(seed)
    positions = []
    for _ in range(n_samples):
        q = [
            rng.uniform(lo, hi) for (lo, hi) in arm.joint_limits
        ]
        pos, _ = end_effector_pose(arm, q)
        positions.append(tuple(pos))
    if positions:
        mins = tuple(min(p[i] for p in positions) for i in range(3))
        maxs = tuple(max(p[i] for p in positions) for i in range(3))
    else:
        mins = (0.0, 0.0, 0.0)
        maxs = (0.0, 0.0, 0.0)
    return positions, {"min": mins, "max": maxs}


def sample_workspace_planar(lengths, step_deg=5.0):
    """Sample planar end-effector positions over a joint-angle grid.

    Each joint angle runs from -180 to +180 degrees in steps of
    ``step_deg``; every combination is evaluated with :func:`fk_planar`.

    Args:
        lengths: link lengths, one per joint.
        step_deg: grid step in degrees.

    Returns:
        A list of ``(x, y)`` end-effector positions (one per grid
        point).
    """
    n_steps = int(round(360.0 / step_deg))
    angles = [
        math.radians(-180.0 + i * step_deg) for i in range(n_steps + 1)
    ]
    positions = []
    for thetas in itertools.product(angles, repeat=len(lengths)):
        positions.append(fk_planar(lengths, thetas).ee)
    return positions
