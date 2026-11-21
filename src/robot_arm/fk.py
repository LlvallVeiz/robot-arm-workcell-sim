"""Forward kinematics for ``robot-arm-workcell-sim``.

Provides planar (direct-trigonometry) and generic standard-DH forward
kinematics, plus the geometric Jacobian.  Pure stdlib; all angles in
radians, all lengths in metres.
"""

from __future__ import annotations

import math
from collections import namedtuple

from robot_arm.vecmath import (
    dh_transform,
    mat4_identity,
    mat4_mul,
    mat4_rot_part,
    mat4_trans_part,
    vcross,
    vsub,
)


class PlanarFKResult(namedtuple("PlanarFKResult", ["joints", "ee", "phi"])):
    """Result of planar forward kinematics.

    Attributes:
        joints: list of ``(x, y)`` joint positions, including the base
            ``(0, 0)`` as the first entry.
        ee: end-effector position ``(x, y)``.
        phi: end-effector orientation (radians), the sum of joint angles.
    """


def fk_planar(lengths, thetas):
    """Compute planar forward kinematics by direct trigonometry.

    Args:
        lengths: link lengths, one per joint.
        thetas: joint angles in radians, one per joint.

    Returns:
        A :class:`PlanarFKResult` with every joint position (base
        included), the end-effector position and the orientation ``phi``.

    Raises:
        ValueError: if the numbers of lengths and angles differ.
    """
    if len(lengths) != len(thetas):
        raise ValueError(
            f"lengths ({len(lengths)}) and angles ({len(thetas)}) must match"
        )
    joints = [(0.0, 0.0)]
    x, y, cum = 0.0, 0.0, 0.0
    for length, theta in zip(lengths, thetas):
        cum += theta
        x += length * math.cos(cum)
        y += length * math.sin(cum)
        joints.append((x, y))
    return PlanarFKResult(joints=joints, ee=(x, y), phi=cum)


def fk_dh(arm, thetas):
    """Compute the chain of frame transforms for a DH arm.

    Args:
        arm: an :class:`~robot_arm.arms.Arm` with a standard-DH chain.
        thetas: joint angles in radians, one per joint.

    Returns:
        A list of ``dof + 1`` 4x4 transforms: the first is the identity
        (base frame), the last is the end-effector frame.  Frame ``i``
        is ``T_0_1 @ ... @ T_{i-1}_i``.

    Raises:
        ValueError: if ``len(thetas) != arm.dof``.
    """
    if len(thetas) != arm.dof:
        raise ValueError(
            f"expected {arm.dof} joint angles, got {len(thetas)}"
        )
    transforms = [mat4_identity()]
    T = mat4_identity()
    for link, theta in zip(arm.dh, thetas):
        A = dh_transform(theta + link.theta_offset, link.d, link.a, link.alpha)
        T = mat4_mul(T, A)
        transforms.append(T)
    return transforms


def joint_positions_3d(arm, thetas):
    """Return the origin of every frame (base included) as ``(x, y, z)``.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        thetas: joint angles in radians, one per joint.

    Returns:
        A list of ``dof + 1`` position tuples, starting at ``(0, 0, 0)``.
    """
    return [mat4_trans_part(T) for T in fk_dh(arm, thetas)]


def end_effector_pose(arm, thetas):
    """Return the end-effector pose as ``(position, rotation)``.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        thetas: joint angles in radians, one per joint.

    Returns:
        ``(pos, R)`` where ``pos`` is an ``(x, y, z)`` tuple and ``R``
        is the 3x3 rotation matrix of the last DH frame.
    """
    T = fk_dh(arm, thetas)[-1]
    return mat4_trans_part(T), mat4_rot_part(T)


def geometric_jacobian(arm, thetas):
    """Return the 6-by-dof geometric Jacobian at ``thetas``.

    Rows are ``[vx, vy, vz, wx, wy, wz]`` and columns are the joints.
    For joint ``i`` (0-based), with ``z_{i-1}`` the third column of the
    rotation part of frame ``i-1`` and ``o_{i-1}`` its origin::

        Jv_i = z_{i-1} x (p_e - o_{i-1})
        Jw_i = z_{i-1}

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        thetas: joint angles in radians, one per joint.

    Returns:
        A 6-row nested list (6 x ``arm.dof``) of floats.
    """
    frames = fk_dh(arm, thetas)
    p_e = mat4_trans_part(frames[-1])
    dof = arm.dof
    J = [[0.0] * dof for _ in range(6)]
    for i in range(dof):
        T_prev = frames[i]
        R_prev = mat4_rot_part(T_prev)
        z = (R_prev[0][2], R_prev[1][2], R_prev[2][2])
        o = mat4_trans_part(T_prev)
        jv = vcross(z, vsub(p_e, o))
        for r in range(3):
            J[r][i] = jv[r]
            J[3 + r][i] = z[r]
    return J


def jacobian_planar(lengths, thetas):
    """Return the 3-by-dof planar Jacobian at ``thetas``.

    Rows are ``[vx, vy, omega_z]`` and columns are the joints.  Column
    ``i`` of the linear part is::

        (-sum_{k>=i} L_k * sin(cumtheta_k),
          sum_{k>=i} L_k * cos(cumtheta_k))

    and the angular row is all ones (every joint rotates about +z).

    Args:
        lengths: link lengths, one per joint.
        thetas: joint angles in radians, one per joint.

    Returns:
        A 3-row nested list (3 x ``len(thetas)``) of floats.

    Raises:
        ValueError: if the numbers of lengths and angles differ.
    """
    if len(lengths) != len(thetas):
        raise ValueError(
            f"lengths ({len(lengths)}) and angles ({len(thetas)}) must match"
        )
    n = len(thetas)
    cum = []
    total = 0.0
    for theta in thetas:
        total += theta
        cum.append(total)
    J = [[0.0] * n for _ in range(3)]
    for i in range(n):
        vx = -sum(
            lengths[k] * math.sin(cum[k]) for k in range(i, n)
        )
        vy = sum(
            lengths[k] * math.cos(cum[k]) for k in range(i, n)
        )
        J[0][i] = vx
        J[1][i] = vy
        J[2][i] = 1.0
    return J
