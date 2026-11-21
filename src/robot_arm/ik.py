"""Inverse kinematics for ``robot-arm-workcell-sim``.

Analytic two/three-link planar solvers plus a damped least-squares
numerical solver for arbitrary DH arms.  Pure stdlib; all angles in
radians, all lengths in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from robot_arm.fk import end_effector_pose, geometric_jacobian
from robot_arm.vecmath import (
    mat3_mul,
    mat3_transpose,
    mat_mul,
    mat_transpose,
    rot_to_axis_angle,
    solve_linear,
    vnorm,
    vsub,
    wrap_angle,
)


def ik_2r(x, y, l1, l2, elbow="down"):
    """Solve the planar 2R inverse kinematics analytically.

    Args:
        x: target end-effector x position (metres).
        y: target end-effector y position (metres).
        l1: first link length (metres).
        l2: second link length (metres).
        elbow: ``"down"`` (``t2 <= 0``) or ``"up"`` (``t2 >= 0``).

    Returns:
        ``(t1, t2)`` in radians, or ``None`` if the target is
        unreachable.

    Raises:
        ValueError: for an unknown ``elbow`` value.
    """
    if elbow not in ("up", "down"):
        raise ValueError(f"unknown elbow {elbow!r} (expected 'up' or 'down')")
    cos_t2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if abs(cos_t2) > 1.0 + 1e-12:
        return None
    cos_t2 = max(-1.0, min(1.0, cos_t2))
    t2 = math.acos(cos_t2)
    if elbow == "down":
        t2 = -t2
    t1 = math.atan2(y, x) - math.atan2(
        l2 * math.sin(t2), l1 + l2 * math.cos(t2)
    )
    return (t1, t2)


def ik_3r(x, y, phi, lengths, elbow="down"):
    """Solve the planar 3R inverse kinematics analytically.

    The wrist position is backed off from the target along the desired
    tool orientation ``phi``, the first two joints are solved with
    :func:`ik_2r`, and the third joint makes up the remaining angle.

    Args:
        x: target end-effector x position (metres).
        y: target end-effector y position (metres).
        phi: desired end-effector orientation (radians).
        lengths: ``(L1, L2, L3)`` link lengths (metres).
        elbow: ``"down"`` or ``"up"`` (passed to :func:`ik_2r`).

    Returns:
        ``(t1, t2, t3)`` in radians, or ``None`` if unreachable.
    """
    l1, l2, l3 = lengths
    wx = x - l3 * math.cos(phi)
    wy = y - l3 * math.sin(phi)
    sol = ik_2r(wx, wy, l1, l2, elbow=elbow)
    if sol is None:
        return None
    t1, t2 = sol
    t3 = wrap_angle(phi - t1 - t2)
    return (t1, t2, t3)


@dataclass
class IKResult:
    """Result of a numerical inverse-kinematics solve.

    Attributes:
        q: joint angles (radians) of the best solution found.
        converged: True if the position (and rotation, when requested)
            tolerances were met.
        iterations: number of iterations performed.
        pos_error: final position error (metres).
        rot_error: final rotation error (radians, rotation-vector norm),
            or 0.0 when no target rotation was given.
        message: human-readable outcome description.
    """

    q: list
    converged: bool
    iterations: int
    pos_error: float
    rot_error: float
    message: str


def _clamp_to_limits(q, arm):
    """Return ``q`` clamped element-wise into ``arm.joint_limits``."""
    return [
        max(lo, min(hi, value))
        for value, (lo, hi) in zip(q, arm.joint_limits)
    ]


def ik_numeric(
    arm,
    target_pos,
    target_rot=None,
    q0=None,
    max_iter=200,
    tol_pos=1e-9,
    tol_rot=1e-9,
    damping=0.05,
    step_limit=0.5,
):
    """Solve IK by damped least squares (Levenberg-Marquardt style).

    Each iteration computes the FK pose, the error vector (position
    error plus, when ``target_rot`` is given, the rotation vector of
    ``R_target @ R_current^T``), and the step ``dq`` from
    ``(J^T J + damping^2 I) dq = J^T e``.  Steps are scaled down if
    ``max|dq|`` exceeds ``step_limit``, and joints are clamped into
    ``arm.joint_limits`` after every step.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        target_pos: desired end-effector ``(x, y, z)`` (metres).
        target_rot: optional desired 3x3 rotation matrix.
        q0: starting joint angles; defaults to a copy of ``arm.home``
            (``arm.home`` itself is never mutated).
        max_iter: maximum number of iterations.
        tol_pos: position tolerance (metres) for convergence.
        tol_rot: rotation tolerance (radians) for convergence.
        damping: Levenberg damping factor.
        step_limit: maximum absolute per-joint step per iteration.

    Returns:
        An :class:`IKResult`.
    """
    if q0 is None:
        q = list(arm.home)
    else:
        q = list(q0)
    q = _clamp_to_limits(q, arm)
    target = tuple(float(v) for v in target_pos)
    dof = arm.dof
    pos_error = float("inf")
    rot_error = 0.0
    iterations = 0
    converged = False
    message = "did not converge within max_iter"

    for it in range(max_iter):
        pos, R = end_effector_pose(arm, q)
        pos_err = vsub(target, pos)
        pos_error = vnorm(pos_err)
        if target_rot is None:
            rot_error = 0.0
            rot_err = (0.0, 0.0, 0.0)
        else:
            rot_err = rot_to_axis_angle(
                mat3_mul(target_rot, mat3_transpose(R))
            )
            rot_error = vnorm(rot_err)
        if pos_error < tol_pos and (
            target_rot is None or rot_error < tol_rot
        ):
            converged = True
            iterations = it
            message = "converged"
            break
        iterations = it + 1
        J_full = geometric_jacobian(arm, q)
        if target_rot is None:
            J = J_full[:3]
            e = list(pos_err)
        else:
            J = J_full
            e = list(pos_err) + list(rot_err)
        # Normal equations: (J^T J + damping^2 I) dq = J^T e.
        Jt = mat_transpose(J)
        JtJ = mat_mul(Jt, J)
        for i in range(dof):
            JtJ[i][i] += damping * damping
        Jt_e = mat_mul(Jt, [[v] for v in e])
        rhs = [row[0] for row in Jt_e]
        try:
            dq = solve_linear(JtJ, rhs)
        except ValueError:
            message = "singular normal equations"
            break
        max_step = max((abs(v) for v in dq), default=0.0)
        if max_step > step_limit:
            scale = step_limit / max_step
            dq = [v * scale for v in dq]
        q = [qi + dqi for qi, dqi in zip(q, dq)]
        q = _clamp_to_limits(q, arm)
    else:
        # Loop ran to max_iter without the convergence break: recompute
        # the final errors at the last q.
        pos, R = end_effector_pose(arm, q)
        pos_error = vnorm(vsub(target, pos))
        if target_rot is None:
            rot_error = 0.0
        else:
            rot_error = vnorm(
                rot_to_axis_angle(mat3_mul(target_rot, mat3_transpose(R)))
            )
        iterations = max_iter

    return IKResult(
        q=q,
        converged=converged,
        iterations=iterations,
        pos_error=pos_error,
        rot_error=rot_error,
        message=message,
    )


def ik_numeric_multi_start(arm, target_pos, target_rot=None, seeds=None, **kwargs):
    """Run :func:`ik_numeric` from several seeds and keep the best.

    The best result prefers a converged solve, then the smallest
    ``pos_error + rot_error``.

    Args:
        arm: an :class:`~robot_arm.arms.Arm`.
        target_pos: desired end-effector ``(x, y, z)`` (metres).
        target_rot: optional desired 3x3 rotation matrix.
        seeds: starting configurations; defaults to ``arm.home``,
            all-zeros, home with joint 2 +90 deg, and home with
            joint 2 -90 deg.  Every seed is clipped into the joint
            limits before use.
        **kwargs: passed through to :func:`ik_numeric`.

    Returns:
        The best :class:`IKResult` across all seeds.
    """
    if seeds is None:
        home = list(arm.home)
        seeds = [list(home), [0.0] * arm.dof]
        if arm.dof >= 2:
            up = list(home)
            up[1] = home[1] + math.pi / 2
            seeds.append(up)
            down = list(home)
            down[1] = home[1] - math.pi / 2
            seeds.append(down)
    best = None
    for seed in seeds:
        seed = _clamp_to_limits(list(seed), arm)
        result = ik_numeric(
            arm, target_pos, target_rot=target_rot, q0=seed, **kwargs
        )
        if best is None:
            best = result
            continue
        key = (not result.converged, result.pos_error + result.rot_error)
        best_key = (
            not best.converged,
            best.pos_error + best.rot_error,
        )
        if key < best_key:
            best = result
    return best
