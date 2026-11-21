"""Tests for ``robot_arm.ik`` (analytic + numerical inverse kinematics)."""

import math
import random

import pytest

from robot_arm.arms import planar_2r, ur5_arm
from robot_arm.fk import end_effector_pose, fk_planar
from robot_arm.ik import (
    ik_2r,
    ik_3r,
    ik_numeric,
    ik_numeric_multi_start,
)
from robot_arm.vecmath import vdist, wrap_angle


def _wrapped_close(a, b, tol=1e-9):
    """True if angles ``a`` and ``b`` agree modulo 2*pi."""
    return abs(wrap_angle(a - b)) < tol


def test_ik_2r_round_trip():
    """FK of random angles -> ik_2r recovers the same EE and angles."""
    rng = random.Random(42)
    l1, l2 = 0.5, 0.4
    for _ in range(20):
        t1 = rng.uniform(-math.pi, math.pi)
        t2 = rng.uniform(-math.pi, math.pi)
        target = fk_planar((l1, l2), (t1, t2)).ee
        elbow = "down" if t2 <= 0 else "up"
        sol = ik_2r(target[0], target[1], l1, l2, elbow=elbow)
        assert sol is not None
        assert _wrapped_close(sol[0], t1)
        assert _wrapped_close(sol[1], t2)
        check = fk_planar((l1, l2), sol).ee
        assert vdist((check[0], check[1], 0.0), (target[0], target[1], 0.0)) < 1e-9


def test_ik_2r_elbow_up_vs_down():
    """Elbow up and down give distinct solutions with the same EE."""
    x, y, l1, l2 = 0.5, 0.3, 0.5, 0.4
    up = ik_2r(x, y, l1, l2, elbow="up")
    down = ik_2r(x, y, l1, l2, elbow="down")
    assert up is not None and down is not None
    assert up[1] > 0 and down[1] < 0
    assert not _wrapped_close(up[1], down[1])
    assert fk_planar((l1, l2), up).ee == pytest.approx((x, y), abs=1e-12)
    assert fk_planar((l1, l2), down).ee == pytest.approx((x, y), abs=1e-12)


def test_ik_2r_unreachable_returns_none():
    """A target beyond full stretch is unreachable."""
    assert ik_2r(5.0, 0.0, 0.5, 0.4) is None
    assert ik_2r(0.0, 5.0, 0.5, 0.4) is None


def test_ik_2r_full_stretch():
    """At full stretch (x = l1 + l2, y = 0) the elbow angle is ~0."""
    sol = ik_2r(0.9, 0.0, 0.5, 0.4)
    assert sol is not None
    assert sol[1] == pytest.approx(0.0, abs=1e-9)
    assert sol[0] == pytest.approx(0.0, abs=1e-9)


def test_ik_2r_rejects_unknown_elbow():
    """ik_2r raises ValueError for an unknown elbow value."""
    with pytest.raises(ValueError):
        ik_2r(0.5, 0.3, 0.5, 0.4, elbow="sideways")


def test_ik_3r_round_trip():
    """FK of random 3R angles -> ik_3r recovers the same EE and phi."""
    rng = random.Random(7)
    lengths = (0.4, 0.35, 0.25)
    for _ in range(20):
        thetas = tuple(rng.uniform(-math.pi, math.pi) for _ in range(3))
        result = fk_planar(lengths, thetas)
        elbow = "down" if thetas[1] <= 0 else "up"
        sol = ik_3r(result.ee[0], result.ee[1], result.phi, lengths, elbow=elbow)
        assert sol is not None
        check = fk_planar(lengths, sol)
        assert check.ee == pytest.approx(result.ee, abs=1e-9)
        assert _wrapped_close(check.phi, result.phi)


def test_ik_3r_unreachable_returns_none():
    """A 3R target beyond reach returns None."""
    assert ik_3r(5.0, 0.0, 0.0, (0.4, 0.35, 0.25)) is None


def test_ik_numeric_ur5_converges():
    """Numeric IK on the UR5 converges to a known FK target."""
    arm = ur5_arm()
    q_target = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]
    pos_t, R_t = end_effector_pose(arm, q_target)
    result = ik_numeric(arm, pos_t, target_rot=R_t, q0=arm.home)
    assert result.converged
    assert result.pos_error < 1e-6
    assert result.rot_error < 1e-6
    assert result.iterations > 0


def test_ik_numeric_position_only():
    """Position-only numeric IK (target_rot=None) reaches the position."""
    arm = ur5_arm()
    q_target = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]
    pos_t, _ = end_effector_pose(arm, q_target)
    result = ik_numeric(arm, pos_t, target_rot=None, q0=arm.home)
    assert result.converged
    assert result.pos_error < 1e-6
    assert result.rot_error == 0.0


def test_ik_numeric_planar_matches_analytic_ee():
    """Numeric IK on planar_2r reaches the analytic ik_2r target EE."""
    arm = planar_2r()
    target = fk_planar((0.5, 0.4), (0.6, -0.8)).ee
    result = ik_numeric(arm, (target[0], target[1], 0.0), target_rot=None)
    assert result.converged
    assert result.pos_error < 1e-6
    check = fk_planar((0.5, 0.4), result.q[:2])
    assert check.ee == pytest.approx(target, abs=1e-6)


def test_ik_numeric_unreachable_does_not_converge():
    """An unreachable target (5, 5, 5) does not converge."""
    arm = ur5_arm()
    _, R_home = end_effector_pose(arm, arm.home)
    result = ik_numeric(arm, (5.0, 5.0, 5.0), target_rot=R_home, q0=arm.home)
    assert not result.converged
    assert result.pos_error > 1.0


def test_ik_numeric_multi_start_converges():
    """Multi-start IK converges for a reachable UR5 target."""
    arm = ur5_arm()
    q_target = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]
    pos_t, R_t = end_effector_pose(arm, q_target)
    result = ik_numeric_multi_start(arm, pos_t, target_rot=R_t)
    assert result.converged
    assert result.pos_error < 1e-6
    assert result.rot_error < 1e-6


def test_ik_numeric_result_respects_joint_limits():
    """The numeric IK solution stays inside the joint limits."""
    arm = ur5_arm()
    q_target = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]
    pos_t, R_t = end_effector_pose(arm, q_target)
    result = ik_numeric(arm, pos_t, target_rot=R_t, q0=arm.home)
    assert arm.check_angles(result.q) == []
    # arm.home itself must not have been mutated by the solver.
    assert arm.home == [
        0.0,
        -math.pi / 2,
        math.pi / 2,
        -math.pi / 2,
        -math.pi / 2,
        0.0,
    ]
