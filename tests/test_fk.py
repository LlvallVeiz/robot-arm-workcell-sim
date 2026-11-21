"""Tests for ``robot_arm.fk`` (forward kinematics and Jacobians)."""

import math

import pytest

from robot_arm.arms import planar_2r, planar_3r, ur5_arm
from robot_arm.fk import (
    end_effector_pose,
    fk_dh,
    fk_planar,
    geometric_jacobian,
    jacobian_planar,
    joint_positions_3d,
)
from robot_arm.vecmath import (
    mat3_mul,
    mat3_transpose,
    mat4_rot_part,
    mat4_trans_part,
    rot_to_axis_angle,
    vsub,
)

def _flat(matrix):
    """Flatten a nested list of numbers into a flat list."""
    out = []
    for row in matrix:
        if isinstance(row, (list, tuple)):
            out.extend(_flat(row))
        else:
            out.append(row)
    return out


# UR5 expected poses, computed with an INDEPENDENT numpy DH implementation
# (scratch script /tmp/ur5_check.py, no robot_arm import) from the published
# UR5 standard-DH table.  Hard-coded here so the test does not just repeat
# the library's own code.
UR5_ZERO_POS = (-0.81725, -0.19145, -0.005491)
UR5_ZERO_ROT = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
UR5_HOME_POS = (-0.4869, -0.10915, 0.431859)
UR5_HOME_ROT = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]


def test_fk_planar_2r_hand_computed():
    """Hand-computed 2R check: L=(0.5, 0.4), angles (30deg, 45deg)."""
    result = fk_planar((0.5, 0.4), (math.radians(30), math.radians(45)))
    assert result.ee == pytest.approx(
        (0.53654031993322765, 0.6363703305156273), abs=1e-12
    )
    assert result.phi == pytest.approx(math.radians(75), abs=1e-12)
    assert result.joints[0] == (0.0, 0.0)
    assert len(result.joints) == 3


def test_fk_planar_rejects_mismatched_counts():
    """fk_planar raises ValueError when lengths/angles counts differ."""
    with pytest.raises(ValueError):
        fk_planar((0.5, 0.4), (0.1,))


@pytest.mark.parametrize(
    "lengths,thetas",
    [
        ((0.5, 0.4), (0.0, 0.0)),
        ((0.5, 0.4), (math.radians(30), math.radians(45))),
        ((0.5, 0.4), (-1.1, 2.3)),
        ((0.5, 0.4), (math.pi, -math.pi / 2)),
    ],
)
def test_fk_planar_vs_fk_dh_2r(lengths, thetas):
    """Planar trig FK and DH FK agree on joint positions (2R)."""
    arm = planar_2r(lengths)
    planar = fk_planar(lengths, thetas)
    frames = fk_dh(arm, thetas)
    assert len(frames) == arm.dof + 1
    assert _flat(frames[0]) == pytest.approx(
        _flat([[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)])
    )
    positions = joint_positions_3d(arm, thetas)
    for (x, y), (px, py, pz) in zip(planar.joints, positions):
        assert (px, py) == pytest.approx((x, y), abs=1e-12)
        assert pz == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "thetas",
    [
        (0.0, 0.0, 0.0),
        (math.radians(45), math.radians(30), math.radians(15)),
        (-0.7, 1.9, -2.4),
        (math.pi, math.pi / 3, -math.pi / 2),
    ],
)
def test_fk_planar_vs_fk_dh_3r(thetas):
    """Planar trig FK and DH FK agree on joint positions (3R)."""
    lengths = (0.4, 0.35, 0.25)
    arm = planar_3r(lengths)
    planar = fk_planar(lengths, thetas)
    positions = joint_positions_3d(arm, thetas)
    assert len(positions) == 4
    for (x, y), (px, py, pz) in zip(planar.joints, positions):
        assert (px, py) == pytest.approx((x, y), abs=1e-12)
        assert pz == pytest.approx(0.0, abs=1e-12)
    assert planar.phi == pytest.approx(sum(thetas), abs=1e-12)


def test_fk_dh_rejects_wrong_joint_count():
    """fk_dh raises ValueError when the angle count is wrong."""
    with pytest.raises(ValueError):
        fk_dh(ur5_arm(), [0.0] * 5)


def test_ur5_fk_zero_pose():
    """UR5 FK at all-zero joints matches the independent numpy DH result."""
    arm = ur5_arm()
    pos, R = end_effector_pose(arm, [0.0] * 6)
    assert pos == pytest.approx(UR5_ZERO_POS, abs=1e-9)
    assert _flat(R) == pytest.approx(_flat(UR5_ZERO_ROT), abs=1e-9)


def test_ur5_fk_home_pose():
    """UR5 FK at the home pose matches the independent numpy DH result."""
    arm = ur5_arm()
    pos, R = end_effector_pose(arm, arm.home)
    assert pos == pytest.approx(UR5_HOME_POS, abs=1e-9)
    assert _flat(R) == pytest.approx(_flat(UR5_HOME_ROT), abs=1e-9)


def test_end_effector_pose_matches_fk_dh():
    """end_effector_pose equals the last fk_dh transform's parts."""
    arm = ur5_arm()
    q = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]
    pos, R = end_effector_pose(arm, q)
    T_last = fk_dh(arm, q)[-1]
    assert pos == pytest.approx(mat4_trans_part(T_last), abs=1e-12)
    assert _flat(R) == pytest.approx(_flat(mat4_rot_part(T_last)), abs=1e-12)


def test_geometric_jacobian_finite_difference_ur5():
    """UR5 geometric Jacobian matches finite differences of the EE pose."""
    arm = ur5_arm()
    q = [0.3, -1.2, 0.9, 0.5, -0.7, 0.2]  # non-singular config
    J = geometric_jacobian(arm, q)
    assert len(J) == 6 and all(len(row) == 6 for row in J)
    eps = 1e-6
    pos0, R0 = end_effector_pose(arm, q)
    for j in range(6):
        q_pert = list(q)
        q_pert[j] += eps
        pos1, R1 = end_effector_pose(arm, q_pert)
        dp = tuple((a - b) / eps for a, b in zip(pos1, pos0))
        # Orientation derivative: rotation vector of R1 @ R0^T over eps.
        dR = mat3_mul(R1, mat3_transpose(R0))
        dw = tuple(v / eps for v in rot_to_axis_angle(dR))
        for r in range(3):
            assert J[r][j] == pytest.approx(dp[r], abs=1e-4)
            assert J[3 + r][j] == pytest.approx(dw[r], abs=1e-4)


def test_jacobian_planar_finite_difference():
    """Planar Jacobian matches finite differences of FK (2R and 3R)."""
    for lengths, thetas in [
        ((0.5, 0.4), (0.6, -0.8)),
        ((0.4, 0.35, 0.25), (0.5, -1.0, 0.7)),
    ]:
        J = jacobian_planar(lengths, thetas)
        assert len(J) == 3 and all(len(row) == len(thetas) for row in J)
        eps = 1e-6
        base = fk_planar(lengths, thetas)
        for j in range(len(thetas)):
            pert = list(thetas)
            pert[j] += eps
            moved = fk_planar(lengths, pert)
            dvx = (moved.ee[0] - base.ee[0]) / eps
            dvy = (moved.ee[1] - base.ee[1]) / eps
            dphi = (moved.phi - base.phi) / eps
            assert J[0][j] == pytest.approx(dvx, abs=1e-4)
            assert J[1][j] == pytest.approx(dvy, abs=1e-4)
            assert J[2][j] == pytest.approx(dphi, abs=1e-4)


def test_joint_positions_3d_includes_base():
    """joint_positions_3d starts at the base origin (0, 0, 0)."""
    arm = planar_2r()
    positions = joint_positions_3d(arm, (0.3, 0.4))
    assert positions[0] == (0.0, 0.0, 0.0)
    assert len(positions) == 3
