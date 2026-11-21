"""Tests for ``robot_arm.workspace`` (reach, manipulability, sampling)."""

import math

import pytest

from robot_arm.arms import planar_2r, planar_3r, ur5_arm
from robot_arm.workspace import (
    check_reachable_2r,
    is_singular,
    manipulability_6dof,
    manipulability_planar,
    planar_reach_2r,
    sample_workspace_6dof,
    sample_workspace_planar,
)

# Wrist singularity for the UR5: q5 = 0 aligns joints 4 and 6.
SINGULAR_Q = [0.0, -math.pi / 2, 0.0, 0.0, 0.0, 0.0]


def test_planar_reach_2r():
    """planar_reach_2r(0.5, 0.4) is (0.1, 0.9)."""
    assert planar_reach_2r(0.5, 0.4) == pytest.approx((0.1, 0.9))


def test_check_reachable_2r():
    """Points inside the annulus are reachable, outside ones are not."""
    assert check_reachable_2r(0.5, 0.0, 0.5, 0.4)
    assert check_reachable_2r(0.9, 0.0, 0.5, 0.4)  # outer boundary
    assert check_reachable_2r(0.1, 0.0, 0.5, 0.4)  # inner boundary
    assert not check_reachable_2r(1.0, 0.0, 0.5, 0.4)
    assert not check_reachable_2r(0.05, 0.0, 0.5, 0.4)  # inside inner hole
    assert not check_reachable_2r(0.0, 0.0, 0.5, 0.4)


def test_manipulability_planar_2r():
    """2R manipulability is 0 at full stretch, maximal at t2 = 90 deg."""
    lengths = (0.5, 0.4)
    assert manipulability_planar(lengths, (0.5, 0.0)) == pytest.approx(0.0)
    assert manipulability_planar(lengths, (0.5, math.pi)) == pytest.approx(0.0)
    assert manipulability_planar(lengths, (0.5, math.pi / 2)) == pytest.approx(
        0.5 * 0.4
    )


def test_manipulability_planar_3r_nonzero():
    """3R manipulability is positive away from singularities."""
    assert manipulability_planar((0.4, 0.35, 0.25), (0.5, 0.6, -0.3)) > 0.0


def test_manipulability_planar_rejects_other_dof():
    """manipulability_planar raises ValueError for non-2R/3R arms."""
    with pytest.raises(ValueError):
        manipulability_planar((0.5,), (0.1,))


def test_manipulability_6dof_home_positive():
    """UR5 manipulability is positive at the (non-singular) home pose."""
    arm = ur5_arm()
    assert manipulability_6dof(arm, arm.home) > 0.0


def test_manipulability_6dof_singular_config():
    """UR5 manipulability is ~0 at the wrist singularity (q5 = 0)."""
    arm = ur5_arm()
    assert manipulability_6dof(arm, SINGULAR_Q) < 1e-6


def test_is_singular_ur5():
    """is_singular is True at the wrist singularity, False at home."""
    arm = ur5_arm()
    assert is_singular(arm, SINGULAR_Q)
    assert not is_singular(arm, arm.home)


def test_is_singular_planar():
    """is_singular follows the planar manipulability for planar arms."""
    arm = planar_2r()
    assert is_singular(arm, (0.5, 0.0))
    assert not is_singular(arm, (0.5, math.pi / 2))


def test_sample_workspace_6dof():
    """6-DOF sampling returns n positions, a valid bbox, inside max reach."""
    arm = ur5_arm()
    positions, bbox = sample_workspace_6dof(arm, n_samples=2000, seed=12345)
    assert len(positions) == 2000
    assert all(len(p) == 3 for p in positions)
    for i in range(3):
        assert bbox["min"][i] < bbox["max"][i]
        assert bbox["min"][i] == pytest.approx(
            min(p[i] for p in positions), abs=1e-12
        )
        assert bbox["max"][i] == pytest.approx(
            max(p[i] for p in positions), abs=1e-12
        )
    # Max reach is the sum of |a| + |d| over the UR5 DH chain = 1.192509 m.
    for p in positions:
        assert math.sqrt(sum(c * c for c in p)) <= 1.1925 + 1e-6


def test_sample_workspace_6dof_reproducible():
    """Same seed gives the same sampled workspace."""
    arm = ur5_arm()
    pos_a, _ = sample_workspace_6dof(arm, n_samples=50, seed=99)
    pos_b, _ = sample_workspace_6dof(arm, n_samples=50, seed=99)
    assert pos_a == pos_b


def test_sample_workspace_planar():
    """Planar grid sampling is non-empty and stays within total length."""
    lengths = (0.5, 0.4)
    positions = sample_workspace_planar(lengths, step_deg=5.0)
    assert len(positions) > 0
    max_r = max(math.hypot(x, y) for x, y in positions)
    assert max_r <= sum(lengths) + 1e-9
    # The grid includes the fully stretched pose, so max radius is ~sum.
    assert max_r == pytest.approx(sum(lengths), abs=1e-9)
