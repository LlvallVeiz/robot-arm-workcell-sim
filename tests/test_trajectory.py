"""Tests for robot_arm.trajectory."""

import math

import pytest

from robot_arm.trajectory import (
    Trajectory,
    TrajectorySample,
    cartesian_line_samples,
    cubic_profile,
    joint_space_trajectory,
    quintic_profile,
    trapezoidal_profile,
)
from robot_arm.vecmath import mat3_identity, rot_z


def _sample_at(profile, t, dt=0.01):
    for sample in profile:
        if abs(sample[0] - t) < 1e-9:
            return sample
    raise AssertionError(f"no sample at t={t}")


# --- trapezoidal_profile -------------------------------------------------

def test_trapezoidal_reaches_vmax():
    prof = trapezoidal_profile(1.0, 0.5, 1.0)
    # Correct trapezoid: T = d/vmax + vmax/amax = 2.0 + 0.5 = 2.5 s.
    assert prof[-1][0] == pytest.approx(2.5)
    assert prof[-1][1] == pytest.approx(1.0)
    assert max(s[2] for s in prof) == pytest.approx(0.5)
    assert prof[-1][2] == pytest.approx(0.0)
    assert prof[-1][3] == pytest.approx(0.0)
    assert prof[0][0] == pytest.approx(0.0)
    assert prof[0][1] == pytest.approx(0.0)
    assert prof[0][2] == pytest.approx(0.0)
    assert all(prof[i][0] < prof[i + 1][0] for i in range(len(prof) - 1))


def test_trapezoidal_triangular():
    prof = trapezoidal_profile(0.1, 0.5, 1.0)
    peak_v = max(s[2] for s in prof)
    assert peak_v < 0.5
    # True peak sqrt(0.1) = 0.3162 falls between dt samples, so the
    # sampled peak is slightly below it.
    assert peak_v == pytest.approx(math.sqrt(0.1), abs=0.01)
    assert prof[-1][1] == pytest.approx(0.1)
    assert prof[-1][0] == pytest.approx(2.0 * math.sqrt(0.1))


def test_trapezoidal_zero_distance():
    assert trapezoidal_profile(0.0, 0.5, 1.0) == [(0.0, 0.0, 0.0, 0.0)]


def test_trapezoidal_bad_args():
    with pytest.raises(ValueError):
        trapezoidal_profile(1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        trapezoidal_profile(1.0, -0.5, 1.0)
    with pytest.raises(ValueError):
        trapezoidal_profile(1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        trapezoidal_profile(-1.0, 0.5, 1.0)


# --- cubic / quintic profiles --------------------------------------------

def test_cubic_profile():
    prof = cubic_profile(0.0, 1.0, 2.0)
    assert _sample_at(prof, 1.0)[1] == pytest.approx(0.5)
    assert prof[0][2] == pytest.approx(0.0)
    assert prof[-1][2] == pytest.approx(0.0)
    assert prof[0][3] == pytest.approx(1.5)
    assert prof[-1][3] == pytest.approx(-1.5)
    assert prof[-1][0] == pytest.approx(2.0)
    assert prof[-1][1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cubic_profile(0.0, 1.0, 0.0)


def test_quintic_profile():
    prof = quintic_profile(0.0, 1.0, 2.0)
    assert _sample_at(prof, 1.0)[1] == pytest.approx(0.5)
    assert prof[0][2] == pytest.approx(0.0)
    assert prof[-1][2] == pytest.approx(0.0)
    assert prof[0][3] == pytest.approx(0.0)
    assert prof[-1][3] == pytest.approx(0.0)
    assert max(s[2] for s in prof) == pytest.approx(0.9375)
    assert prof[-1][0] == pytest.approx(2.0)
    assert prof[-1][1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        quintic_profile(0.0, 1.0, -1.0)


# --- Trajectory container -------------------------------------------------

def test_trajectory_properties_empty():
    traj = Trajectory(samples=[], dt=0.01)
    assert traj.duration == 0.0
    assert traj.ndof == 0
    assert traj.times() == []
    assert traj.positions() == []
    assert traj.max_joint_speed() == 0.0
    assert traj.max_joint_accel() == 0.0
    assert traj.total_joint_travel() == 0.0


# --- joint_space_trajectory ----------------------------------------------

def test_joint_space_trajectory_two_waypoints():
    traj = joint_space_trajectory([[0.0, 0.0], [1.0, -1.0]])
    assert traj.positions()[0] == pytest.approx([0.0, 0.0])
    assert traj.positions()[-1] == pytest.approx([1.0, -1.0])
    assert traj.duration > 0
    assert math.isfinite(traj.max_joint_speed())
    assert traj.total_joint_travel() == pytest.approx(2.0)
    assert traj.ndof == 2


def test_joint_space_trajectory_durations_respected():
    traj = joint_space_trajectory(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], durations=[1.0, 2.0])
    assert traj.duration == pytest.approx(3.0)
    # No duplicated boundary sample: the middle waypoint appears once
    # as a sample at t == 1.0.
    at_one = [s for s in traj.samples if abs(s.t - 1.0) < 1e-9]
    assert len(at_one) == 1
    assert at_one[0].q == pytest.approx([1.0, 0.0])


def test_joint_space_trajectory_cubic_kind():
    traj = joint_space_trajectory([[0.0], [1.0]], kind="cubic",
                                  durations=[2.0])
    assert traj.positions()[-1] == pytest.approx([1.0])
    assert traj.samples[0].qd == pytest.approx([0.0])


def test_joint_space_trajectory_errors():
    with pytest.raises(ValueError):
        joint_space_trajectory([[0.0, 0.0]])
    with pytest.raises(ValueError):
        joint_space_trajectory([[0.0, 0.0], [1.0]])
    with pytest.raises(ValueError):
        joint_space_trajectory([[0.0], [1.0]], kind="linear")
    with pytest.raises(ValueError):
        joint_space_trajectory([[0.0], [1.0]], durations=[1.0, 2.0])
    with pytest.raises(ValueError):
        joint_space_trajectory([[0.0], [1.0]], durations=[0.0])


# --- cartesian_line_samples ----------------------------------------------

def test_cartesian_line_midpoint_position():
    samples = cartesian_line_samples((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                     duration=1.0, dt=0.01)
    assert samples[0][0] == pytest.approx((0.0, 0.0, 0.0))
    assert samples[-1][0] == pytest.approx((1.0, 0.0, 0.0))
    assert samples[50][0] == pytest.approx((0.5, 0.0, 0.0))
    assert len(samples) == 101


def test_cartesian_line_orientation_interpolation():
    samples = cartesian_line_samples(
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        R0=mat3_identity(), R1=rot_z(math.pi / 2),
        duration=1.0, dt=0.01)
    expected = rot_z(math.pi / 4)
    mid = samples[50][1]
    for i in range(3):
        for j in range(3):
            assert mid[i][j] == pytest.approx(expected[i][j], abs=1e-9)
    first = samples[0][1]
    for i in range(3):
        for j in range(3):
            assert first[i][j] == pytest.approx(mat3_identity()[i][j],
                                                abs=1e-9)
    last = samples[-1][1]
    for i in range(3):
        for j in range(3):
            assert last[i][j] == pytest.approx(rot_z(math.pi / 2)[i][j],
                                               abs=1e-9)


def test_cartesian_line_no_rotation_is_identity():
    samples = cartesian_line_samples((0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                     duration=0.5, dt=0.1)
    for _pos, R in samples:
        for i in range(3):
            for j in range(3):
                assert R[i][j] == pytest.approx(mat3_identity()[i][j])
    with pytest.raises(ValueError):
        cartesian_line_samples((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                               duration=0.0)
