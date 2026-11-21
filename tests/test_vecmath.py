"""Tests for ``robot_arm.vecmath`` (pure-stdlib 3D math)."""

import math

import pytest

from robot_arm.vecmath import (
    axis_angle_to_rot,
    dh_transform,
    mat3_vec,
    mat4_identity,
    mat4_inverse_rigid,
    mat4_mul,
    mat4_trans_part,
    mat_det,
    rot_to_axis_angle,
    rot_x,
    rot_z,
    solve_linear,
    vcross,
    vnormalize,
    wrap_angle,
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


def test_rot_x_pi_over_2_maps_y_to_z():
    """rot_x(pi/2) sends (0, 1, 0) to (0, 0, 1)."""
    result = mat3_vec(rot_x(math.pi / 2), (0.0, 1.0, 0.0))
    assert result == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_rot_z_pi_over_2_maps_x_to_y():
    """rot_z(pi/2) sends (1, 0, 0) to (0, 1, 0)."""
    result = mat3_vec(rot_z(math.pi / 2), (1.0, 0.0, 0.0))
    assert result == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_dh_transform_zero_angle_unit_link():
    """dh_transform(0, 0, 1, 0) translates by (1, 0, 0)."""
    T = dh_transform(0.0, 0.0, 1.0, 0.0)
    assert mat4_trans_part(T) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert T[0][0] == pytest.approx(1.0)
    assert T[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_mat4_mul_identity():
    """Multiplying by the 4x4 identity leaves a transform unchanged."""
    T = dh_transform(0.7, 0.1, 0.3, 0.4)
    I = mat4_identity()
    assert _flat(mat4_mul(I, T)) == pytest.approx(_flat(T))
    assert _flat(mat4_mul(T, I)) == pytest.approx(_flat(T))


def test_mat4_inverse_rigid_round_trip():
    """T @ inv(T) and inv(T) @ T are the identity."""
    T = dh_transform(0.7, 0.1, 0.3, 0.4)
    Tinv = mat4_inverse_rigid(T)
    assert _flat(mat4_mul(T, Tinv)) == pytest.approx(
        _flat(mat4_identity()), abs=1e-12
    )
    assert _flat(mat4_mul(Tinv, T)) == pytest.approx(
        _flat(mat4_identity()), abs=1e-12
    )


def test_solve_linear_known_system():
    """[[2, 1], [1, 3]] x = [5, 10] has solution [1, 3]."""
    assert solve_linear([[2.0, 1.0], [1.0, 3.0]], [5.0, 10.0]) == pytest.approx(
        [1.0, 3.0], abs=1e-12
    )


def test_solve_linear_raises_on_singular():
    """solve_linear raises ValueError for a singular matrix."""
    with pytest.raises(ValueError):
        solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


def test_mat_det():
    """Determinants: identity is 1, diag(2, 3) is 6, singular is 0."""
    assert mat_det(mat4_identity()) == pytest.approx(1.0)
    assert mat_det([[2.0, 0.0], [0.0, 3.0]]) == pytest.approx(6.0)
    assert mat_det([[1.0, 2.0], [2.0, 4.0]]) == pytest.approx(0.0)


def test_rot_to_axis_angle_round_trip():
    """axis_angle_to_rot followed by rot_to_axis_angle recovers the vector."""
    axis = (1.0, 2.0, 3.0)
    angle = 0.7
    R = axis_angle_to_rot(axis, angle)
    rotvec = rot_to_axis_angle(R)
    norm = math.sqrt(sum(c * c for c in axis))
    expected = tuple(c / norm * angle for c in axis)
    assert rotvec == pytest.approx(expected, abs=1e-12)


def test_wrap_angle_three_pi():
    """wrap_angle(3*pi) is -pi."""
    assert wrap_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_vcross_unit_axes():
    """vcross((1, 0, 0), (0, 1, 0)) is (0, 0, 1)."""
    assert vcross((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == (0.0, 0.0, 1.0)


def test_vnormalize_raises_on_zero_vector():
    """vnormalize((0, 0, 0)) raises ValueError."""
    with pytest.raises(ValueError):
        vnormalize((0.0, 0.0, 0.0))
