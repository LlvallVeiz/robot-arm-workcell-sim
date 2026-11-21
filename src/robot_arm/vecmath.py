"""Pure-stdlib 3D mathematics for ``robot-arm-workcell-sim``.

Vectors are plain ``tuple[float, float, float]``; rotations are 3x3 nested
lists; homogeneous transforms are 4x4 nested lists (row-major).  The
kinematics core needs no third-party dependency, so the package runs on any
Python 3.9+ interpreter.

Conventions
-----------
* Right-handed frames, angles in radians, lengths in metres.
* A homogeneous transform ``T`` maps a point ``p`` in the child frame to the
  parent frame as ``p_parent = R @ p + t`` (``w = 1``).
* Standard Denavit-Hartenberg transform for one joint::

      A = RotZ(theta) @ TransZ(d) @ TransX(a) @ RotX(alpha)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Vectors (3-tuples)
# ---------------------------------------------------------------------------


def vadd(a, b):
    """Return ``a + b``."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    """Return ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a, s):
    """Return ``s * a``."""
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    """Return the dot product ``a . b``."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    """Return the cross product ``a x b``."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def vnorm(a):
    """Return the Euclidean norm ``|a|``."""
    return math.sqrt(vdot(a, a))


def vdist(a, b):
    """Return the Euclidean distance between ``a`` and ``b``."""
    return vnorm(vsub(a, b))


def vnormalize(a):
    """Return the unit vector along ``a``.

    Raises:
        ValueError: if ``a`` is the zero vector.
    """
    n = vnorm(a)
    if n == 0.0:
        raise ValueError("cannot normalise the zero vector")
    return vscale(a, 1.0 / n)


# ---------------------------------------------------------------------------
# 3x3 rotation matrices
# ---------------------------------------------------------------------------


def mat3_identity():
    """Return the 3x3 identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]


def mat3_mul(A, B):
    """Return the 3x3 product ``A @ B``."""
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat3_vec(A, v):
    """Return ``A @ v`` for a 3-vector ``v``."""
    return tuple(sum(A[i][k] * v[k] for k in range(3)) for i in range(3))


def mat3_transpose(A):
    """Return the transpose of a 3x3 matrix."""
    return [[A[j][i] for j in range(3)] for i in range(3)]


def rot_x(angle):
    """Rotation about the x-axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def rot_y(angle):
    """Rotation about the y-axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rot_z(angle):
    """Rotation about the z-axis by ``angle`` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def axis_angle_to_rot(axis, angle):
    """Return the 3x3 rotation matrix for ``axis``/``angle`` (Rodrigues)."""
    x, y, z = vnormalize(axis)
    c, s = math.cos(angle), math.sin(angle)
    cc = 1.0 - c
    return [
        [c + x * x * cc, x * y * cc - z * s, x * z * cc + y * s],
        [y * x * cc + z * s, c + y * y * cc, y * z * cc - x * s],
        [z * x * cc - y * s, z * y * cc + x * s, c + z * z * cc],
    ]


def rot_to_axis_angle(R):
    """Return the rotation vector (unit axis * angle) of rotation ``R``.

    The vector is zero for the identity rotation.  The branch near an angle
    of pi recovers the axis from the diagonal of ``R``.
    """
    trace = R[0][0] + R[1][1] + R[2][2]
    cos_angle = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0)
    if math.pi - angle < 1e-4:
        # Near 180 degrees the skew-symmetric part vanishes; recover the
        # axis from the diagonal instead.
        x = math.sqrt(max(0.0, (R[0][0] + 1.0) / 2.0))
        y = math.sqrt(max(0.0, (R[1][1] + 1.0) / 2.0))
        z = math.sqrt(max(0.0, (R[2][2] + 1.0) / 2.0))
        if R[0][1] < 0.0:
            y = -y
        if R[0][2] < 0.0:
            z = -z
        axis = vnormalize((x, y, z))
        return vscale(axis, angle)
    denom = 2.0 * math.sin(angle)
    axis = (
        (R[2][1] - R[1][2]) / denom,
        (R[0][2] - R[2][0]) / denom,
        (R[1][0] - R[0][1]) / denom,
    )
    return vscale(axis, angle)


def rotvec(R):
    """Alias for :func:`rot_to_axis_angle`."""
    return rot_to_axis_angle(R)


# ---------------------------------------------------------------------------
# 4x4 homogeneous transforms
# ---------------------------------------------------------------------------


def mat4_identity():
    """Return the 4x4 identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_mul(A, B):
    """Return the 4x4 product ``A @ B``."""
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat4_vec(T, v):
    """Apply homogeneous transform ``T`` to a 3-vector point ``v``."""
    return (
        T[0][0] * v[0] + T[0][1] * v[1] + T[0][2] * v[2] + T[0][3],
        T[1][0] * v[0] + T[1][1] * v[1] + T[1][2] * v[2] + T[1][3],
        T[2][0] * v[0] + T[2][1] * v[1] + T[2][2] * v[2] + T[2][3],
    )


def mat4_trans_part(T):
    """Return the translation part of ``T`` as a 3-tuple."""
    return (T[0][3], T[1][3], T[2][3])


def mat4_rot_part(T):
    """Return the 3x3 rotation part of ``T``."""
    return [row[:3] for row in T[:3]]


def trans_xyz(x, y, z):
    """Return a pure-translation 4x4 transform."""
    T = mat4_identity()
    T[0][3] = x
    T[1][3] = y
    T[2][3] = z
    return T


def dh_transform(theta, d, a, alpha):
    """Return the standard DH transform for one revolute joint.

    ``A = RotZ(theta) @ TransZ(d) @ TransX(a) @ RotX(alpha)`` (Craig 2005,
    Lynch & Park 2017).  ``theta`` is the joint variable; ``d``, ``a`` and
    ``alpha`` are the constant DH parameters of the link.
    """
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return [
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_inverse_rigid(T):
    """Return the inverse of a rigid-body transform (``R`` orthonormal)."""
    R = mat4_rot_part(T)
    Rt = mat3_transpose(R)
    t = mat4_trans_part(T)
    tinv = vscale(mat3_vec(Rt, t), -1.0)
    out = mat4_identity()
    for i in range(3):
        for j in range(3):
            out[i][j] = Rt[i][j]
        out[i][3] = tinv[i]
    return out


# ---------------------------------------------------------------------------
# Generic small-matrix linear algebra
# ---------------------------------------------------------------------------


def mat_mul(A, B):
    """Return the product of two generic matrices (nested lists)."""
    n_rows, n_mid, n_cols = len(A), len(B), len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(n_mid)) for j in range(n_cols)] for i in range(n_rows)]


def mat_transpose(A):
    """Return the transpose of a generic matrix (nested lists)."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


def solve_linear(A, b):
    """Solve ``A @ x = b`` by Gaussian elimination with partial pivoting.

    Args:
        A: square matrix as nested lists (not modified).
        b: right-hand side vector.

    Returns:
        The solution vector ``x`` as a list of floats.

    Raises:
        ValueError: if ``A`` is (numerically) singular.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        raise ValueError("solve_linear needs a square matrix matching b")
    M = [list(A[i]) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14:
            raise ValueError("singular matrix")
        M[col], M[pivot] = M[pivot], M[col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def mat_det(A):
    """Return the determinant of a square matrix (LU with pivoting)."""
    n = len(A)
    if n == 0 or any(len(row) != n for row in A):
        raise ValueError("mat_det needs a square matrix")
    M = [list(map(float, row)) for row in A]
    det = 1.0
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-14:
            return 0.0
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]
            det = -det
        det *= M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n):
                M[r][c] -= factor * M[col][c]
    return det


def wrap_angle(angle):
    """Wrap an angle in radians to the interval ``(-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
