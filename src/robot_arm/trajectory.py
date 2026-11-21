"""Joint-space and Cartesian trajectory generation.

Pure-stdlib motion profiles for ``robot-arm-workcell-sim``: trapezoidal
velocity profiles, cubic/quintic joint interpolation, joint-space
multi-waypoint trajectories and straight Cartesian lines with orientation
interpolation.  All times are seconds, angles radians, lengths metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot_arm.vecmath import axis_angle_to_rot, rot_to_axis_angle, vnorm


@dataclass
class TrajectorySample:
    """One sample of a joint-space trajectory.

    Attributes:
        t: time of the sample (seconds).
        q: joint positions (radians).
        qd: joint velocities (radians/second).
        qdd: joint accelerations (radians/second^2).
    """

    t: float
    q: list
    qd: list
    qdd: list


@dataclass
class Trajectory:
    """A sampled joint-space trajectory.

    Attributes:
        samples: time-ordered :class:`TrajectorySample` objects.
        dt: nominal sample period (seconds).
    """

    samples: list = field(default_factory=list)
    dt: float = 0.01

    @property
    def duration(self):
        """Duration of the trajectory (time of the last sample, 0.0 if empty)."""
        if not self.samples:
            return 0.0
        return self.samples[-1].t

    @property
    def ndof(self):
        """Number of joints (length of ``q`` of the first sample, 0 if empty)."""
        if not self.samples:
            return 0
        return len(self.samples[0].q)

    def times(self):
        """Return the list of sample times."""
        return [s.t for s in self.samples]

    def positions(self):
        """Return the list of joint-position vectors, one per sample."""
        return [list(s.q) for s in self.samples]

    def max_joint_speed(self):
        """Return the largest absolute joint velocity over all samples."""
        best = 0.0
        for s in self.samples:
            for v in s.qd:
                if abs(v) > best:
                    best = abs(v)
        return best

    def max_joint_accel(self):
        """Return the largest absolute joint acceleration over all samples."""
        best = 0.0
        for s in self.samples:
            for a in s.qdd:
                if abs(a) > best:
                    best = abs(a)
        return best

    def total_joint_travel(self):
        """Return the summed absolute joint travel over all joints.

        For each joint this sums ``|dq|`` between consecutive samples, then
        sums over joints.
        """
        if len(self.samples) < 2:
            return 0.0
        total = 0.0
        for prev, cur in zip(self.samples, self.samples[1:]):
            for a, b in zip(prev.q, cur.q):
                total += abs(b - a)
        return total


def _sample_times(duration, dt):
    """Return sample times ``0, dt, 2*dt, ...`` plus the exact ``duration``.

    Args:
        duration: total duration (must be > 0).
        dt: nominal sample period (must be > 0).
    """
    times = []
    n_steps = int(math.floor(duration / dt))
    for k in range(n_steps + 1):
        t = k * dt
        if t < duration:
            times.append(t)
    if not times or abs(times[-1] - duration) > 1e-12:
        times.append(float(duration))
    return times


def trapezoidal_profile(distance, vmax, amax, dt=0.01):
    """Return a trapezoidal (or triangular) velocity profile.

    The profile moves ``distance`` metres with peak velocity ``vmax`` and
    peak acceleration ``amax``.  When ``distance < vmax**2 / amax`` the
    profile is triangular with peak velocity ``sqrt(distance * amax)``.

    Args:
        distance: distance to travel (>= 0).
        vmax: maximum velocity (> 0).
        amax: maximum acceleration (> 0).
        dt: sample period.

    Returns:
        List of ``(t, s, v, a)`` tuples; the final sample is exactly
        ``(T, distance, 0.0, 0.0)``.

    Raises:
        ValueError: if ``distance`` is negative or ``vmax``/``amax`` <= 0.
    """
    if distance < 0:
        raise ValueError(f"distance must be >= 0, got {distance}")
    if vmax <= 0:
        raise ValueError(f"vmax must be > 0, got {vmax}")
    if amax <= 0:
        raise ValueError(f"amax must be > 0, got {amax}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if distance == 0:
        return [(0.0, 0.0, 0.0, 0.0)]

    if distance < vmax * vmax / amax:
        # Triangular profile: never reaches vmax.
        v_peak = math.sqrt(distance * amax)
        t_acc = v_peak / amax
        total = 2.0 * t_acc
        t_cruise_end = t_acc
    else:
        # Trapezoidal profile: accelerate, cruise, decelerate.
        v_peak = vmax
        t_acc = vmax / amax
        total = distance / vmax + vmax / amax
        t_cruise_end = total - t_acc
    d_acc = 0.5 * amax * t_acc * t_acc

    def _eval(t):
        if t <= t_acc:
            return (0.5 * amax * t * t, amax * t, amax)
        if t <= t_cruise_end:
            return (d_acc + v_peak * (t - t_acc), v_peak, 0.0)
        tau = total - t
        return (distance - 0.5 * amax * tau * tau, amax * tau, -amax)

    samples = []
    for t in _sample_times(total, dt):
        if abs(t - total) <= 1e-12:
            samples.append((float(total), float(distance), 0.0, 0.0))
        else:
            s, v, a = _eval(t)
            samples.append((float(t), float(s), float(v), float(a)))
    return samples


def cubic_profile(q0, q1, duration, dt=0.01):
    """Return a cubic joint profile from ``q0`` to ``q1``.

    ``q(t) = q0 + dq * (3*s^2 - 2*s^3)`` with ``s = t / T``; velocity and
    acceleration are zero at both ends for velocity only (acceleration is
    ``+/-6*dq/T^2`` at the ends).

    Args:
        q0: start position.
        q1: end position.
        duration: profile duration (> 0).
        dt: sample period.

    Returns:
        List of ``(t, q, qd, qdd)`` tuples, ending exactly at ``duration``.

    Raises:
        ValueError: if ``duration`` <= 0.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    dq = q1 - q0
    samples = []
    for t in _sample_times(duration, dt):
        s = t / duration
        q = q0 + dq * (3.0 * s * s - 2.0 * s * s * s)
        qd = dq / duration * (6.0 * s - 6.0 * s * s)
        qdd = dq / (duration * duration) * (6.0 - 12.0 * s)
        samples.append((float(t), float(q), float(qd), float(qdd)))
    return samples


def quintic_profile(q0, q1, duration, dt=0.01):
    """Return a quintic joint profile from ``q0`` to ``q1``.

    ``q(t) = q0 + dq * (10*s^3 - 15*s^4 + 6*s^5)`` with ``s = t / T``;
    velocity and acceleration are both zero at both ends.

    Args:
        q0: start position.
        q1: end position.
        duration: profile duration (> 0).
        dt: sample period.

    Returns:
        List of ``(t, q, qd, qdd)`` tuples, ending exactly at ``duration``.

    Raises:
        ValueError: if ``duration`` <= 0.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    dq = q1 - q0
    samples = []
    for t in _sample_times(duration, dt):
        s = t / duration
        q = q0 + dq * (10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)
        qd = dq / duration * (30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4)
        qdd = dq / (duration * duration) * (60.0 * s - 180.0 * s**2 + 120.0 * s**3)
        samples.append((float(t), float(q), float(qd), float(qdd)))
    return samples


def joint_space_trajectory(waypoints, durations=None, kind="quintic",
                           dt=0.01, cruise_speed=1.0):
    """Build a joint-space trajectory through ``waypoints``.

    Each segment between consecutive waypoints is one cubic or quintic
    profile per joint, all sharing the segment duration.

    Args:
        waypoints: list of joint vectors (at least 2, all the same length).
        durations: optional per-segment durations (``len(waypoints) - 1``,
            all > 0).  When omitted, each segment lasts
            ``max_j |dq_j| / cruise_speed`` floored at 0.2 s.
        kind: ``"quintic"`` or ``"cubic"``.
        dt: sample period.
        cruise_speed: joint speed used to derive durations when
            ``durations`` is None (must be > 0).

    Returns:
        A :class:`Trajectory`; boundary samples are not duplicated
        (the first sample of each later segment is dropped) while time
        stays continuous.

    Raises:
        ValueError: for fewer than 2 waypoints, mismatched waypoint
            dimensions, an unknown ``kind``, bad durations or a
            non-positive ``cruise_speed``/``dt``.
    """
    if kind not in ("quintic", "cubic"):
        raise ValueError(f"unknown profile kind {kind!r} (expected 'quintic' or 'cubic')")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if len(waypoints) < 2:
        raise ValueError(f"need at least 2 waypoints, got {len(waypoints)}")
    ndof = len(waypoints[0])
    if ndof == 0:
        raise ValueError("waypoints must not be empty vectors")
    for i, wp in enumerate(waypoints):
        if len(wp) != ndof:
            raise ValueError(
                f"waypoint {i} has {len(wp)} joints, expected {ndof}"
            )
    n_seg = len(waypoints) - 1
    if durations is None:
        if cruise_speed <= 0:
            raise ValueError(f"cruise_speed must be > 0, got {cruise_speed}")
        durations = []
        for i in range(n_seg):
            delta = max(abs(b - a) for a, b in zip(waypoints[i], waypoints[i + 1]))
            durations.append(max(delta / cruise_speed, 0.2))
    else:
        if len(durations) != n_seg:
            raise ValueError(
                f"durations must have {n_seg} entries, got {len(durations)}"
            )
        for d in durations:
            if d <= 0:
                raise ValueError(f"segment durations must be > 0, got {d}")

    profile = quintic_profile if kind == "quintic" else cubic_profile
    samples = []
    t_offset = 0.0
    for seg in range(n_seg):
        seg_dur = float(durations[seg])
        per_joint = [
            profile(waypoints[seg][j], waypoints[seg + 1][j], seg_dur, dt)
            for j in range(ndof)
        ]
        n_samples = len(per_joint[0])
        for k in range(n_samples):
            if seg > 0 and k == 0:
                continue  # boundary sample already emitted by prior segment
            samples.append(TrajectorySample(
                t=t_offset + per_joint[0][k][0],
                q=[per_joint[j][k][1] for j in range(ndof)],
                qd=[per_joint[j][k][2] for j in range(ndof)],
                qdd=[per_joint[j][k][3] for j in range(ndof)],
            ))
        t_offset += seg_dur
    return Trajectory(samples=samples, dt=dt)


def cartesian_line_samples(p0, p1, R0=None, R1=None, duration=1.0, dt=0.01):
    """Sample a straight Cartesian line with orientation interpolation.

    Position interpolates linearly, ``p(s) = p0 + s * (p1 - p0)``.
    Orientation interpolates along the rotation-vector (axis-angle) path
    from ``R0`` to ``R1``; when either rotation is ``None`` every sample
    uses the identity rotation.

    Args:
        p0: start position ``(x, y, z)``.
        p1: end position ``(x, y, z)``.
        R0: start rotation (3x3) or None.
        R1: end rotation (3x3) or None.
        duration: motion duration (> 0).
        dt: sample period.

    Returns:
        List of ``(pos, R)`` tuples, ``pos`` a 3-tuple and ``R`` a 3x3
        rotation matrix, sampled at ``0, dt, ...`` plus ``duration``.

    Raises:
        ValueError: if ``duration`` <= 0.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    from robot_arm.vecmath import mat3_identity, mat3_mul, mat3_transpose

    if R0 is not None and R1 is not None:
        rel = mat3_mul(R1, mat3_transpose(R0))
        rotvec = rot_to_axis_angle(rel)
        angle = vnorm(rotvec)
    else:
        rotvec = (0.0, 0.0, 0.0)
        angle = 0.0

    samples = []
    for t in _sample_times(duration, dt):
        s = t / duration
        pos = tuple(p0[i] + s * (p1[i] - p0[i]) for i in range(3))
        if R0 is None or R1 is None or angle < 1e-12:
            R = mat3_identity() if (R0 is None or R1 is None) else [list(row) for row in R0]
        else:
            axis = tuple(rotvec[i] / angle for i in range(3))
            R = mat3_mul(axis_angle_to_rot(axis, angle * s), R0)
        samples.append((pos, R))
    return samples
