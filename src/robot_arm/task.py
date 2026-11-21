"""Pick-and-place task planning for a workcell.

Plans a 9-phase pick-and-place cycle: move to the pick approach, descend,
grasp, lift, transit to the place approach, descend, release, retreat and
return home.  Joint-space phases use quintic profiles; Cartesian phases
are converted to joint space sample-by-sample through inverse kinematics.

The FK/IK helpers are imported lazily inside the functions that need them
so this module can be imported before ``robot_arm.fk`` / ``robot_arm.ik``
are present; the functions themselves follow the published specs of those
modules exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robot_arm.trajectory import (
    Trajectory,
    TrajectorySample,
    cartesian_line_samples,
    joint_space_trajectory,
)

#: Tool orientation with the tool z-axis pointing down (-z of the base).
TOOL_DOWN_ROT = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]


@dataclass
class PickPlaceRequest:
    """A pick-and-place request.

    Attributes:
        pick: ``(x, y, z)`` pick position in metres.
        place: ``(x, y, z)`` place position in metres.
        approach_height: height of the approach pose above pick/place.
        lift_height: height the part is lifted after grasping.
        dwell: grasp/release dwell time in seconds.
        transit_speed: joint-space cruise speed (rad/s) for transit phases.
    """

    pick: tuple
    place: tuple
    approach_height: float = 0.12
    lift_height: float = 0.15
    dwell: float = 0.4
    transit_speed: float = 1.0


@dataclass
class PlanSegment:
    """One phase of a pick-and-place plan.

    Attributes:
        label: phase name, e.g. ``"move_to_pick_approach"``.
        trajectory: joint-space :class:`Trajectory` for the phase.
        gripper: ``"open"``, ``"closed"`` or ``"hold"``.
        cartesian: True when the phase follows a Cartesian path.
    """

    label: str
    trajectory: Trajectory
    gripper: str
    cartesian: bool


@dataclass
class Plan:
    """A complete pick-and-place plan.

    Attributes:
        segments: ordered :class:`PlanSegment` list.
        total_time: sum of the segment durations.
        ik_failures: IK failure messages encountered (empty on success).
        waypoints: one dict per phase boundary with keys ``"label"``,
            ``"joints"`` (joint config at the segment end) and ``"ee"``
            (end-effector position at the segment end).
    """

    segments: list
    total_time: float
    ik_failures: list = field(default_factory=list)
    waypoints: list = field(default_factory=list)


def _ee_position(cell, q):
    """Return the end-effector position for joint config ``q``."""
    if cell.arm.planar:
        from robot_arm.fk import fk_planar

        result = fk_planar(cell.arm.lengths, list(q))
        ee = result.ee
        return (float(ee[0]), float(ee[1]), 0.0)
    from robot_arm.fk import end_effector_pose

    pos, _rot = end_effector_pose(cell.arm, list(q))
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def _ik_for_pose(cell, pose, label, ik_failures, q_seed=None):
    """Solve IK for one pose; on failure record it and raise ValueError.

    Args:
        cell: the workcell.
        pose: ``(x, y, phi_or_None)`` for planar arms, ``(pos, R)`` for
            6-DOF arms.
        label: phase label used in the error message.
        ik_failures: list to append a failure message to before raising.
        q_seed: seed joint config for numeric IK (6-DOF only).

    Returns:
        The joint solution as a list of floats.
    """
    if cell.arm.planar:
        if cell.arm.dof == 2:
            from robot_arm.ik import ik_2r

            x, y, _phi = pose
            sol = ik_2r(x, y, cell.arm.lengths[0], cell.arm.lengths[1],
                        elbow="down")
        else:
            from robot_arm.ik import ik_3r

            x, y, phi = pose
            if phi is None:
                phi = -math.pi / 2
            sol = ik_3r(x, y, phi, cell.arm.lengths, elbow="down")
        if sol is None:
            msg = f"no IK solution for {label} at {pose}"
            ik_failures.append(msg)
            raise ValueError(f"IK failed for {label}: {msg}")
        return [float(v) for v in sol]

    from robot_arm.ik import ik_numeric, ik_numeric_multi_start

    pos, rot = pose
    result = None
    if q_seed is not None:
        # Prefer the solution continuous with the seed configuration:
        # it keeps joint-space moves short and avoids wild elbow swings
        # that a far-away multi-start branch can produce.
        candidate = ik_numeric(cell.arm, tuple(pos), target_rot=rot,
                               q0=list(q_seed))
        if candidate.converged:
            result = candidate
    if result is None:
        result = ik_numeric_multi_start(cell.arm, tuple(pos),
                                        target_rot=rot, seeds=None)
    if result is None or not result.converged:
        detail = "" if result is None else f" ({result.message})"
        msg = f"numeric IK did not converge for {label} at {pos}{detail}"
        ik_failures.append(msg)
        raise ValueError(f"IK failed for {label}: {msg}")
    return [float(v) for v in result.q]


def _approach_pose(cell, point, approach_height):
    """Return the approach pose above ``point``.

    For 6-DOF arms the pose is ``(pos, TOOL_DOWN_ROT)`` with ``z`` raised
    by ``approach_height``; for planar arms it is ``(x, y + h, phi)``
    with ``phi = -pi/2`` (3R) or ``None`` (2R).
    """
    if cell.arm.planar:
        phi = -math.pi / 2 if cell.arm.dof == 3 else None
        return (point[0], point[1] + approach_height, phi)
    return ((point[0], point[1], point[2] + approach_height),
            [list(row) for row in TOOL_DOWN_ROT])


def _lifted_pose(cell, point, lift_height):
    """Return the pose ``lift_height`` above ``point`` (same form as approach)."""
    return _approach_pose(cell, point, lift_height)


def _cartesian_poses(cell, p_from, p_to, duration, dt):
    """Return the pose list for a straight Cartesian move.

    ``p_from``/``p_to`` are ``(x, y, z)`` points.  For 6-DOF arms the
    poses are ``(pos, TOOL_DOWN_ROT)``; for planar arms they are
    ``(x, y, phi)`` with ``phi = -pi/2`` (3R) or ``None`` (2R).
    """
    samples = cartesian_line_samples(
        p_from, p_to,
        [list(row) for row in TOOL_DOWN_ROT],
        [list(row) for row in TOOL_DOWN_ROT],
        duration=duration, dt=dt,
    )
    poses = []
    phi = -math.pi / 2 if (cell.arm.planar and cell.arm.dof == 3) else None
    for pos, _rot in samples:
        if cell.arm.planar:
            poses.append((pos[0], pos[1], phi))
        else:
            poses.append((pos, [list(row) for row in TOOL_DOWN_ROT]))
    return poses


def cartesian_to_joint_trajectory(cell, poses, dt, q_seed):
    """Convert a list of Cartesian poses to a joint-space trajectory.

    Planar arms use the analytic IK (``ik_2r``/``ik_3r``; for 3R with
    ``phi`` None the wrist angle defaults to ``-pi/2``).  6-DOF arms use
    numeric IK seeded with the previous solution (``q_seed`` first).
    Joint velocities come from central finite differences of the joint
    positions (forward/backward at the edges) and accelerations from
    finite differences of the velocities.

    Args:
        cell: the workcell holding the arm.
        poses: list of ``(pos, R)`` for 6-DOF arms or ``(x, y, phi_or_None)``
            for planar arms.
        dt: sample period (seconds).
        q_seed: seed joint config for the first numeric-IK solve.

    Returns:
        ``(trajectory, failures)`` where ``trajectory`` is a
        :class:`Trajectory` with ``dt`` and times ``0..(n-1)*dt`` (``None``
        when any IK solve fails) and ``failures`` is a list of messages.
    """
    failures = []
    if not poses:
        return Trajectory(samples=[], dt=dt), failures

    solutions = []
    q_prev = list(q_seed) if q_seed is not None else None
    for idx, pose in enumerate(poses):
        if cell.arm.planar:
            if cell.arm.dof == 2:
                from robot_arm.ik import ik_2r

                x, y, _phi = pose
                sol = ik_2r(x, y, cell.arm.lengths[0],
                            cell.arm.lengths[1], elbow="down")
            else:
                from robot_arm.ik import ik_3r

                x, y, phi = pose
                if phi is None:
                    phi = -math.pi / 2
                sol = ik_3r(x, y, phi, cell.arm.lengths, elbow="down")
            if sol is None:
                failures.append(
                    f"IK failed for cartesian sample {idx} at {pose}: "
                    "no analytic solution (target out of reach)"
                )
                return None, failures
            solutions.append([float(v) for v in sol])
        else:
            from robot_arm.ik import ik_numeric

            pos, rot = pose
            result = ik_numeric(cell.arm, tuple(pos), target_rot=rot,
                                q0=q_prev)
            if result is None or not result.converged:
                detail = "" if result is None else f" ({result.message})"
                failures.append(
                    f"IK failed for cartesian sample {idx} at {pos}: "
                    f"numeric IK did not converge{detail}"
                )
                return None, failures
            solutions.append([float(v) for v in result.q])
            q_prev = list(result.q)

    n = len(solutions)
    ndof = len(solutions[0])
    qd = []
    for i in range(n):
        if n == 1:
            qd.append([0.0] * ndof)
        elif i == 0:
            qd.append([(solutions[1][j] - solutions[0][j]) / dt
                       for j in range(ndof)])
        elif i == n - 1:
            qd.append([(solutions[-1][j] - solutions[-2][j]) / dt
                       for j in range(ndof)])
        else:
            qd.append([(solutions[i + 1][j] - solutions[i - 1][j]) / (2.0 * dt)
                       for j in range(ndof)])
    qdd = []
    for i in range(n):
        if n == 1:
            qdd.append([0.0] * ndof)
        elif i == 0:
            qdd.append([(qd[1][j] - qd[0][j]) / dt for j in range(ndof)])
        elif i == n - 1:
            qdd.append([(qd[-1][j] - qd[-2][j]) / dt for j in range(ndof)])
        else:
            qdd.append([(qd[i + 1][j] - qd[i - 1][j]) / (2.0 * dt)
                        for j in range(ndof)])

    samples = [
        TrajectorySample(t=i * dt, q=list(solutions[i]),
                         qd=list(qd[i]), qdd=list(qdd[i]))
        for i in range(n)
    ]
    return Trajectory(samples=samples, dt=dt), failures


def _dwell_trajectory(q, dwell, dt):
    """Return a constant-``q`` trajectory lasting ``dwell`` seconds.

    Sampled at ``dt`` with at least 2 samples; velocities and
    accelerations are zero.
    """
    times = [0.0]
    t = dt
    while t < dwell - 1e-12:
        times.append(t)
        t += dt
    if len(times) < 2 or abs(times[-1] - dwell) > 1e-12:
        times.append(float(dwell))
    return Trajectory(
        samples=[TrajectorySample(t=t, q=list(q),
                                  qd=[0.0] * len(q), qdd=[0.0] * len(q))
                 for t in times],
        dt=dt,
    )


def _cartesian_segment(cell, label, p_from, p_to, q_seed, dt,
                       transit_speed, ik_failures):
    """Build one Cartesian plan segment between two points.

    Returns ``(PlanSegment, q_end)``; raises ValueError (after recording
    in ``ik_failures``) when the IK for any sample fails.
    """
    dist = math.sqrt(sum((b - a) ** 2 for a, b in zip(p_from, p_to)))
    duration = max(dist / transit_speed, 0.2) if transit_speed > 0 else 0.2
    poses = _cartesian_poses(cell, p_from, p_to, duration, dt)
    traj, failures = cartesian_to_joint_trajectory(cell, poses, dt, q_seed)
    if traj is None:
        ik_failures.extend(failures)
        raise ValueError(
            f"IK failed for {label}: " + "; ".join(failures)
        )
    return traj, list(traj.samples[-1].q)


def plan_pick_place(cell, request, dt=0.01):
    """Plan a 9-phase pick-and-place cycle for ``cell`` and ``request``.

    Phases (label, gripper): ``move_to_pick_approach`` (open, joint),
    ``descend_to_pick`` (open, Cartesian), ``grasp`` (closed, dwell),
    ``lift_from_pick`` (closed, Cartesian),
    ``transit_to_place_approach`` (closed, joint),
    ``descend_to_place`` (closed, Cartesian), ``release`` (open, dwell),
    ``retreat_from_place`` (open, Cartesian), ``return_home`` (open, joint).

    Args:
        cell: the workcell.
        request: the :class:`PickPlaceRequest`.
        dt: sample period (seconds).

    Returns:
        A :class:`Plan` whose final joints equal ``cell.home``.

    Raises:
        ValueError: when IK fails for any target pose; the message is
            recorded in the (unreturned) failure list before raising.
    """
    ik_failures = []
    home = [float(v) for v in cell.home]
    pick = tuple(float(v) for v in request.pick)
    place = tuple(float(v) for v in request.place)

    q_approach_pick = _ik_for_pose(
        cell, _approach_pose(cell, pick, request.approach_height),
        "move_to_pick_approach", ik_failures, q_seed=list(home))
    q_approach_place = _ik_for_pose(
        cell, _approach_pose(cell, place, request.approach_height),
        "transit_to_place_approach", ik_failures,
        q_seed=q_approach_pick)

    segments = []
    waypoints = []
    current_q = list(home)

    def _finish(label, traj, gripper, cartesian, q_end):
        segments.append(PlanSegment(label=label, trajectory=traj,
                                     gripper=gripper, cartesian=cartesian))
        waypoints.append({
            "label": label,
            "joints": list(q_end),
            "ee": _ee_position(cell, q_end),
        })
        return list(q_end)

    # 1. Move (joint space) from home to the pick approach pose.
    traj = joint_space_trajectory(
        [current_q, q_approach_pick], kind="quintic", dt=dt,
        cruise_speed=request.transit_speed)
    current_q = _finish("move_to_pick_approach", traj, "open", False,
                        list(traj.samples[-1].q))

    # 2. Descend (Cartesian) from approach to pick.
    approach_pick_pt = (pick[0], pick[1],
                        pick[2] + request.approach_height) \
        if not cell.arm.planar else \
        (pick[0], pick[1] + request.approach_height, pick[2])
    traj, q_pick = _cartesian_segment(
        cell, "descend_to_pick", approach_pick_pt, pick, current_q, dt,
        request.transit_speed, ik_failures)
    current_q = _finish("descend_to_pick", traj, "open", True, q_pick)

    # 3. Grasp dwell at the pick joints.
    traj = _dwell_trajectory(current_q, request.dwell, dt)
    current_q = _finish("grasp", traj, "closed", False,
                        list(traj.samples[-1].q))

    # 4. Lift (Cartesian) from pick to pick + lift height.
    lifted_pick_pt = (pick[0], pick[1], pick[2] + request.lift_height) \
        if not cell.arm.planar else \
        (pick[0], pick[1] + request.lift_height, pick[2])
    traj, q_lifted = _cartesian_segment(
        cell, "lift_from_pick", pick, lifted_pick_pt, current_q, dt,
        request.transit_speed, ik_failures)
    current_q = _finish("lift_from_pick", traj, "closed", True, q_lifted)

    # 5. Transit (joint space) to the place approach pose.
    traj = joint_space_trajectory(
        [current_q, q_approach_place], kind="quintic", dt=dt,
        cruise_speed=request.transit_speed)
    current_q = _finish("transit_to_place_approach", traj, "closed", False,
                        list(traj.samples[-1].q))

    # 6. Descend (Cartesian) from approach to place.
    approach_place_pt = (place[0], place[1],
                         place[2] + request.approach_height) \
        if not cell.arm.planar else \
        (place[0], place[1] + request.approach_height, place[2])
    traj, q_place = _cartesian_segment(
        cell, "descend_to_place", approach_place_pt, place, current_q, dt,
        request.transit_speed, ik_failures)
    current_q = _finish("descend_to_place", traj, "closed", True, q_place)

    # 7. Release dwell at the place joints.
    traj = _dwell_trajectory(current_q, request.dwell, dt)
    current_q = _finish("release", traj, "open", False,
                        list(traj.samples[-1].q))

    # 8. Retreat (Cartesian) from place to the place approach pose.
    traj, q_retreat = _cartesian_segment(
        cell, "retreat_from_place", place, approach_place_pt, current_q, dt,
        request.transit_speed, ik_failures)
    current_q = _finish("retreat_from_place", traj, "open", True, q_retreat)

    # 9. Return home (joint space); ends exactly at cell.home.
    traj = joint_space_trajectory(
        [current_q, home], kind="quintic", dt=dt,
        cruise_speed=request.transit_speed)
    current_q = _finish("return_home", traj, "open", False,
                        list(traj.samples[-1].q))

    total_time = sum(seg.trajectory.duration for seg in segments)
    return Plan(segments=segments, total_time=total_time,
                ik_failures=ik_failures, waypoints=waypoints)


def plan_to_dict(plan):
    """Serialise a :class:`Plan` to a plain dict.

    Returns ``{"total_time", "segments", "ik_failures", "n_waypoints"}``
    where each segment entry has ``"label"``, ``"duration"``,
    ``"gripper"``, ``"cartesian"`` and ``"n_samples"``.
    """
    return {
        "total_time": plan.total_time,
        "segments": [
            {
                "label": seg.label,
                "duration": seg.trajectory.duration,
                "gripper": seg.gripper,
                "cartesian": seg.cartesian,
                "n_samples": len(seg.trajectory.samples),
            }
            for seg in plan.segments
        ],
        "ik_failures": list(plan.ik_failures),
        "n_waypoints": len(plan.waypoints),
    }
