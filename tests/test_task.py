"""Tests for robot_arm.task (pick-and-place planning)."""

import math

import pytest

from robot_arm.cell import (
    default_cell_2r,
    default_cell_6dof,
    pallet_slot_positions,
)
from robot_arm.task import (
    TOOL_DOWN_ROT,
    PickPlaceRequest,
    cartesian_to_joint_trajectory,
    plan_pick_place,
    plan_to_dict,
)


def test_tool_down_rot():
    assert TOOL_DOWN_ROT == [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0]]


def _check_plan_shape(plan, cell):
    assert len(plan.segments) == 9
    labels = [s.label for s in plan.segments]
    assert labels == [
        "move_to_pick_approach", "descend_to_pick", "grasp",
        "lift_from_pick", "transit_to_place_approach", "descend_to_place",
        "release", "retreat_from_place", "return_home",
    ]
    assert plan.total_time > 0
    assert plan.total_time == pytest.approx(
        sum(s.trajectory.duration for s in plan.segments))
    assert plan.ik_failures == []
    assert len(plan.waypoints) == 9
    final_q = plan.segments[-1].trajectory.samples[-1].q
    assert final_q == pytest.approx(list(cell.home), abs=1e-9)
    by_label = {s.label: s for s in plan.segments}
    assert by_label["move_to_pick_approach"].gripper == "open"
    assert by_label["transit_to_place_approach"].gripper == "closed"
    assert by_label["grasp"].gripper == "closed"
    assert by_label["release"].gripper == "open"
    assert by_label["descend_to_pick"].cartesian is True
    assert by_label["move_to_pick_approach"].cartesian is False
    assert by_label["grasp"].cartesian is False


def test_plan_pick_place_2r():
    cell = default_cell_2r()
    req = PickPlaceRequest(pick=cell.conveyor.pick,
                           place=pallet_slot_positions(cell)[0])
    plan = plan_pick_place(cell, req)
    _check_plan_shape(plan, cell)
    d = plan_to_dict(plan)
    assert len(d["segments"]) == 9
    assert d["total_time"] == pytest.approx(plan.total_time)
    assert d["ik_failures"] == []
    assert d["n_waypoints"] == 9
    assert d["segments"][0]["label"] == "move_to_pick_approach"
    assert d["segments"][0]["n_samples"] >= 2


def test_plan_pick_place_6dof():
    cell = default_cell_6dof()
    req = PickPlaceRequest(pick=cell.conveyor.pick,
                           place=pallet_slot_positions(cell)[0])
    plan = plan_pick_place(cell, req)
    _check_plan_shape(plan, cell)
    d = plan_to_dict(plan)
    assert len(d["segments"]) == 9
    assert d["total_time"] == pytest.approx(plan.total_time)


def test_plan_pick_place_unreachable_pick():
    cell = default_cell_2r()
    req = PickPlaceRequest(pick=(5.0, 0.0, 0.0),
                           place=pallet_slot_positions(cell)[0])
    with pytest.raises(ValueError):
        plan_pick_place(cell, req)


def test_plan_pick_place_unreachable_place():
    cell = default_cell_2r()
    req = PickPlaceRequest(pick=cell.conveyor.pick, place=(5.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        plan_pick_place(cell, req)


def test_cartesian_to_joint_trajectory_2r():
    cell = default_cell_2r()
    poses = [(0.55, 0.27, None), (0.55, 0.21, None), (0.55, 0.15, None)]
    traj, failures = cartesian_to_joint_trajectory(
        cell, poses, dt=0.05, q_seed=list(cell.home))
    assert failures == []
    assert traj is not None
    assert len(traj.samples) == 3
    assert traj.dt == pytest.approx(0.05)
    assert traj.times() == pytest.approx([0.0, 0.05, 0.10])
    assert traj.ndof == 2


def test_cartesian_to_joint_trajectory_unreachable():
    cell = default_cell_2r()
    poses = [(5.0, 0.0, None)]
    traj, failures = cartesian_to_joint_trajectory(
        cell, poses, dt=0.05, q_seed=list(cell.home))
    assert traj is None
    assert len(failures) == 1
