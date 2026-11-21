"""Tests for plotting (matplotlib) and the SVG fallback (visualize.py)."""

import math

import pytest

from robot_arm.cell import default_cell_2r, default_cell_6dof
from robot_arm.fk import end_effector_pose, fk_planar, joint_positions_3d
from robot_arm.task import PickPlaceRequest, plan_pick_place
from robot_arm.visualize import (
    plot_6dof_path,
    plot_joint_trajectories,
    plot_planar_trajectory,
    svg_planar_trajectory,
)

matplotlib = pytest.importorskip("matplotlib")


def _planar_samples(cell):
    plan = plan_pick_place(
        cell,
        PickPlaceRequest(pick=cell.conveyor.pick,
                         place=(0.15, 0.55, 0.0)),
    )
    ee_path, configs = [], []
    step = max(1, sum(len(s.trajectory.samples) for s in plan.segments) // 40)
    count = 0
    for seg in plan.segments:
        for sample in seg.trajectory.samples:
            fk = fk_planar(cell.arm.lengths, sample.q)
            ee_path.append((fk.ee[0], fk.ee[1], 0.0))
            if count % step == 0:
                configs.append([(x, y, 0.0) for (x, y) in fk.joints])
            count += 1
    return ee_path, configs


def test_plot_planar_trajectory_writes_png(tmp_path):
    cell = default_cell_2r()
    ee_path, configs = _planar_samples(cell)
    out = tmp_path / "traj.png"
    assert plot_planar_trajectory(cell, ee_path, configs, str(out)) == str(out)
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_svg_planar_trajectory_writes_svg(tmp_path):
    cell = default_cell_2r()
    ee_path, configs = _planar_samples(cell)
    out = tmp_path / "traj.svg"
    assert svg_planar_trajectory(cell, ee_path, configs, str(out)) == str(out)
    text = out.read_text()
    assert text.startswith("<svg")
    assert "end-effector" not in text  # path is geometry, not labels
    assert "<circle" in text and "<path" in text


def test_plot_joint_trajectories_writes_png(tmp_path):
    times = [i * 0.01 for i in range(101)]
    q_series = [[math.sin(t) for t in times], [math.cos(t) for t in times]]
    qd_series = [[math.cos(t) for t in times], [-math.sin(t) for t in times]]
    out = tmp_path / "joints.png"
    plot_joint_trajectories(times, q_series, qd_series,
                            ["joint 1", "joint 2"], str(out))
    assert out.stat().st_size > 1000


def test_plot_6dof_path_writes_png(tmp_path):
    cell = default_cell_6dof()
    plan = plan_pick_place(
        cell,
        PickPlaceRequest(pick=cell.conveyor.pick,
                         place=(0.30, 0.30, 0.05)),
    )
    ee_path, configs = [], []
    for seg in plan.segments:
        for sample in seg.trajectory.samples[::10]:
            pos, _ = end_effector_pose(cell.arm, sample.q)
            ee_path.append(pos)
            configs.append(joint_positions_3d(cell.arm, sample.q))
    out = tmp_path / "path.png"
    plot_6dof_path(cell, ee_path, configs, str(out))
    assert out.stat().st_size > 1000
