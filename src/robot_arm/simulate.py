"""End-to-end scenario simulation for ``robot-arm-workcell-sim``.

A scenario is a JSON file that points at a workcell definition and describes
one pick-and-place task.  :func:`run_scenario` loads the cell, plans the
task, executes the plan kinematically (forward kinematics at every sample),
computes metrics, and writes the trajectory CSV, metrics JSON, plots and a
summary report into an output directory.  :func:`run_batch` runs every
``*.json`` scenario in a directory.

Scenario JSON format::

    {
      "name": "pick-place-2r-conveyor-to-pallet",
      "description": "Pick a part from the conveyor, place it on pallet slot 0.",
      "cell_file": "cells/cell_2r.json",
      "task": {
        "pick": "conveyor",
        "place_slot": 0,
        "approach_height": 0.12,
        "lift_height": 0.15,
        "dwell": 0.4,
        "transit_speed": 1.0
      },
      "dt": 0.01
    }

``"pick"`` may be ``"conveyor"`` or an ``[x, y, z]`` list; the place target
may be ``{"place_slot": N}`` or an explicit ``"place": [x, y, z]`` list.
``"cell_file"`` is resolved relative to the scenario file's directory.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

from .cell import load_workcell, pallet_slot_positions
from .collision import link_clearance_2d, link_clearance_3d
from .fk import end_effector_pose, fk_planar, joint_positions_3d
from .task import PickPlaceRequest, plan_pick_place
from . import report as report_mod
from . import visualize as viz


@dataclass
class SimSample:
    """One sampled instant of an executed plan."""

    t: float
    q: list
    qd: list
    ee: tuple
    gripper: str
    clearance: float


@dataclass
class SimResult:
    """Metrics and samples of one executed pick-and-place plan."""

    scenario: str
    cell: str
    success: bool
    cycle_time: float
    total_joint_travel: float
    max_joint_speed: float
    min_clearance: float
    n_samples: int
    phases: list
    samples: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self):
        """Return the JSON-serialisable metric summary (no raw samples)."""
        return {
            "scenario": self.scenario,
            "cell": self.cell,
            "success": self.success,
            "cycle_time_s": round(self.cycle_time, 4),
            "total_joint_travel_rad": round(self.total_joint_travel, 4),
            "max_joint_speed_rad_s": round(self.max_joint_speed, 4),
            "min_clearance_m": (
                round(self.min_clearance, 4)
                if self.min_clearance != float("inf")
                else None
            ),
            "n_samples": self.n_samples,
            "phases": self.phases,
            "notes": self.notes,
        }


def _ee_and_joints(cell, q):
    """Return ``(ee_xyz, joint_positions)`` for configuration ``q``."""
    if cell.arm.planar:
        fk = fk_planar(cell.arm.lengths, q)
        ee = (fk.ee[0], fk.ee[1], 0.0)
        joints = [(x, y, 0.0) for (x, y) in fk.joints]
        return ee, joints
    pos, _rot = end_effector_pose(cell.arm, q)
    return pos, joint_positions_3d(cell.arm, q)


def _clearance(cell, joints):
    """Return the minimum collision margin for one configuration."""
    if not cell.obstacles:
        return float("inf")
    if cell.arm.planar:
        xy = [(x, y) for (x, y, _z) in joints]
        return link_clearance_2d(xy, cell.arm.link_radii, cell.obstacles)
    return link_clearance_3d(joints, cell.arm.link_radii, cell.obstacles)


def simulate_plan(cell, plan, dt=0.01):
    """Execute ``plan`` kinematically and compute its metrics.

    Every trajectory sample is evaluated with forward kinematics and a
    capsule-vs-obstacle clearance check, so the metrics reflect the plan
    that would actually be sent to the arm.

    Args:
        cell: the :class:`~robot_arm.cell.Workcell` the plan runs in.
        plan: a :class:`~robot_arm.task.Plan` from :func:`plan_pick_place`.
        dt: sample period (seconds); kept for API symmetry.

    Returns:
        A :class:`SimResult`.
    """
    samples = []
    phases = []
    t_global = 0.0
    total_travel = 0.0
    max_speed = 0.0
    min_clearance = float("inf")

    for segment in plan.segments:
        traj = segment.trajectory
        seg_min_clearance = float("inf")
        for sample in traj.samples:
            ee, joints = _ee_and_joints(cell, sample.q)
            clearance = _clearance(cell, joints)
            seg_min_clearance = min(seg_min_clearance, clearance)
            samples.append(
                SimSample(
                    t=t_global + sample.t,
                    q=list(sample.q),
                    qd=list(sample.qd),
                    ee=ee,
                    gripper=segment.gripper,
                    clearance=clearance,
                )
            )
        seg_travel = traj.total_joint_travel()
        seg_speed = traj.max_joint_speed()
        total_travel += seg_travel
        max_speed = max(max_speed, seg_speed)
        min_clearance = min(min_clearance, seg_min_clearance)
        phases.append(
            {
                "label": segment.label,
                "duration_s": round(traj.duration, 4),
                "gripper": segment.gripper,
                "cartesian": segment.cartesian,
                "n_samples": len(traj.samples),
                "joint_travel_rad": round(seg_travel, 4),
                "max_joint_speed_rad_s": round(seg_speed, 4),
                "min_clearance_m": (
                    round(seg_min_clearance, 4)
                    if seg_min_clearance != float("inf")
                    else None
                ),
            }
        )
        t_global += traj.duration

    success = (
        not plan.ik_failures
        and len(samples) > 0
        and min_clearance > 0.0
    )
    notes = []
    if plan.ik_failures:
        notes.append(f"IK failures: {plan.ik_failures}")
    if min_clearance <= 0.0:
        notes.append(f"collision: minimum clearance {min_clearance:.4f} m <= 0")
    if min_clearance == float("inf"):
        notes.append("no obstacles defined in this cell; clearance not applicable")

    return SimResult(
        scenario="",
        cell=cell.name,
        success=success,
        cycle_time=plan.total_time,
        total_joint_travel=total_travel,
        max_joint_speed=max_speed,
        min_clearance=min_clearance,
        n_samples=len(samples),
        phases=phases,
        samples=samples,
        notes=notes,
    )


def write_trajectory_csv(result, path):
    """Write the per-sample trajectory log (CSV) for ``result``."""
    if not result.samples:
        raise ValueError("no samples to write")
    ndof = len(result.samples[0].q)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["t_s"]
            + [f"q{i + 1}_rad" for i in range(ndof)]
            + [f"qd{i + 1}_rad_s" for i in range(ndof)]
            + ["ee_x_m", "ee_y_m", "ee_z_m", "gripper", "clearance_m"]
        )
        for s in result.samples:
            writer.writerow(
                [f"{s.t:.4f}"]
                + [f"{v:.6f}" for v in s.q]
                + [f"{v:.6f}" for v in s.qd]
                + [f"{s.ee[0]:.6f}", f"{s.ee[1]:.6f}", f"{s.ee[2]:.6f}"]
                + [s.gripper, f"{s.clearance:.6f}"]
            )


def _resolve_task_target(cell, task_spec):
    """Resolve the pick/place targets of a scenario task spec."""
    pick = task_spec.get("pick", "conveyor")
    if pick == "conveyor":
        if cell.conveyor is None:
            raise ValueError("task uses conveyor pick but the cell has no conveyor")
        pick_pos = tuple(cell.conveyor.pick)
    else:
        pick_pos = tuple(float(v) for v in pick)

    if "place_slot" in task_spec:
        slots = pallet_slot_positions(cell)
        idx = int(task_spec["place_slot"])
        if not 0 <= idx < len(slots):
            raise ValueError(
                f"place_slot {idx} out of range (pallet has {len(slots)} slots)"
            )
        place_pos = slots[idx]
    elif "place" in task_spec:
        place_pos = tuple(float(v) for v in task_spec["place"])
    else:
        raise ValueError("task needs 'place_slot' or 'place'")
    return pick_pos, place_pos


def run_scenario(scenario_path, out_dir):
    """Run one scenario JSON file and write all outputs to ``out_dir``.

    Outputs: ``trajectory.csv``, ``metrics.json``, ``report.md``,
    ``report.html`` and plots (``trajectory.png``, ``joints.png`` and, for
    planar cells, ``trajectory.svg``).

    Returns:
        The result dictionary (metrics + output file names).
    """
    scenario_path = os.path.abspath(scenario_path)
    with open(scenario_path) as fh:
        scenario = json.load(fh)
    base_dir = os.path.dirname(scenario_path)
    name = scenario.get("name") or os.path.splitext(
        os.path.basename(scenario_path)
    )[0]
    dt = float(scenario.get("dt", 0.01))

    cell_file = scenario.get("cell_file")
    if not cell_file:
        raise ValueError("scenario needs a 'cell_file'")
    cell = load_workcell(os.path.join(base_dir, cell_file))

    task_spec = scenario.get("task", {})
    pick_pos, place_pos = _resolve_task_target(cell, task_spec)
    request = PickPlaceRequest(
        pick=pick_pos,
        place=place_pos,
        approach_height=float(task_spec.get("approach_height", 0.12)),
        lift_height=float(task_spec.get("lift_height", 0.15)),
        dwell=float(task_spec.get("dwell", 0.4)),
        transit_speed=float(task_spec.get("transit_speed", 1.0)),
    )
    plan = plan_pick_place(cell, request, dt=dt)
    result = simulate_plan(cell, plan, dt=dt)
    result.scenario = name

    os.makedirs(out_dir, exist_ok=True)
    result_dict = result.to_dict()
    result_dict["pick_m"] = [round(v, 4) for v in pick_pos]
    result_dict["place_m"] = [round(v, 4) for v in place_pos]
    result_dict["description"] = scenario.get("description", "")

    # Trajectory log.
    csv_path = os.path.join(out_dir, "trajectory.csv")
    write_trajectory_csv(result, csv_path)

    # Plots.
    ee_path = [s.ee for s in result.samples]
    joint_configs = []
    step = max(1, len(result.samples) // 40)
    for s in result.samples[::step]:
        _ee, joints = _ee_and_joints(cell, s.q)
        joint_configs.append(joints)
    times = [s.t for s in result.samples]
    ndof = len(result.samples[0].q)
    q_series = [[s.q[j] for s in result.samples] for j in range(ndof)]
    qd_series = [[s.qd[j] for s in result.samples] for j in range(ndof)]
    joint_names = [f"joint {i + 1}" for i in range(ndof)]

    files = {"trajectory_csv": "trajectory.csv"}
    try:
        if cell.arm.planar:
            png = os.path.join(out_dir, "trajectory.png")
            viz.plot_planar_trajectory(
                cell, ee_path, joint_configs, png,
                pick=pick_pos, place=place_pos,
            )
            files["trajectory_png"] = "trajectory.png"
            svg = os.path.join(out_dir, "trajectory.svg")
            viz.svg_planar_trajectory(
                cell, ee_path, joint_configs, svg,
                pick=pick_pos, place=place_pos,
            )
            files["trajectory_svg"] = "trajectory.svg"
        else:
            png = os.path.join(out_dir, "trajectory.png")
            viz.plot_6dof_path(cell, ee_path, joint_configs, png)
            files["trajectory_png"] = "trajectory.png"
        joints_png = os.path.join(out_dir, "joints.png")
        viz.plot_joint_trajectories(times, q_series, qd_series, joint_names, joints_png)
        files["joints_png"] = "joints.png"
    except (RuntimeError, ImportError) as exc:
        # matplotlib missing or broken (e.g. no mplot3d): plots are
        # optional, so the simulation result still stands; note it.
        result_dict.setdefault("notes", []).append(str(exc))
        result.notes.append(str(exc))

    # Metrics + reports.
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(result_dict, fh, indent=2)
    files["metrics_json"] = "metrics.json"
    result_dict["files"] = files
    report_mod.build_markdown_report(result_dict, os.path.join(out_dir, "report.md"))
    report_mod.build_html_report(result_dict, os.path.join(out_dir, "report.html"))
    files["report_md"] = "report.md"
    files["report_html"] = "report.html"
    result_dict["files"] = files
    return result_dict


def run_batch(scenarios_dir, out_dir):
    """Run every ``*.json`` scenario in ``scenarios_dir``.

    Each scenario gets its own output folder ``<out_dir>/<scenario-stem>/``.
    Returns the list of result dictionaries.
    """
    results = []
    for fname in sorted(os.listdir(scenarios_dir)):
        if not fname.endswith(".json"):
            continue
        stem = os.path.splitext(fname)[0]
        results.append(
            run_scenario(
                os.path.join(scenarios_dir, fname),
                os.path.join(out_dir, stem),
            )
        )
    return results
