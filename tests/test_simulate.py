"""Tests for the scenario simulation runner (simulate.py)."""

import json
import os

import pytest

from robot_arm.cell import default_cell_2r, default_cell_6dof, pallet_slot_positions
from robot_arm.simulate import run_batch, run_scenario, simulate_plan, write_trajectory_csv
from robot_arm.task import PickPlaceRequest, plan_pick_place

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _request(cell):
    slots = pallet_slot_positions(cell)
    return PickPlaceRequest(pick=cell.conveyor.pick, place=slots[0])


def test_simulate_plan_2r_success():
    cell = default_cell_2r()
    plan = plan_pick_place(cell, _request(cell))
    result = simulate_plan(cell, plan)
    assert result.success
    assert result.cycle_time == pytest.approx(plan.total_time)
    assert result.cycle_time > 0
    assert result.total_joint_travel > 0
    assert result.max_joint_speed > 0
    assert result.min_clearance > 0  # column is clear of the swept area
    assert result.n_samples == len(result.samples) > 100
    assert len(result.phases) == 9
    assert result.notes == []


def test_simulate_plan_6dof_success():
    cell = default_cell_6dof()
    plan = plan_pick_place(cell, _request(cell))
    result = simulate_plan(cell, plan)
    assert result.success
    assert result.cycle_time > 0
    assert result.min_clearance > 0
    assert len(result.phases) == 9


def test_simulate_plan_to_dict_round_trip():
    cell = default_cell_2r()
    plan = plan_pick_place(cell, _request(cell))
    result = simulate_plan(cell, plan)
    d = result.to_dict()
    assert d["success"] is True
    assert d["cycle_time_s"] == pytest.approx(result.cycle_time, abs=1e-4)
    assert len(d["phases"]) == 9
    json.dumps(d)  # must be JSON-serialisable


def test_write_trajectory_csv(tmp_path):
    cell = default_cell_2r()
    plan = plan_pick_place(cell, _request(cell))
    result = simulate_plan(cell, plan)
    path = tmp_path / "traj.csv"
    write_trajectory_csv(result, str(path))
    lines = path.read_text().splitlines()
    assert lines[0].startswith("t_s,q1_rad,q2_rad")
    assert len(lines) == result.n_samples + 1
    assert "open" in lines[1] and "closed" in "".join(lines)


def test_run_scenario_2r_writes_all_outputs(tmp_path):
    scenario = os.path.join(ROOT, "examples", "scenarios", "pick_place_2r.json")
    out = tmp_path / "out2r"
    result = run_scenario(scenario, str(out))
    assert result["success"] is True
    for name in ("trajectory.csv", "metrics.json", "report.md",
                 "report.html", "trajectory.png", "joints.png",
                 "trajectory.svg"):
        assert (out / name).exists(), f"missing {name}"
        assert (out / name).stat().st_size > 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["scenario"] == "pick-place-2r-conveyor-to-pallet"
    assert metrics["min_clearance_m"] > 0
    assert "pick-place-2r" in (out / "report.md").read_text()


def test_run_scenario_6dof_uses_projection_fallback(tmp_path):
    scenario = os.path.join(ROOT, "examples", "scenarios", "pick_place_6dof.json")
    out = tmp_path / "out6dof"
    result = run_scenario(scenario, str(out))
    assert result["success"] is True
    assert (out / "trajectory.png").exists()
    assert (out / "trajectory.png").stat().st_size > 0


def test_run_batch_runs_all_scenarios(tmp_path):
    scenarios = os.path.join(ROOT, "examples", "scenarios")
    results = run_batch(scenarios, str(tmp_path / "batch"))
    assert len(results) == 3
    assert all(r["success"] for r in results)
    names = {r["scenario"] for r in results}
    assert names == {
        "pick-place-2r-conveyor-to-pallet",
        "pick-place-3r-conveyor-to-pallet",
        "pick-place-6dof-conveyor-to-pallet",
    }
