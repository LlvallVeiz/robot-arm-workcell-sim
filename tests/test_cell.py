"""Tests for robot_arm.cell."""

import json
import math

import pytest

from robot_arm.arms import planar_2r, planar_3r, ur5_arm
from robot_arm.cell import (
    Conveyor,
    Workcell,
    default_cell_2r,
    default_cell_6dof,
    load_workcell,
    max_reach,
    pallet_slot_positions,
    validate_workcell,
)
from robot_arm.collision import SphereObstacle


FULL_JSON = {
    "name": "test-cell",
    "arm": {"kind": "2r", "lengths": [0.5, 0.4]},
    "obstacles": [{"center": [0.45, 0.55, 0.0], "radius": 0.08,
                   "name": "column"}],
    "conveyor": {"pick": [0.55, 0.15, 0.0], "approach_height": 0.12},
    "pallet": {"origin": [0.15, 0.55, 0.0], "rows": 2, "cols": 3,
               "dx": 0.12, "dy": 0.12},
    "limits": {"max_joint_speed": 2.0, "max_joint_accel": 5.0},
    "home_deg": [45, 45],
}


def test_load_workcell_full(tmp_path):
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(FULL_JSON))
    cell = load_workcell(str(path))
    assert cell.name == "test-cell"
    assert cell.arm.name == "planar-2r"
    assert cell.arm.dof == 2
    assert len(cell.obstacles) == 1
    assert cell.obstacles[0].radius == pytest.approx(0.08)
    assert cell.obstacles[0].center == pytest.approx((0.45, 0.55, 0.0))
    assert cell.conveyor.pick == pytest.approx((0.55, 0.15, 0.0))
    assert cell.conveyor.approach_height == pytest.approx(0.12)
    assert cell.pallet.rows == 2
    assert cell.pallet.cols == 3
    assert cell.max_joint_speed == pytest.approx(2.0)
    assert cell.max_joint_accel == pytest.approx(5.0)
    assert cell.home == pytest.approx([math.pi / 4, math.pi / 4])


def test_load_workcell_defaults(tmp_path):
    path = tmp_path / "cell.json"
    path.write_text(json.dumps({"arm": {"kind": "6dof"}}))
    cell = load_workcell(str(path))
    assert cell.arm.name == "ur5"
    assert cell.max_joint_speed == pytest.approx(1.5)
    assert cell.max_joint_accel == pytest.approx(4.0)
    assert cell.home == pytest.approx(cell.arm.home)
    assert cell.obstacles == []
    assert cell.conveyor is None
    assert cell.pallet is None


def test_load_workcell_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        load_workcell(str(bad))

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({"arm": {"kind": "7dof"}}))
    with pytest.raises(ValueError):
        load_workcell(str(unknown))

    missing_arm = tmp_path / "noarm.json"
    missing_arm.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError):
        load_workcell(str(missing_arm))

    missing_kind = tmp_path / "nokind.json"
    missing_kind.write_text(json.dumps({"arm": {}}))
    with pytest.raises(ValueError):
        load_workcell(str(missing_kind))

    wrong_type = tmp_path / "wrongtype.json"
    wrong_type.write_text(json.dumps({"arm": {"kind": "2r"},
                                      "pallet": {"origin": "oops"}}))
    with pytest.raises(ValueError):
        load_workcell(str(wrong_type))


def test_pallet_slot_positions():
    cell = default_cell_2r()
    slots = pallet_slot_positions(cell)
    assert len(slots) == 6
    assert slots[0] == pytest.approx((0.15, 0.55, 0.0))
    assert slots[-1] == pytest.approx((0.15 + 2 * 0.12, 0.55 + 0.12, 0.0))
    # Row-major: second slot is the next column, fourth slot the next row.
    assert slots[1] == pytest.approx((0.27, 0.55, 0.0))
    assert slots[3] == pytest.approx((0.15, 0.67, 0.0))
    assert pallet_slot_positions(Workcell(name="n", arm=planar_2r())) == []


def test_validate_default_cells():
    assert validate_workcell(default_cell_2r()) == []
    assert validate_workcell(default_cell_6dof()) == []


def test_validate_out_of_reach_conveyor():
    cell = default_cell_2r()
    cell.conveyor = Conveyor(pick=(5.0, 0.0, 0.0), approach_height=0.12)
    errors = validate_workcell(cell)
    assert any("conveyor pick out of reach" in e for e in errors)


def test_validate_bad_pallet_and_obstacle():
    cell = default_cell_6dof()
    cell.pallet.rows = 0
    cell.pallet.dx = -1.0
    cell.obstacles = [SphereObstacle(center=(0.0, 0.0, 0.0), radius=0.0)]
    errors = validate_workcell(cell)
    assert len(errors) >= 3


def test_validate_home_length():
    cell = default_cell_2r()
    cell.home = [0.0]
    assert validate_workcell(cell) != []


def test_max_reach():
    assert max_reach(default_cell_2r()) == pytest.approx(0.9)
    assert max_reach(default_cell_6dof()) == pytest.approx(1.19251)
    assert max_reach(Workcell(name="x", arm=planar_3r((0.4, 0.35, 0.25)))) \
        == pytest.approx(1.0)


def test_default_cells_content():
    cell2 = default_cell_2r()
    assert cell2.name == "demo-cell-2r"
    assert cell2.arm.planar
    assert cell2.conveyor.pick == pytest.approx((0.55, 0.15, 0.0))
    cell6 = default_cell_6dof()
    assert cell6.name == "demo-cell-6dof"
    assert cell6.arm.name == "ur5"
    assert cell6.conveyor.pick == pytest.approx((0.45, -0.25, 0.05))
    assert cell6.home == pytest.approx(ur5_arm().home)
