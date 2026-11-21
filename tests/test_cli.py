"""Tests for the command-line interface (cli.py)."""

import json
import os

import pytest

from robot_arm.cli import main

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_cli_fk_2r(capsys):
    assert main(["fk", "--arm", "2r", "--angles", "30", "45"]) == 0
    out = capsys.readouterr().out
    assert "end effector: (0.5365, 0.6364) m" in out
    assert "75.00 deg" in out


def test_cli_fk_wrong_angle_count(capsys):
    assert main(["fk", "--arm", "2r", "--angles", "30"]) == 2


def test_cli_ik_2r(capsys):
    # Full-precision FK target of (30 deg, 45 deg) on the 0.5/0.4 m arm,
    # so the FK re-check error is exactly zero.
    assert main(["ik", "--arm", "2r", "--target",
                 "0.53654031993322765", "0.6363703305156273"]) == 0
    out = capsys.readouterr().out
    assert "position error=" in out
    assert "0.00e+00" in out


def test_cli_ik_unreachable_returns_1(capsys):
    assert main(["ik", "--arm", "2r", "--target", "5", "0"]) == 1
    assert "unreachable" in capsys.readouterr().out


def test_cli_ik_6dof(capsys):
    assert main(["ik", "--arm", "6dof", "--target", "0.4", "-0.2", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "position error:" in out


def test_cli_plan_2r(capsys):
    cell = os.path.join(ROOT, "examples", "scenarios", "cells", "cell_2r.json")
    assert main(["plan", "--cell", cell, "--pick", "conveyor",
                 "--place-slot", "0"]) == 0
    out = capsys.readouterr().out
    assert "move_to_pick_approach" in out
    assert "return_home" in out


def test_cli_simulate_writes_outputs(tmp_path, capsys):
    scenario = os.path.join(ROOT, "examples", "scenarios", "pick_place_2r.json")
    out = tmp_path / "sim"
    assert main(["simulate", "--scenario", scenario, "--out", str(out)]) == 0
    assert (out / "metrics.json").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["success"] is True
    assert "success: YES" in capsys.readouterr().out


def test_cli_report_rebuilds_from_metrics(tmp_path):
    scenario = os.path.join(ROOT, "examples", "scenarios", "pick_place_2r.json")
    out = tmp_path / "sim"
    assert main(["simulate", "--scenario", scenario, "--out", str(out)]) == 0
    rebuilt = tmp_path / "rebuilt"
    assert main(["report", "--metrics", str(out / "metrics.json"),
                 "--out", str(rebuilt)]) == 0
    assert (rebuilt / "report.md").exists()
    assert (rebuilt / "report.html").exists()
