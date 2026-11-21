"""Tests for the Markdown/HTML report builders (report.py)."""

from robot_arm.report import build_html_report, build_markdown_report

RESULT = {
    "scenario": "pick-place-2r-conveyor-to-pallet",
    "description": "Demo pick and place.",
    "cell": "demo-cell-2r",
    "success": True,
    "cycle_time_s": 7.0178,
    "total_joint_travel_rad": 8.7119,
    "max_joint_speed_rad_s": 2.022,
    "min_clearance_m": 0.2052,
    "n_samples": 713,
    "pick_m": [0.55, 0.15, 0.0],
    "place_m": [0.15, 0.55, 0.0],
    "phases": [
        {"label": "move_to_pick_approach", "duration_s": 1.2,
         "gripper": "open", "cartesian": False, "joint_travel_rad": 1.0,
         "max_joint_speed_rad_s": 1.0, "min_clearance_m": 0.3},
        {"label": "descend_to_pick", "duration_s": 0.8,
         "gripper": "open", "cartesian": True, "joint_travel_rad": 0.2,
         "max_joint_speed_rad_s": 0.5, "min_clearance_m": None},
    ],
    "notes": [],
    "files": {"trajectory_csv": "trajectory.csv"},
}


def test_build_markdown_report(tmp_path):
    out = tmp_path / "report.md"
    build_markdown_report(RESULT, str(out))
    text = out.read_text()
    assert "# Simulation report — pick-place-2r-conveyor-to-pallet" in text
    assert "| Cycle time | 7.02 s |" in text
    assert "| Min collision clearance | 0.2052 m |" in text
    assert "move_to_pick_approach" in text
    assert "trajectory.csv" in text


def test_build_html_report(tmp_path):
    out = tmp_path / "report.html"
    build_html_report(RESULT, str(out))
    text = out.read_text()
    assert "<!DOCTYPE html>" in text
    assert "pick-place-2r-conveyor-to-pallet" in text
    assert "7.02 s" in text
    assert "descend_to_pick" in text


def test_build_reports_with_no_obstacles(tmp_path):
    result = dict(RESULT, min_clearance_m=None, notes=["no obstacles"])
    md = tmp_path / "report.md"
    build_markdown_report(result, str(md))
    assert "n/a (no obstacles)" in md.read_text()
    assert "no obstacles" in md.read_text()
