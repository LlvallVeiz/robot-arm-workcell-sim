"""Command-line interface for ``robot-arm-workcell-sim``.

Subcommands::

    robot-arm fk        --arm 2r --angles 30 45
    robot-arm ik        --arm 2r --target 0.54 0.64
    robot-arm ik        --arm 6dof --target 0.4 -0.2 0.3
    robot-arm plan      --cell cells/cell_2r.json --pick conveyor --place-slot 0
    robot-arm simulate  --scenario scenarios/pick_place_2r.json --out out/2r
    robot-arm batch     --dir scenarios --out out
    robot-arm report    --metrics out/2r/metrics.json --out out/2r

Angles on the command line are in degrees; positions in metres.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from .arms import make_arm
from .cell import load_workcell, pallet_slot_positions
from .fk import end_effector_pose, fk_planar
from .ik import ik_2r, ik_3r, ik_numeric_multi_start
from .report import build_html_report, build_markdown_report
from .simulate import run_batch, run_scenario
from .task import TOOL_DOWN_ROT, PickPlaceRequest, plan_pick_place


def _deg_list(values):
    return [math.radians(float(v)) for v in values]


def _fmt_vec(vec, unit=""):
    return "(" + ", ".join(f"{v:.4f}{unit}" for v in vec) + ")"


def cmd_fk(args):
    """Print forward kinematics for the requested arm and joint angles."""
    arm = make_arm(args.arm)
    thetas = _deg_list(args.angles)
    if len(thetas) != arm.dof:
        print(f"error: {arm.name} needs {arm.dof} angles, got {len(thetas)}",
              file=sys.stderr)
        return 2
    if arm.planar:
        fk = fk_planar(arm.lengths, thetas)
        print(f"arm: {arm.name}  ({arm.description})")
        print("joints (x, y) [m]:")
        for i, (x, y) in enumerate(fk.joints):
            print(f"  joint {i}: ({x:.4f}, {y:.4f})")
        print(f"end effector: ({fk.ee[0]:.4f}, {fk.ee[1]:.4f}) m")
        print(f"tool orientation: {math.degrees(fk.phi):.2f} deg")
    else:
        pos, rot = end_effector_pose(arm, thetas)
        print(f"arm: {arm.name}  ({arm.description})")
        print(f"end effector: {_fmt_vec(pos)} m")
        print("rotation matrix:")
        for row in rot:
            print("  " + " ".join(f"{v: .4f}" for v in row))
    return 0


def cmd_ik(args):
    """Print inverse kinematics solutions for a target pose."""
    arm = make_arm(args.arm)
    if arm.planar:
        if len(args.target) < 2:
            print("error: planar IK needs at least x y", file=sys.stderr)
            return 2
        x, y = float(args.target[0]), float(args.target[1])
        if arm.dof == 2:
            sol = ik_2r(x, y, *arm.lengths, elbow=args.elbow)
            if sol is None:
                print(f"target ({x:.4f}, {y:.4f}) is unreachable")
                return 1
            print(f"IK 2R elbow={args.elbow}: "
                  f"theta1={math.degrees(sol[0]):.4f} deg, "
                  f"theta2={math.degrees(sol[1]):.4f} deg")
        else:
            phi = math.radians(float(args.phi)) if args.phi is not None else 0.0
            sol = ik_3r(x, y, phi, arm.lengths, elbow=args.elbow)
            if sol is None:
                print(f"target ({x:.4f}, {y:.4f}, phi={math.degrees(phi):.1f} deg) "
                      "is unreachable")
                return 1
            print(f"IK 3R elbow={args.elbow}: " + ", ".join(
                f"theta{i + 1}={math.degrees(v):.4f} deg"
                for i, v in enumerate(sol)
            ))
        # Verify by re-running FK.
        thetas = list(sol)
        fk = fk_planar(arm.lengths, thetas)
        err = math.hypot(fk.ee[0] - x, fk.ee[1] - y)
        print(f"FK check: ee=({fk.ee[0]:.6f}, {fk.ee[1]:.6f}) m, "
              f"position error={err:.2e} m")
        return 0
    # 6-DOF: numeric IK.
    if len(args.target) < 3:
        print("error: 6-DOF IK needs x y z", file=sys.stderr)
        return 2
    target = tuple(float(v) for v in args.target[:3])
    result = ik_numeric_multi_start(arm, target, TOOL_DOWN_ROT)
    if not result.converged:
        print(f"numeric IK did not converge for target {_fmt_vec(target)} "
              f"(pos error {result.pos_error:.2e} m after "
              f"{result.iterations} iterations)")
        return 1
    print("IK 6-DOF (damped least squares, tool pointing down):")
    print("  joints [deg]: " + ", ".join(f"{math.degrees(v):.4f}" for v in result.q))
    print(f"  iterations: {result.iterations}, "
          f"position error: {result.pos_error:.2e} m, "
          f"orientation error: {result.rot_error:.2e} rad")
    return 0


def _resolve_pick_place(cell, args):
    pick_arg = args.pick
    if pick_arg == "conveyor" or pick_arg == ["conveyor"]:
        if cell.conveyor is None:
            print("error: cell has no conveyor", file=sys.stderr)
            sys.exit(2)
        pick = tuple(cell.conveyor.pick)
    else:
        pick = tuple(float(v) for v in pick_arg)
    if args.place_slot is not None:
        slots = pallet_slot_positions(cell)
        if not 0 <= args.place_slot < len(slots):
            print(f"error: place slot {args.place_slot} out of range "
                  f"(pallet has {len(slots)} slots)", file=sys.stderr)
            sys.exit(2)
        place = slots[args.place_slot]
    else:
        place = tuple(float(v) for v in args.place)
    return pick, place


def cmd_plan(args):
    """Plan a pick-and-place task and print the phase summary."""
    cell = load_workcell(args.cell)
    pick, place = _resolve_pick_place(cell, args)
    request = PickPlaceRequest(
        pick=pick, place=place,
        approach_height=args.approach, lift_height=args.lift,
        dwell=args.dwell, transit_speed=args.speed,
    )
    plan = plan_pick_place(cell, request, dt=args.dt)
    print(f"workcell: {cell.name}  (arm: {cell.arm.name})")
    print(f"pick {_fmt_vec(pick)} m  ->  place {_fmt_vec(place)} m")
    print(f"total time: {plan.total_time:.2f} s, "
          f"{len(plan.segments)} phases")
    for seg in plan.segments:
        print(f"  {seg.label:28s} {seg.trajectory.duration:6.2f} s  "
              f"gripper={seg.gripper:6s} cartesian={'yes' if seg.cartesian else 'no'}")
    if plan.ik_failures:
        print("IK failures:", plan.ik_failures, file=sys.stderr)
        return 1
    return 0


def cmd_simulate(args):
    """Run one scenario and print its metrics."""
    result = run_scenario(args.scenario, args.out)
    print(f"scenario: {result['scenario']}  (cell: {result['cell']})")
    print(f"success: {'YES' if result['success'] else 'NO'}")
    print(f"cycle time: {result['cycle_time_s']:.2f} s")
    print(f"total joint travel: {result['total_joint_travel_rad']:.3f} rad")
    print(f"max joint speed: {result['max_joint_speed_rad_s']:.3f} rad/s")
    clearance = result.get("min_clearance_m")
    print("min collision clearance: "
          f"{clearance:.4f} m" if clearance is not None else "n/a")
    print(f"samples: {result['n_samples']}")
    print(f"outputs written to: {os.path.abspath(args.out)}")
    for note in result.get("notes", []):
        print(f"note: {note}")
    return 0 if result["success"] else 1


def cmd_batch(args):
    """Run every scenario in a directory."""
    results = run_batch(args.dir, args.out)
    print(f"ran {len(results)} scenario(s)")
    for result in results:
        status = "OK " if result["success"] else "FAIL"
        print(f"  [{status}] {result['scenario']}: "
              f"{result['cycle_time_s']:.2f} s, "
              f"min clearance "
              f"{result['min_clearance_m']} m")
    return 0 if all(r["success"] for r in results) else 1


def cmd_report(args):
    """Rebuild the Markdown/HTML reports from a metrics JSON file."""
    with open(args.metrics) as fh:
        result = json.load(fh)
    os.makedirs(args.out, exist_ok=True)
    md = build_markdown_report(result, os.path.join(args.out, "report.md"))
    html_path = build_html_report(result, os.path.join(args.out, "report.html"))
    print(f"wrote {md}")
    print(f"wrote {html_path}")
    return 0


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="robot-arm",
        description="Robot arm workcell simulator: FK/IK, motion planning, "
                    "pick-and-place simulation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fk", help="forward kinematics")
    p.add_argument("--arm", choices=["2r", "3r", "6dof"], default="2r")
    p.add_argument("--angles", nargs="+", required=True,
                   help="joint angles in degrees")
    p.set_defaults(func=cmd_fk)

    p = sub.add_parser("ik", help="inverse kinematics")
    p.add_argument("--arm", choices=["2r", "3r", "6dof"], default="2r")
    p.add_argument("--target", nargs="+", required=True,
                   help="target x y [z] in metres")
    p.add_argument("--phi", default=None,
                   help="target tool orientation in degrees (planar 3R)")
    p.add_argument("--elbow", choices=["up", "down"], default="down")
    p.set_defaults(func=cmd_ik)

    p = sub.add_parser("plan", help="plan a pick-and-place task")
    p.add_argument("--cell", required=True, help="workcell JSON file")
    p.add_argument("--pick", nargs="+", default="conveyor",
                   help="'conveyor' or x y z in metres")
    p.add_argument("--place", nargs=3, type=float, default=None)
    p.add_argument("--place-slot", type=int, default=None)
    p.add_argument("--approach", type=float, default=0.12)
    p.add_argument("--lift", type=float, default=0.15)
    p.add_argument("--dwell", type=float, default=0.4)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("simulate", help="run one scenario end to end")
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("batch", help="run every scenario in a directory")
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("report", help="rebuild reports from a metrics JSON")
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv=None):
    """CLI entry point (``robot-arm`` console script)."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
