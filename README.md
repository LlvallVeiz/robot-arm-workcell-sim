# robot-arm-workcell-sim

Robot arm workcell simulator: forward/inverse kinematics, motion planning,
collision checking and end-to-end pick-and-place simulation for a planar
2R arm, a planar 3R arm and a 6-DOF arm with Universal Robots UR5 geometry.

- **Kinematics:** FK for planar 2R/3R (direct trigonometry) and 6-DOF
  (standard Denavit–Hartenberg chain); analytic IK for 2R/3R
  (elbow-up/down) and damped-least-squares numeric IK for 6-DOF;
  workspace reachability and singularity (manipulability) checks.
- **Motion planning:** trapezoidal velocity profiles, cubic/quintic
  joint-space interpolation, Cartesian straight-line moves with
  orientation interpolation, and a 9-phase pick-and-place planner
  (approach → descend → grasp → lift → transit → place → release →
  retreat → home).
- **Workcell simulation:** cells defined in JSON (arm, conveyor, pallet,
  obstacles); scenario runner writes a trajectory CSV, metrics JSON,
  trajectory/joint plots and a Markdown + HTML report; batch mode runs
  every scenario in a directory.
- **Collision:** links as capsules, obstacles as spheres/circles; the
  minimum clearance over the whole executed plan is reported, and a plan
  only counts as successful while it stays positive.

The kinematics core is **pure Python stdlib** (no NumPy). Matplotlib is
only needed for PNG plots (a pure-stdlib SVG fallback exists for planar
cells) and pytest only for the test suite.

## Background / theory

A serial arm is a chain of rigid transforms. With the standard DH
convention (Denavit & Hartenberg 1955), each joint contributes
`A = RotZ(theta) · TransZ(d) · TransX(a) · RotX(alpha)`, and forward
kinematics is the product of those matrices from base to tool. The
geometric Jacobian maps joint velocities to the end-effector twist; its
inverse problem is solved analytically for the planar arms and with
damped least squares (`dq = (JᵀJ + λ²I)⁻¹Jᵀe`, Wampler 1986) for the
6-DOF arm, which stays bounded near singularities. Singularity is
measured with Yoshikawa's manipulability `w = sqrt(det(JJᵀ))`. See
[`docs/RESEARCH.md`](docs/RESEARCH.md) for the full grounding and
references.

## Project structure

```
robot-arm-workcell-sim/
├── src/robot_arm/
│   ├── vecmath.py     # pure-stdlib 3D math: vectors, 4x4 transforms, solvers
│   ├── arms.py        # arm definitions: planar 2R/3R, UR5 DH geometry
│   ├── fk.py          # forward kinematics + geometric Jacobian
│   ├── ik.py          # analytic IK (2R/3R) + numeric DLS IK (6-DOF)
│   ├── workspace.py   # reachability, manipulability, singularity, sampling
│   ├── trajectory.py  # trapezoidal / cubic / quintic profiles, joint & Cartesian paths
│   ├── collision.py   # capsule-vs-sphere/circle clearance
│   ├── cell.py        # workcell definition (JSON), conveyor, pallet, validation
│   ├── task.py        # 9-phase pick-and-place planner
│   ├── simulate.py    # scenario runner: plan -> execute -> metrics -> files
│   ├── visualize.py   # matplotlib plots + stdlib SVG fallback
│   ├── report.py      # Markdown/HTML summary reports
│   └── cli.py         # `robot-arm` command-line interface
├── tests/             # pytest suite (119 tests)
├── examples/
│   ├── scenarios/     # 3 workcells + 3 pick-and-place scenarios (JSON)
│   ├── output/        # verified run outputs (CSV, metrics, plots, reports)
│   └── RESULTS.md     # real numbers from the example runs
├── docs/
│   └── RESEARCH.md    # research grounding + references
├── requirements.txt
├── pyproject.toml
└── LICENSE            # MIT
```

## Installation

```bash
pip install -e .          # installs the `robot-arm` command (needs matplotlib)
# or, without installing:
export PYTHONPATH=src
python -m robot_arm.cli --help
```

Python 3.9+; the core needs nothing else. `pytest` runs the tests.

## Usage

```bash
# Forward kinematics: planar 2R arm at 30 and 45 degrees
robot-arm fk --arm 2r --angles 30 45

# Inverse kinematics: analytic 2R, and numeric 6-DOF (tool pointing down)
robot-arm ik --arm 2r --target 0.5365 0.6364
robot-arm ik --arm 6dof --target 0.4 -0.2 0.3

# Plan one pick-and-place task in a cell
robot-arm plan --cell examples/scenarios/cells/cell_2r.json \
    --pick conveyor --place-slot 0

# Run one scenario end to end (CSV + metrics + plots + reports)
robot-arm simulate --scenario examples/scenarios/pick_place_6dof.json \
    --out out/6dof

# Run every scenario in a directory
robot-arm batch --dir examples/scenarios --out out

# Rebuild reports from a metrics file
robot-arm report --metrics out/6dof/metrics.json --out out/6dof
```

## Example output (real runs, 21 Nov 2026)

FK hand-check — planar 2R, L1 = 0.5 m, L2 = 0.4 m, angles 30°/45°:

```
$ robot-arm fk --arm 2r --angles 30 45
arm: planar-2r  (Planar 2R arm, L1=0.5 m, L2=0.4 m)
joints (x, y) [m]:
  joint 0: (0.0000, 0.0000)
  joint 1: (0.4330, 0.2500)
  joint 2: (0.5365, 0.6364)
end effector: (0.5365, 0.6364) m
tool orientation: 75.00 deg
```

This matches the hand computation exactly:
x = 0.5·cos30° + 0.4·cos75° = 0.5365 m,
y = 0.5·sin30° + 0.4·sin75° = 0.6364 m.

IK round-trip — FK then analytic IK returns the same end-effector
position with error `0.00e+00 m`; numeric 6-DOF IK from the home seed
converges in 16 iterations to a position error of `6.62e-11 m`
(`robot-arm ik --arm 6dof --target 0.4 -0.2 0.3`). Analytic and numeric
IK agree to < 1e-6 m on the planar arms.

UR5 FK cross-check (independent NumPy DH implementation, same table):
all-zero joints → `(-0.81725, -0.19145, -0.005491)` m; home pose
`[0, -90, 90, -90, -90, 0]` deg → `(-0.4869, -0.10915, 0.431859)` m.

Pick-and-place scenarios (all three succeed; full tables in
[`examples/RESULTS.md`](examples/RESULTS.md)):

| Scenario | Cycle time | Joint travel | Max joint speed | Min clearance | Samples |
| --- | --- | --- | --- | --- | --- |
| 2R conveyor → pallet slot 0 | 7.02 s | 8.712 rad | 2.022 rad/s | 0.2052 m | 713 |
| 3R conveyor → pallet slot 0 | 6.76 s | 12.916 rad | 2.389 rad/s | 0.3361 m | 686 |
| 6-DOF UR5 conveyor → pallet slot 0 | 12.08 s | 24.253 rad | 2.046 rad/s | 0.4045 m | 1218 |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
# 119 passed
```

Coverage by module: `vecmath` (12), `fk` (17, incl. the hand-computed
2R check and the UR5 cross-check), `ik` (13, incl. round-trips and
analytic-vs-numeric agreement), `workspace` (12, incl. manipulability
zero at full stretch and at the UR5 wrist singularity), `trajectory`
(16), `collision` (10), `cell` (8), `task` (10, incl. end-to-end plans
on all three arms), `simulate` (7), `visualize` (4), `report` (3),
`cli` (8).

## Design decisions

- **Stdlib-only core.** Kinematics, planning and simulation use no
  third-party code, so the package runs on any Python 3.9+ install and
  the math stays auditable. Matplotlib is an optional plotting
  dependency only.
- **Standard DH, not modified DH.** The UR5 table is the standard-DH
  form from the Johns Hopkins notes; mixing conventions would silently
  shift the tool pose, so the choice is pinned in `arms.py` and in
  `docs/RESEARCH.md`.
- **Seeded IK for continuity.** Approach-pose IK is seeded from the
  current configuration (home, then the pick-approach solution) before
  falling back to multi-start, so joint-space moves stay short and
  don't swing the elbow through obstacles. Cartesian segments seed each
  sample from the previous solution.
- **Capsule collision model.** Links are segments with a radius and
  obstacles are spheres/circles. It is conservative and fast enough to
  check every trajectory sample, which is what makes the reported
  minimum clearance trustworthy.
- **Obstacles are placed where the arm doesn't sweep.** Two early
  placements were caught by the simulator itself: a column at
  `(0.45, 0.55)` sat on the 2R transit path (clearance went negative)
  and a fixture post at `(0.35, -0.35, 0.35)` sat inside the UR5 swept
  volume (clearance −0.12 m). Both were moved to verified-clear spots;
  the history is documented in `cell.py`.
- **3R reachability is wrist-limited.** With the tool pointing straight
  down, the wrist must stay within the first two links' reach
  (0.75 m), so the 3R pallet sits lower and closer than the 2R one.
  The planner reports unreachable poses instead of silently
  approximating them.

## Limitations & future work

- Kinematics only: no dynamics, torque limits, gravity or friction;
  joint speeds are profile limits, not motor limits.
- Collision model is bounding-volume only (capsules/spheres); no mesh
  geometry, no self-collision check between non-adjacent links, and no
  obstacle avoidance in the planner — a plan that would collide is
  reported as failed, not rerouted.
- The numeric IK needs a reachable target and a reasonable seed; it
  reports non-convergence instead of returning a wrong pose.
- The 6-DOF arm has no analytic IK; the wrist is not spherical in the
  UR5 geometry, so a closed form is not attempted.
- No gripper dynamics, payload, or conveyor timing simulation; the
  conveyor is a pick pose, the pallet a grid of slots.
- On this VM the Debian-packaged `mpl_toolkits` is broken, so the 6-DOF
  plot falls back to three 2D projections (top/front/side) instead of a
  3D view; on a normal matplotlib install the 3D plot is used.
