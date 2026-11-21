# Example results — real runs, 21 Nov 2026

All numbers below come from actual runs of the CLI on this machine
(`robot-arm batch --dir examples/scenarios --out examples/output`);
nothing here is hand-written. Reproduce with:

```bash
PYTHONPATH=src python -m pytest tests/ -q        # 119 passed
PYTHONPATH=src python -m robot_arm.cli batch --dir examples/scenarios --out examples/output
```

## Validation spot-checks

| Check | Result |
| --- | --- |
| FK 2R hand computation, L = (0.5, 0.4) m, angles (30°, 45°) | EE (0.536540, 0.636370) m, phi 75° — matches hand computation to 1e-12 |
| IK 2R round-trip (FK → IK → FK) | position error 0.00e+00 m |
| IK 3R round-trip incl. tool orientation | position error < 1e-9 m |
| Numeric IK 6-DOF, target (0.4, -0.2, 0.3), tool down | 16 iterations, position error 6.62e-11 m, orientation error 0.00e+00 rad |
| Numeric IK 6-DOF from home seed, FK target of [0.3, -1.2, 0.9, 0.5, -0.7, 0.2] rad | 11 iterations, position error ~6e-10 m |
| Analytic vs numeric IK agreement (planar arms) | < 1e-6 m |
| UR5 FK, all-zero joints (cross-checked vs independent NumPy DH) | (-0.81725, -0.19145, -0.005491) m |
| UR5 FK, home pose [0, -90, 90, -90, -90, 0]° (same cross-check) | (-0.4869, -0.10915, 0.431859) m |
| Manipulability, planar 2R at theta2 = 90° | 0.200 (max, = L1·L2); 0 at full stretch (theta2 = 0°) |
| Manipulability, UR5 home pose | ≈ 0.08117; exactly 0.0 at wrist singularity (q5 = 0) |
| Trapezoidal profile, distance 1.0 m, vmax 0.5 m/s, amax 1.0 m/s² | total time 2.5 s (T = d/vmax + vmax/amax), peak velocity 0.5 m/s, ends at 1.0 m |
| Quintic profile, 0 → 1 in 2 s | midpoint 0.5, zero end velocity/acceleration, max velocity 0.9375 |
| UR5 max reach (sum \|a\| + \|d\|) | 1.19251 m |

## Scenario runs (pick conveyor → pallet slot 0, 9 phases each)

### pick-place-2r-conveyor-to-pallet — 7.02 s, 713 samples

| Phase | Duration [s] | Gripper | Travel [rad] | Max speed [rad/s] | Min clearance [m] |
| --- | --- | --- | --- | --- | --- |
| move_to_pick_approach | 2.443 | open | 2.822 | 1.875 | 0.5731 |
| descend_to_pick | 0.200 | open | 0.270 | 0.843 | 0.5731 |
| grasp | 0.400 | closed | 0.000 | 0.000 | 0.5940 |
| lift_from_pick | 0.200 | closed | 0.340 | 1.112 | 0.5676 |
| transit_to_place_approach | 0.773 | closed | 0.971 | 1.875 | 0.2486 |
| descend_to_place | 0.200 | closed | 0.464 | 2.022 | 0.2052 |
| release | 0.400 | open | 0.000 | 0.000 | 0.2052 |
| retreat_from_place | 0.200 | open | 0.464 | 2.022 | 0.2052 |
| return_home | 2.202 | open | 3.381 | 1.875 | 0.2486 |

Totals: joint travel 8.712 rad, max joint speed 2.022 rad/s,
minimum clearance **0.2052 m** (at the place pose, elbow vs column).

### pick-place-3r-conveyor-to-pallet — 6.76 s, 686 samples

| Phase | Duration [s] | Gripper | Travel [rad] | Max speed [rad/s] | Min clearance [m] |
| --- | --- | --- | --- | --- | --- |
| move_to_pick_approach | 2.184 | open | 4.251 | 1.875 | 0.5283 |
| descend_to_pick | 0.200 | open | 0.594 | 1.728 | 0.5119 |
| grasp | 0.400 | closed | 0.000 | 0.000 | 0.5119 |
| lift_from_pick | 0.200 | closed | 0.778 | 2.389 | 0.5119 |
| transit_to_place_approach | 0.404 | closed | 0.808 | 1.875 | 0.3832 |
| descend_to_place | 0.200 | closed | 0.736 | 2.136 | 0.3361 |
| release | 0.400 | open | 0.000 | 0.000 | 0.3361 |
| retreat_from_place | 0.200 | open | 0.736 | 2.136 | 0.3361 |
| return_home | 2.567 | open | 5.013 | 1.875 | 0.3832 |

Totals: joint travel 12.916 rad, max joint speed 2.389 rad/s,
minimum clearance **0.3361 m**.

### pick-place-6dof-conveyor-to-pallet — 12.08 s, 1218 samples

| Phase | Duration [s] | Gripper | Travel [rad] | Max speed [rad/s] | Min clearance [m] |
| --- | --- | --- | --- | --- | --- |
| move_to_pick_approach | 3.992 | open | 7.786 | 1.875 | 0.4045 |
| descend_to_pick | 0.200 | open | 0.646 | 1.637 | 0.5779 |
| grasp | 0.400 | closed | 0.000 | 0.000 | 0.5779 |
| lift_from_pick | 0.200 | closed | 0.809 | 2.046 | 0.5779 |
| transit_to_place_approach | 1.246 | closed | 3.039 | 1.875 | 0.5550 |
| descend_to_place | 0.200 | closed | 0.806 | 2.036 | 0.5550 |
| release | 0.400 | open | 0.000 | 0.000 | 0.5706 |
| retreat_from_place | 0.200 | open | 0.806 | 2.036 | 0.5550 |
| return_home | 4.637 | open | 9.315 | 1.875 | 0.4045 |

Totals: joint travel 24.253 rad, max joint speed 2.046 rad/s,
minimum clearance **0.4045 m** (fixture post at (-0.5, 0.5, 0.3)).

The long `move_to_pick_approach` / `return_home` phases are real: the
home-seeded IK solution for the pick approach is
[138.7°, -75.7°, 115.0°, -129.3°, -90.0°, 228.7°], a large reconfiguration
from home, mostly in joints 1 and 6.

## Output files per scenario (`examples/output/<scenario>/`)

- `trajectory.csv` — per-sample log (t, joints, joint velocities, EE
  position, gripper, clearance); 713 / 686 / 1218 rows.
- `metrics.json` — the numbers in the tables above.
- `trajectory.png` — workcell plot (2D for planar arms; three 2D
  projections for the 6-DOF arm on this VM, where the Debian-packaged
  `mpl_toolkits` is broken — see README).
- `trajectory.svg` — pure-stdlib SVG fallback (planar scenarios only).
- `joints.png` — joint angles and velocities vs time.
- `report.md` / `report.html` — summary reports.
