# Research notes — robot-arm-workcell-sim

How the kinematics, planning and collision models in this project are
grounded, and which sources each choice comes from.  Every reference below
is a real publication or document; URLs are included only where they were
checked during the build (21 Nov 2026).

## 1. Forward kinematics — Denavit–Hartenberg convention

A serial arm is described by the standard Denavit–Hartenberg (DH)
parameters: four numbers per joint — link length `a`, link twist `alpha`,
link offset `d`, joint angle `theta` — with the per-joint transform

    A = RotZ(theta) · TransZ(d) · TransX(a) · RotX(alpha)

and the base-to-tool transform as the product of the per-joint transforms.
This is the convention of Denavit & Hartenberg (1955) as presented in
Craig (2005) and Lynch & Park (2017).  Note that a *modified* DH
convention also exists (used by some UR documentation); this project uses
the **standard** convention throughout, and the UR5 table below is the
standard-DH form, so the two must not be mixed.

- Denavit, J. & Hartenberg, R. S. (1955). "A kinematic notation for
  lower-pair mechanisms based on matrices." *Transactions of the ASME,
  Journal of Applied Mechanics*, 22: 215–221.
- Craig, J. J. (2005). *Introduction to Robotics: Mechanics and Control*,
  3rd ed. Pearson Prentice Hall.
- Lynch, K. M. & Park, F. C. (2017). *Modern Robotics: Mechanics,
  Planning, and Control*. Cambridge University Press. ISBN 9781107156302.
  Free preprint at http://modernrobotics.org.

## 2. The 6-DOF arm — Universal Robots UR5 geometry

The 6-DOF arm uses the published standard-DH parameters of the UR5
(metres), as tabulated in the Johns Hopkins UR5 inverse-kinematics notes
and the DTU RSE wiki:

| joint | a | alpha | d |
| --- | --- | --- | --- |
| 1 | 0 | +pi/2 | 0.089159 |
| 2 | -0.425 | 0 | 0 |
| 3 | -0.39225 | 0 | 0 |
| 4 | 0 | +pi/2 | 0.10915 |
| 5 | 0 | -pi/2 | 0.09465 |
| 6 | 0 | 0 | 0.0823 |

These are real manufacturer geometry, not invented numbers.  The FK
implementation was cross-checked against an independent NumPy DH
implementation written from the same table: at all-zero joints the tool
position is `(-0.81725, -0.19145, -0.005491)` m with rotation
`[[1,0,0],[0,0,-1],[0,1,0]]`, and at the home pose
`[0, -90, 90, -90, -90, 0]` degrees it is
`(-0.4869, -0.10915, 0.431859)` m — both asserted in `tests/test_fk.py`.

- Keating, R. (2014, updated by N. J. Cowan 2016). *UR5 Inverse
  Kinematics* (M.E. 530.646 course notes, Johns Hopkins University).
  https://github.com/kopecn/spmSwiftRobotics/raw/refs/heads/dev/docs/ik/ur5_inverse_kinematics.pdf
- DTU RSE wiki, "UR5" — DH parameter table.
  https://rsewiki.electro.dtu.dk/index.php?title=UR5&oldid=1282

## 3. Inverse kinematics

**Analytic (planar 2R/3R).** The 2R solution is the classic two-link
geometry: `cos(theta2) = (x^2 + y^2 - L1^2 - L2^2) / (2 L1 L2)` with the
elbow-up (`theta2 > 0`) and elbow-down (`theta2 < 0`) branches, and
`theta1 = atan2(y, x) - atan2(L2 sin(theta2), L1 + L2 cos(theta2))`.  The
3R solution subtracts the wrist offset `L3 * (cos phi, sin phi)` and
solves the remaining 2R problem.  Both are derived in Craig (2005) and in
Lynch & Park (2017, ch. 6).

**Numeric (6-DOF, damped least squares).** The geometric Jacobian `J`
relates joint velocities to end-effector twist; the iterative update is

    dq = (J^T J + lambda^2 I)^-1 J^T e

where `e` is the pose error (position error plus the rotation-vector of
`R_target · R_current^T`).  The damping term keeps the update bounded
near singularities, at the cost of a small residual error — the standard
trade-off analysed by Wampler (1986) and Nakamura & Hanafusa (1986), and
reviewed with experiments by Chiaverini, Siciliano & Egeland (1994).
Our solver converges from the home seed in 11 iterations to a position
error of ~6e-10 m for the test target (see `tests/test_ik.py`), and the
analytic and numeric IK agree on the planar arms to < 1e-6 m.

- Wampler, C. W. (1986). "Manipulator inverse kinematic solutions based
  on vector formulations and damped least-squares methods." *IEEE
  Transactions on Systems, Man, and Cybernetics*, 16(1): 93–101.
  https://doi.org/10.1109/TSMC.1986.289285
- Nakamura, Y. & Hanafusa, H. (1986). "Inverse kinematic solutions with
  singularity robustness for robot manipulator control." *ASME Journal of
  Dynamic Systems, Measurement, and Control*, 108(3): 163–171.
- Chiaverini, S., Siciliano, B. & Egeland, O. (1994). "Review of the
  damped least-squares inverse kinematics with experiments on an
  industrial robot manipulator." *IEEE Transactions on Control Systems
  Technology*, 2(2): 123–134. https://doi.org/10.1109/87.294335

## 4. Workspace and singularities

Reachability of the planar 2R arm is the annulus `|L1 - L2| <= r <=
L1 + L2`.  Singularity is measured with Yoshikawa's manipulability,
`w = sqrt(det(J J^T))` (for the planar 2R arm this reduces to
`|L1 L2 sin(theta2)|`): it is zero at full extension (`theta2 = 0`) and
maximal at `theta2 = 90 deg`.  The UR5 home pose has manipulability
≈ 0.08117, and the wrist singularity (`q5 = 0`) gives exactly 0.0, both
verified numerically in `tests/test_workspace.py`.

- Yoshikawa, T. (1985). "Manipulability of robotic mechanisms."
  *International Journal of Robotics Research*, 4(2): 3–9.
  https://doi.org/10.1177/027836498500400201
- Siciliano, B., Sciavicco, L., Villani, L. & Oriolo, G. (2009).
  *Robotics: Modelling, Planning and Control*. Springer.

## 5. Trajectory planning

Joint-space moves use quintic (and optionally cubic) polynomial
interpolation — zero velocity and acceleration at both ends for the
quintic — and Cartesian straight lines use linear position interpolation
with axis-angle orientation interpolation.  The trapezoidal velocity
profile (accelerate / cruise / decelerate, triangular when the distance
is too short to reach cruise speed) follows the standard treatment in
Biagiotti & Melchiorri (2008) and Lynch & Park (2017, ch. 9).  Pick-and-
place is decomposed into the usual industrial phases (approach, descend,
grasp, lift, transit, place, retreat) as in Siciliano et al. (2009).

- Biagiotti, L. & Melchiorri, C. (2008). *Trajectory Planning for
  Automatic Machines and Robots*. Springer.
- Spong, M. W., Hutchinson, S. & Vidyasagar, M. (2006). *Robot Modeling
  and Control*. Wiley.

## 6. Collision checking

Links are modelled as capsules (a segment plus a radius) and obstacles
as spheres (3D) or circles (planar); the clearance of a configuration is
the minimum over links and obstacles of
`distance(segment, centre) - obstacle_radius - link_radius`.  This is the
standard bounding-volume approximation used before full mesh-based
checking; it is conservative (it never reports more clearance than the
real geometry has) and cheap enough to evaluate at every trajectory
sample.  The reported "minimum collision clearance" is the smallest such
margin over the whole executed plan; a plan is only reported successful
when it stays strictly positive.

## 7. Safety context (background only)

Industrial robot cells are governed by ISO 10218 (safety requirements
for industrial robots); this simulator models kinematics and geometry
only — it does not implement safety-rated monitoring, speed/separation
monitoring, or protective stops, and must not be mistaken for a safety
controller.

- ISO 10218-1:2011. *Robots and robotic devices — Safety requirements
  for industrial robots — Part 1: Robots*. International Organization
  for Standardization.

## References (all)

1. Denavit, J. & Hartenberg, R. S. (1955). "A kinematic notation for
   lower-pair mechanisms based on matrices." *Trans. ASME J. Appl.
   Mech.*, 22: 215–221.
2. Craig, J. J. (2005). *Introduction to Robotics: Mechanics and
   Control*, 3rd ed. Pearson Prentice Hall.
3. Lynch, K. M. & Park, F. C. (2017). *Modern Robotics: Mechanics,
   Planning, and Control*. Cambridge University Press.
   http://modernrobotics.org
4. Siciliano, B., Sciavicco, L., Villani, L. & Oriolo, G. (2009).
   *Robotics: Modelling, Planning and Control*. Springer.
5. Spong, M. W., Hutchinson, S. & Vidyasagar, M. (2006). *Robot
   Modeling and Control*. Wiley.
6. Biagiotti, L. & Melchiorri, C. (2008). *Trajectory Planning for
   Automatic Machines and Robots*. Springer.
7. Wampler, C. W. (1986). "Manipulator inverse kinematic solutions
   based on vector formulations and damped least-squares methods."
   *IEEE Trans. Syst. Man Cybern.*, 16(1): 93–101.
   https://doi.org/10.1109/TSMC.1986.289285
8. Nakamura, Y. & Hanafusa, H. (1986). "Inverse kinematic solutions
   with singularity robustness for robot manipulator control." *ASME
   J. Dyn. Syst. Meas. Control*, 108(3): 163–171.
9. Chiaverini, S., Siciliano, B. & Egeland, O. (1994). "Review of the
   damped least-squares inverse kinematics with experiments on an
   industrial robot manipulator." *IEEE Trans. Control Syst.
   Technol.*, 2(2): 123–134. https://doi.org/10.1109/87.294335
10. Yoshikawa, T. (1985). "Manipulability of robotic mechanisms."
    *Int. J. Robot. Res.*, 4(2): 3–9.
    https://doi.org/10.1177/027836498500400201
11. Keating, R. (2014). *UR5 Inverse Kinematics* (Johns Hopkins).
    https://github.com/kopecn/spmSwiftRobotics/raw/refs/heads/dev/docs/ik/ur5_inverse_kinematics.pdf
12. DTU RSE wiki, "UR5". https://rsewiki.electro.dtu.dk/index.php?title=UR5&oldid=1282
13. ISO 10218-1:2011. *Robots and robotic devices — Safety requirements
    for industrial robots — Part 1: Robots*. ISO.
