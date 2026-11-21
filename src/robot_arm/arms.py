"""Arm definitions for ``robot-arm-workcell-sim``.

Three arms are provided:

* :func:`planar_2r` — a planar 2-link arm (lengths in metres).
* :func:`planar_3r` — a planar 3-link arm with a wrist orientation angle.
* :func:`ur5_arm` — a 6-DOF articulated arm using the published standard
  Denavit-Hartenberg parameters of the Universal Robots UR5 (Keating 2014,
  DTU RSE wiki; see ``docs/RESEARCH.md``).  The UR5 values are real
  manufacturer geometry, not invented numbers, which makes the 6-DOF
  forward kinematics independently checkable.

All angles are in radians and all lengths in metres.  Planar arms are also
expressed as DH chains (``d = 0``, ``alpha = 0``, ``a = link length``), so
the generic DH forward kinematics in :mod:`robot_arm.fk` works for every
arm in this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DHLink:
    """One link of a standard-DH chain.

    Attributes:
        d: offset along the previous z-axis (metres).
        a: link length along the current x-axis (metres).
        alpha: twist about the current x-axis (radians).
        theta_offset: constant offset added to the joint variable (radians).
    """

    d: float
    a: float
    alpha: float
    theta_offset: float = 0.0


@dataclass
class Arm:
    """A serial revolute arm described by a standard-DH chain.

    Attributes:
        name: short identifier, e.g. ``"planar-2r"`` or ``"ur5"``.
        description: one-line human description.
        dh: list of :class:`DHLink`, one per joint, base to tip.
        joint_limits: ``(lower, upper)`` radian limits per joint.
        link_radii: collision-capsule radius (metres) per link.
        home: a safe "ready" joint configuration (radians).
        planar: True for arms that move in the x-y plane only.
        lengths: link lengths (metres); filled for planar arms.
    """

    name: str
    description: str
    dh: list
    joint_limits: list
    link_radii: list
    home: list
    planar: bool = False
    lengths: tuple = field(default_factory=tuple)

    @property
    def dof(self):
        """Number of joints (degrees of freedom)."""
        return len(self.dh)

    def check_angles(self, thetas):
        """Return a list of limit violations for ``thetas`` (empty if OK)."""
        problems = []
        if len(thetas) != self.dof:
            problems.append(
                f"expected {self.dof} joint angles, got {len(thetas)}"
            )
            return problems
        for i, (q, (lo, hi)) in enumerate(zip(thetas, self.joint_limits)):
            if q < lo - 1e-12 or q > hi + 1e-12:
                problems.append(
                    f"joint {i + 1} angle {q:.4f} rad outside "
                    f"[{lo:.4f}, {hi:.4f}]"
                )
        return problems


def planar_2r(lengths=(0.5, 0.4)):
    """Return a planar 2R arm.

    Args:
        lengths: ``(L1, L2)`` link lengths in metres (default 0.5, 0.4).
    """
    l1, l2 = lengths
    return Arm(
        name="planar-2r",
        description=f"Planar 2R arm, L1={l1} m, L2={l2} m",
        dh=[
            DHLink(d=0.0, a=l1, alpha=0.0),
            DHLink(d=0.0, a=l2, alpha=0.0),
        ],
        joint_limits=[(-math.pi, math.pi), (-math.pi, math.pi)],
        link_radii=[0.03, 0.025],
        home=[math.radians(45.0), math.radians(45.0)],
        planar=True,
        lengths=tuple(lengths),
    )


def planar_3r(lengths=(0.4, 0.35, 0.25)):
    """Return a planar 3R arm (third link carries the tool orientation).

    Args:
        lengths: ``(L1, L2, L3)`` link lengths in metres.
    """
    l1, l2, l3 = lengths
    return Arm(
        name="planar-3r",
        description=f"Planar 3R arm, L1={l1} m, L2={l2} m, L3={l3} m",
        dh=[
            DHLink(d=0.0, a=l1, alpha=0.0),
            DHLink(d=0.0, a=l2, alpha=0.0),
            DHLink(d=0.0, a=l3, alpha=0.0),
        ],
        joint_limits=[(-math.pi, math.pi)] * 3,
        link_radii=[0.03, 0.025, 0.02],
        home=[math.radians(45.0), math.radians(30.0), math.radians(15.0)],
        planar=True,
        lengths=tuple(lengths),
    )


def ur5_arm():
    """Return a 6-DOF arm with the Universal Robots UR5 DH parameters.

    Standard DH table (metres / radians), as published in the UR5 inverse
    kinematics notes (Keating, Johns Hopkins University, 2014) and the DTU
    RSE wiki:

    =====  =========  =========  ========
    joint  a          alpha      d
    =====  =========  =========  ========
    1      0          +pi/2      0.089159
    2      -0.425     0          0
    3      -0.39225   0          0
    4      0          +pi/2      0.10915
    5      0          -pi/2      0.09465
    6      0          0          0.0823
    =====  =========  =========  ========

    Joint limits are widened to +/-2*pi (the UR5 hardware allows +/-360
    degrees per joint); the planner keeps solutions inside this range.
    """
    return Arm(
        name="ur5",
        description="6-DOF articulated arm, Universal Robots UR5 DH geometry",
        dh=[
            DHLink(d=0.089159, a=0.0, alpha=math.pi / 2),
            DHLink(d=0.0, a=-0.425, alpha=0.0),
            DHLink(d=0.0, a=-0.39225, alpha=0.0),
            DHLink(d=0.10915, a=0.0, alpha=math.pi / 2),
            DHLink(d=0.09465, a=0.0, alpha=-math.pi / 2),
            DHLink(d=0.0823, a=0.0, alpha=0.0),
        ],
        joint_limits=[(-2.0 * math.pi, 2.0 * math.pi)] * 6,
        link_radii=[0.06, 0.055, 0.05, 0.045, 0.04, 0.035],
        # A common "ready" pose: shoulder 0, elbow bent, wrist down.
        home=[
            0.0,
            -math.pi / 2,
            math.pi / 2,
            -math.pi / 2,
            -math.pi / 2,
            0.0,
        ],
        planar=False,
    )


def make_arm(kind, lengths=None):
    """Build an arm by kind name (``"2r"``, ``"3r"`` or ``"6dof"``).

    Args:
        kind: ``"2r"`` (planar 2R), ``"3r"`` (planar 3R) or ``"6dof"``
            (UR5 geometry).
        lengths: optional link lengths for the planar arms.

    Raises:
        ValueError: for an unknown ``kind``.
    """
    kind = kind.lower()
    if kind == "2r":
        return planar_2r(tuple(lengths) if lengths else (0.5, 0.4))
    if kind == "3r":
        return planar_3r(tuple(lengths) if lengths else (0.4, 0.35, 0.25))
    if kind in ("6dof", "ur5"):
        return ur5_arm()
    raise ValueError(f"unknown arm kind {kind!r} (expected '2r', '3r' or '6dof')")
