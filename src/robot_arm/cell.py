"""Workcell definition: arm, obstacles, conveyor, pallet and limits.

A workcell is loaded from a small JSON file (stdlib ``json`` only) or built
from the built-in demo cells.  All lengths are metres, angles radians
(``home_deg`` in JSON files is converted to radians on load).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from robot_arm.arms import make_arm, planar_2r, ur5_arm
from robot_arm.collision import CircleObstacle, SphereObstacle
from robot_arm.vecmath import vnorm


@dataclass
class Conveyor:
    """A conveyor pick point.

    Attributes:
        pick: ``(x, y, z)`` pick position in metres.
        approach_height: height above the pick for the approach pose.
        name: optional label.
    """

    pick: tuple
    approach_height: float
    name: str = "conveyor"


@dataclass
class Pallet:
    """A rectangular pallet grid of place slots.

    Attributes:
        origin: ``(x, y, z)`` position of the first slot.
        rows: number of rows.
        cols: number of columns.
        dx: column spacing in metres.
        dy: row spacing in metres.
    """

    origin: tuple
    rows: int
    cols: int
    dx: float
    dy: float


@dataclass
class Workcell:
    """A robot workcell.

    Attributes:
        name: cell name.
        arm: the :class:`robot_arm.arms.Arm` mounted in the cell.
        obstacles: list of :class:`SphereObstacle` (3D) or
            :class:`CircleObstacle` (planar) obstacles.
        conveyor: optional conveyor pick point.
        pallet: optional pallet grid.
        max_joint_speed: joint-speed limit (rad/s).
        max_joint_accel: joint-acceleration limit (rad/s^2).
        home: home joint configuration; defaults to ``arm.home``.
    """

    name: str
    arm: object
    obstacles: list = field(default_factory=list)
    conveyor: object = None
    pallet: object = None
    max_joint_speed: float = 1.5
    max_joint_accel: float = 4.0
    home: list = None

    def __post_init__(self):
        """Default ``home`` to the arm's home configuration."""
        if self.home is None and self.arm is not None:
            self.home = list(self.arm.home)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _vec3(value, field_name):
    """Validate a ``[x, y, z]`` JSON value and return it as a tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must be a list of 3 numbers")
    if not all(_is_number(v) for v in value):
        raise ValueError(f"{field_name} must be a list of 3 numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def _positive_number(value, field_name):
    if not _is_number(value):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def load_workcell(path):
    """Load a :class:`Workcell` from a JSON file.

    Schema::

        {"name": str,
         "arm": {"kind": "2r" | "3r" | "6dof", "lengths": [...] (optional)},
         "obstacles": [{"center": [x, y, z], "radius": r, "name": "..."}],
         "conveyor": {"pick": [x, y, z], "approach_height": h},
         "pallet": {"origin": [x, y, z], "rows": n, "cols": n, "dx": d, "dy": d},
         "limits": {"max_joint_speed": v, "max_joint_accel": a},
         "home_deg": [...] (degrees, converted to radians)}

    All sections except ``"arm"`` are optional.

    Args:
        path: path to the JSON file.

    Returns:
        The loaded :class:`Workcell`.

    Raises:
        ValueError: on invalid JSON, a missing ``"arm"``/``"kind"``, an
            unknown arm kind, or wrong field types.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in workcell file {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"workcell file {path!r} must contain a JSON object")
    if "arm" not in data:
        raise ValueError(f"workcell file {path!r} is missing required field 'arm'")
    arm_spec = data["arm"]
    if not isinstance(arm_spec, dict):
        raise ValueError("'arm' must be an object with a 'kind' field")
    if "kind" not in arm_spec:
        raise ValueError("'arm' is missing required field 'kind'")
    kind = arm_spec["kind"]
    if not isinstance(kind, str):
        raise ValueError("'arm.kind' must be a string")
    lengths = arm_spec.get("lengths")
    if lengths is not None:
        if not isinstance(lengths, list) or not all(_is_number(v) for v in lengths):
            raise ValueError("'arm.lengths' must be a list of numbers")
        lengths = [float(v) for v in lengths]
    try:
        arm = make_arm(kind, lengths)
    except ValueError as exc:
        raise ValueError(f"unknown arm kind {kind!r}: {exc}") from exc

    name = data.get("name", "workcell")
    if not isinstance(name, str):
        raise ValueError("'name' must be a string")

    obstacles = []
    raw_obstacles = data.get("obstacles", [])
    if not isinstance(raw_obstacles, list):
        raise ValueError("'obstacles' must be a list")
    for i, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            raise ValueError(f"obstacles[{i}] must be an object")
        if "center" not in raw or "radius" not in raw:
            raise ValueError(f"obstacles[{i}] needs 'center' and 'radius'")
        center = _vec3(raw["center"], f"obstacles[{i}].center")
        radius = _positive_number(raw["radius"], f"obstacles[{i}].radius")
        obs_name = raw.get("name", "")
        if not isinstance(obs_name, str):
            raise ValueError(f"obstacles[{i}].name must be a string")
        obstacles.append(SphereObstacle(center=center, radius=radius, name=obs_name))

    conveyor = None
    if "conveyor" in data and data["conveyor"] is not None:
        raw = data["conveyor"]
        if not isinstance(raw, dict):
            raise ValueError("'conveyor' must be an object")
        if "pick" not in raw:
            raise ValueError("'conveyor' is missing required field 'pick'")
        pick = _vec3(raw["pick"], "conveyor.pick")
        approach = _positive_number(
            raw.get("approach_height", 0.12), "conveyor.approach_height"
        )
        conv_name = raw.get("name", "conveyor")
        if not isinstance(conv_name, str):
            raise ValueError("'conveyor.name' must be a string")
        conveyor = Conveyor(pick=pick, approach_height=approach, name=conv_name)

    pallet = None
    if "pallet" in data and data["pallet"] is not None:
        raw = data["pallet"]
        if not isinstance(raw, dict):
            raise ValueError("'pallet' must be an object")
        for key in ("origin", "rows", "cols", "dx", "dy"):
            if key not in raw:
                raise ValueError(f"'pallet' is missing required field '{key}'")
        origin = _vec3(raw["origin"], "pallet.origin")
        if not isinstance(raw["rows"], int) or isinstance(raw["rows"], bool):
            raise ValueError("'pallet.rows' must be an integer")
        if not isinstance(raw["cols"], int) or isinstance(raw["cols"], bool):
            raise ValueError("'pallet.cols' must be an integer")
        dx = _positive_number(raw["dx"], "pallet.dx")
        dy = _positive_number(raw["dy"], "pallet.dy")
        pallet = Pallet(origin=origin, rows=raw["rows"], cols=raw["cols"],
                        dx=dx, dy=dy)

    max_joint_speed = 1.5
    max_joint_accel = 4.0
    if "limits" in data and data["limits"] is not None:
        raw = data["limits"]
        if not isinstance(raw, dict):
            raise ValueError("'limits' must be an object")
        if "max_joint_speed" in raw:
            max_joint_speed = _positive_number(
                raw["max_joint_speed"], "limits.max_joint_speed"
            )
        if "max_joint_accel" in raw:
            max_joint_accel = _positive_number(
                raw["max_joint_accel"], "limits.max_joint_accel"
            )

    home = None
    if "home_deg" in data and data["home_deg"] is not None:
        raw = data["home_deg"]
        if not isinstance(raw, list) or not all(_is_number(v) for v in raw):
            raise ValueError("'home_deg' must be a list of numbers")
        home = [math.radians(float(v)) for v in raw]
        if len(home) != arm.dof:
            raise ValueError(
                f"'home_deg' has {len(home)} joints, arm has {arm.dof}"
            )

    return Workcell(
        name=name,
        arm=arm,
        obstacles=obstacles,
        conveyor=conveyor,
        pallet=pallet,
        max_joint_speed=max_joint_speed,
        max_joint_accel=max_joint_accel,
        home=home,
    )


def pallet_slot_positions(cell):
    """Return the pallet slot positions in row-major order.

    Slot ``(row, col)`` is ``origin + (col*dx, row*dy, 0)``.  Returns an
    empty list when the cell has no pallet.
    """
    if cell.pallet is None:
        return []
    slots = []
    for row in range(cell.pallet.rows):
        for col in range(cell.pallet.cols):
            slots.append((
                cell.pallet.origin[0] + col * cell.pallet.dx,
                cell.pallet.origin[1] + row * cell.pallet.dy,
                cell.pallet.origin[2],
            ))
    return slots


def max_reach(cell):
    """Return a conservative maximum reach of the cell's arm in metres.

    For planar arms this is the sum of the link lengths; for a 6-DOF arm
    it is the sum of ``|a|`` and ``|d|`` over the DH links (1.19251 m for
    the UR5).
    """
    if cell.arm is None:
        return 0.0
    if cell.arm.planar:
        return float(sum(cell.arm.lengths))
    return float(sum(abs(link.a) + abs(link.d) for link in cell.arm.dh))


def validate_workcell(cell):
    """Validate a workcell and return a list of error strings.

    An empty list means the cell is valid.  Checks: the arm exists, the
    home configuration has the right length, pallet dimensions are
    positive, every obstacle radius is positive, and the conveyor pick
    and every pallet slot are within the arm's reach.
    """
    errors = []
    if cell.arm is None:
        errors.append("arm is None")
        return errors
    if cell.home is None or len(cell.home) != cell.arm.dof:
        got = 0 if cell.home is None else len(cell.home)
        errors.append(
            f"home has {got} joints, arm has {cell.arm.dof}"
        )
    if cell.pallet is not None:
        if cell.pallet.rows <= 0:
            errors.append(f"pallet rows must be > 0, got {cell.pallet.rows}")
        if cell.pallet.cols <= 0:
            errors.append(f"pallet cols must be > 0, got {cell.pallet.cols}")
        if cell.pallet.dx <= 0:
            errors.append(f"pallet dx must be > 0, got {cell.pallet.dx}")
        if cell.pallet.dy <= 0:
            errors.append(f"pallet dy must be > 0, got {cell.pallet.dy}")
    for obs in cell.obstacles:
        if obs.radius <= 0:
            errors.append(
                f"obstacle {obs.name!r} radius must be > 0, got {obs.radius}"
            )
    reach = max_reach(cell)
    if cell.conveyor is not None:
        if vnorm(cell.conveyor.pick) > reach + 1e-9:
            errors.append("conveyor pick out of reach")
    for slot in pallet_slot_positions(cell):
        if vnorm(slot) > reach + 1e-9:
            errors.append(f"pallet slot {slot} out of reach")
    return errors


def default_cell_2r():
    """Return the demo 2R planar workcell.

    A planar 2R arm with one circular column obstacle, a conveyor pick at
    ``(0.55, 0.15, 0.0)`` and a 2x3 pallet starting at ``(0.15, 0.55, 0.0)``.
    All pick/place points are within the 0.9 m reach and clear of the
    obstacle.  The column sits at ``(-0.55, 0.45)``, off to the side of the
    arm's swept area: an earlier placement at ``(0.45, 0.55)`` lay almost
    on the end-effector transit path (mid-transit EE near ``(0.42, 0.52)``)
    and drove the minimum clearance negative, and ``(-0.35, 0.45)`` left
    only 0.005 m because the elbow passes there at the place pose, so it
    was moved further out.
    """
    return Workcell(
        name="demo-cell-2r",
        arm=planar_2r(),
        obstacles=[CircleObstacle(center=(-0.55, 0.45), radius=0.08,
                                  name="column")],
        conveyor=Conveyor(pick=(0.55, 0.15, 0.0), approach_height=0.12),
        pallet=Pallet(origin=(0.15, 0.55, 0.0), rows=2, cols=3,
                      dx=0.12, dy=0.12),
    )


def default_cell_6dof():
    """Return the demo 6-DOF (UR5) workcell.

    A UR5 arm with one spherical fixture-post obstacle, a conveyor pick at
    ``(0.45, -0.25, 0.05)`` and a 2x3 pallet starting at
    ``(0.30, 0.30, 0.05)``.  All pick/place points are within the
    1.19251 m reach.  The fixture post sits at ``(-0.5, 0.5, 0.3)`` in
    the back-left corner of the cell: an earlier placement at
    ``(0.35, -0.35, 0.35)`` lay inside the arm's swept volume and the
    plan grazed it (minimum clearance -0.12 m), so it was moved to a
    spot verified clear over the whole plan (minimum clearance 0.40 m).
    """
    return Workcell(
        name="demo-cell-6dof",
        arm=ur5_arm(),
        obstacles=[SphereObstacle(center=(-0.5, 0.5, 0.3), radius=0.10,
                                  name="fixture-post")],
        conveyor=Conveyor(pick=(0.45, -0.25, 0.05), approach_height=0.12),
        pallet=Pallet(origin=(0.30, 0.30, 0.05), rows=2, cols=3,
                      dx=0.12, dy=0.12),
    )
