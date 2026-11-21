"""Visualisation for ``robot-arm-workcell-sim``.

Matplotlib (``Agg`` backend, no display needed) draws the 2D workcell
trajectory, the joint time-series, and — when the ``mpl_toolkits.mplot3d``
toolkit imports cleanly — the 3D end-effector path of the 6-DOF arm.  If
mplot3d is broken (this VM ships a Debian-packaged ``mpl_toolkits`` that
fails to import, so the ``"3d"`` projection is never registered),
:func:`plot_6dof_path` falls back to three 2D projections.  A pure-stdlib
SVG fallback (:func:`svg_planar_trajectory`) covers planar cells when
matplotlib is not installed at all, so the core package stays usable
without any third-party dependency.  Importing this module never requires
matplotlib; the import happens lazily inside the plot functions.
"""

from __future__ import annotations

import math

MATPLOTLIB_AVAILABLE = False
try:  # pragma: no cover - environment dependent
    import matplotlib  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


def _require_matplotlib():
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "matplotlib is not installed; install it (pip install matplotlib) "
            "or use svg_planar_trajectory() for planar cells"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_planar_trajectory(cell, ee_path, joint_configs, out_path, pick=None, place=None):
    """Plot a planar workcell: arm poses, EE path, obstacles, pick/place.

    Args:
        cell: the workcell (planar arm, circle obstacles).
        ee_path: list of ``(x, y, z)`` end-effector positions.
        joint_configs: list of joint-position lists ``[(x, y, z), ...]``,
            drawn as arm sticks at sampled instants.
        out_path: PNG file to write.
        pick: optional ``(x, y, z)`` pick marker.
        place: optional ``(x, y, z)`` place marker.
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    # Obstacles.
    for obs in cell.obstacles:
        circle = plt.Circle(
            obs.center, obs.radius, color="#c0392b", alpha=0.35,
            label=f"obstacle {obs.name}" if obs.name else "obstacle",
        )
        ax.add_patch(circle)
        ax.plot(obs.center[0], obs.center[1], "rx", ms=6)
    # End-effector path.
    xs = [p[0] for p in ee_path]
    ys = [p[1] for p in ee_path]
    ax.plot(xs, ys, "-", color="#2e86c1", lw=1.5, label="end-effector path")
    ax.plot(xs[0], ys[0], "go", ms=7, label="start")
    ax.plot(xs[-1], ys[-1], "rs", ms=7, label="end")
    # Arm configurations (a subset, fading).
    n = len(joint_configs)
    for i, joints in enumerate(joint_configs):
        alpha = 0.25 + 0.65 * (i / max(1, n - 1))
        ax.plot(
            [j[0] for j in joints], [j[1] for j in joints],
            "-o", color="#1d2a32", alpha=alpha, ms=3, lw=1.2,
        )
    if pick is not None:
        ax.plot(pick[0], pick[1], "D", color="#20a145", ms=8, label="pick")
    if place is not None:
        ax.plot(place[0], place[1], "s", color="#e67e22", ms=8, label="place")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Planar trajectory — {cell.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_joint_trajectories(times, q_series, qd_series, joint_names, out_path):
    """Plot joint angles and joint velocities vs. time (two panels)."""
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for series, name in zip(q_series, joint_names):
        axes[0].plot(times, [math.degrees(v) for v in series], label=name)
    axes[0].set_ylabel("joint angle [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    for series, name in zip(qd_series, joint_names):
        axes[1].plot(times, [math.degrees(v) for v in series], label=name)
    axes[1].set_ylabel("joint velocity [deg/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Joint trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_6dof_path(cell, ee_path, joint_configs, out_path):
    """Plot the 3D end-effector path and arm sticks of a 6-DOF arm.

    When the ``mpl_toolkits.mplot3d`` import fails (e.g. this VM ships a
    broken Debian-packaged ``mpl_toolkits`` alongside a working pip
    matplotlib, so the ``"3d"`` projection is never registered), the plot
    falls back to three 2D projections (top/front/side) drawn with plain
    matplotlib instead of failing the whole simulation run.
    """
    plt = _require_matplotlib()
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)
    except ImportError:
        return _plot_6dof_projections(cell, ee_path, joint_configs, out_path)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs = [p[0] for p in ee_path]
    ys = [p[1] for p in ee_path]
    zs = [p[2] for p in ee_path]
    ax.plot(xs, ys, zs, "-", color="#2e86c1", lw=1.5, label="end-effector path")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c="green", s=40, label="start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="red", s=40, label="end")
    n = len(joint_configs)
    for i, joints in enumerate(joint_configs):
        alpha = 0.25 + 0.65 * (i / max(1, n - 1))
        ax.plot(
            [j[0] for j in joints], [j[1] for j in joints], [j[2] for j in joints],
            "-o", color="#1d2a32", alpha=alpha, ms=3, lw=1.0,
        )
    for obs in cell.obstacles:
        ax.scatter(
            [obs.center[0]], [obs.center[1]], [obs.center[2]],
            c="red", marker="x", s=60,
        )
        ax.text(obs.center[0], obs.center[1], obs.center[2],
                f" {obs.name or 'obstacle'} r={obs.radius:.2f}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"6-DOF end-effector path — {cell.name}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_6dof_projections(cell, ee_path, joint_configs, out_path):
    """Fallback for :func:`plot_6dof_path` when mplot3d is unavailable.

    Draws the end-effector path and arm sticks as three plain-2D
    projections (top x-y, front x-z, side y-z) so a simulation still gets
    a usable trajectory figure without the 3D toolkit.
    """
    plt = _require_matplotlib()
    views = [
        ("top view (x-y)", 0, 1, "x [m]", "y [m]"),
        ("front view (x-z)", 0, 2, "x [m]", "z [m]"),
        ("side view (y-z)", 1, 2, "y [m]", "z [m]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, i, j, xlabel, ylabel) in zip(axes, views):
        ax.plot(
            [p[i] for p in ee_path], [p[j] for p in ee_path],
            "-", color="#2e86c1", lw=1.5, label="end-effector path",
        )
        ax.plot(ee_path[0][i], ee_path[0][j], "go", ms=6, label="start")
        ax.plot(ee_path[-1][i], ee_path[-1][j], "rs", ms=6, label="end")
        n = len(joint_configs)
        for k, joints in enumerate(joint_configs):
            alpha = 0.25 + 0.65 * (k / max(1, n - 1))
            ax.plot(
                [pt[i] for pt in joints], [pt[j] for pt in joints],
                "-o", color="#1d2a32", alpha=alpha, ms=3, lw=1.0,
            )
        for obs in cell.obstacles:
            ax.plot(
                obs.center[i], obs.center[j], "rx", ms=8,
                label=f"obstacle {obs.name}" if obs.name else "obstacle",
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"6-DOF end-effector path (2D projections) — {cell.name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def svg_planar_trajectory(cell, ee_path, joint_configs, out_path, pick=None, place=None):
    """Draw the planar trajectory as a standalone SVG (no dependencies).

    Pure-stdlib fallback used when matplotlib is unavailable.  The drawing
    uses a 700x600 viewBox with a fixed scale and a y-flip so that +y
    points up on screen.
    """
    all_x = [p[0] for p in ee_path] + [0.0]
    all_y = [p[1] for p in ee_path] + [0.0]
    for joints in joint_configs:
        all_x += [j[0] for j in joints]
        all_y += [j[1] for j in joints]
    for obs in cell.obstacles:
        all_x += [obs.center[0] - obs.radius, obs.center[0] + obs.radius]
        all_y += [obs.center[1] - obs.radius, obs.center[1] + obs.radius]
    xmin, xmax = min(all_x) - 0.1, max(all_x) + 0.1
    ymin, ymax = min(all_y) - 0.1, max(all_y) + 0.1
    width, height = 700, 600
    sx = width / max(1e-9, xmax - xmin)
    sy = height / max(1e-9, ymax - ymin)
    scale = min(sx, sy)

    def px(x):
        return (x - xmin) * scale + 20

    def py(y):
        return height - ((y - ymin) * scale + 20)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 40}" '
        f'height="{height + 40}" viewBox="0 0 {width + 40} {height + 40}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="24" font-family="sans-serif" font-size="16">'
        f"Planar trajectory — {cell.name}</text>",
    ]
    for obs in cell.obstacles:
        parts.append(
            f'<circle cx="{px(obs.center[0]):.1f}" cy="{py(obs.center[1]):.1f}" '
            f'r="{obs.radius * scale:.1f}" fill="#c0392b" fill-opacity="0.35" '
            f'stroke="#c0392b"/>'
        )
    path_d = " ".join(
        f"{'M' if i == 0 else 'L'}{px(p[0]):.1f},{py(p[1]):.1f}"
        for i, p in enumerate(ee_path)
    )
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="#2e86c1" stroke-width="2"/>'
    )
    for joints in joint_configs:
        line = " ".join(
            f"{'M' if i == 0 else 'L'}{px(j[0]):.1f},{py(j[1]):.1f}"
            for i, j in enumerate(joints)
        )
        parts.append(
            f'<path d="{line}" fill="none" stroke="#1d2a32" '
            f'stroke-width="1.5" stroke-opacity="0.5"/>'
        )
        for j in joints:
            parts.append(
                f'<circle cx="{px(j[0]):.1f}" cy="{py(j[1]):.1f}" r="3" '
                f'fill="#1d2a32" fill-opacity="0.5"/>'
            )
    for point, color, label in ((pick, "#20a145", "pick"), (place, "#e67e22", "place")):
        if point is not None:
            parts.append(
                f'<rect x="{px(point[0]) - 5:.1f}" y="{py(point[1]) - 5:.1f}" '
                f'width="10" height="10" fill="{color}"/>'
                f'<text x="{px(point[0]) + 8:.1f}" y="{py(point[1]) - 8:.1f}" '
                f'font-family="sans-serif" font-size="12">{label}</text>'
            )
    parts.append("</svg>")
    with open(out_path, "w") as fh:
        fh.write("\n".join(parts))
    return out_path
