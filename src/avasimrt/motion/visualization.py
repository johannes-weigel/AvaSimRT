"""Trajectory visualization: 3D paths, 2D top-down, position over time."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from avasimrt.result import Sample
from avasimrt.preprocessing.result import ResolvedPosition

logger = logging.getLogger(__name__)


def _extract_arrays(samples: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract timestamps and position arrays from samples."""
    n = len(samples)
    timestamps = np.empty(n, dtype=np.float64)
    x = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    z = np.empty(n, dtype=np.float64)

    for i, s in enumerate(samples):
        timestamps[i] = s.timestamp
        x[i], y[i], z[i] = s.node.position

    return timestamps, x, y, z


def plot_trajectory_3d(
    samples: list[Sample],
    node_id: str,
    *,
    show: bool = False,
    save_path: Path | None = None,
    terrain_obj_path: Path | None = None,
    anchors: Sequence[ResolvedPosition],
) -> Figure | None:
    """Plot 3D trajectory path."""
    if not samples:
        logger.warning("No samples for node '%s', skipping 3D plot", node_id)
        return None

    _, x, y, z = _extract_arrays(samples)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Render terrain mesh if provided
    if terrain_obj_path is not None and terrain_obj_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(terrain_obj_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices
                ax.plot_trisurf(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    triangles=mesh.faces,
                    alpha=0.3,
                    cmap='terrain',
                    edgecolor='none',
                )
        except Exception as e:
            logger.warning("Failed to load terrain mesh from %s: %s", terrain_obj_path, e)

    # Plot anchors as static points
    if anchors:
        anchor_x = [a.x for a in anchors]
        anchor_y = [a.y for a in anchors]
        anchor_z = [a.z for a in anchors]
        anchor_id = [a.id for a in anchors]
        ax.scatter(anchor_x, anchor_y, anchor_z, color="blue", s=150, marker="^", label=anchor_id, zorder=10) # type: ignore

    ax.plot(x, y, z, linewidth=1.5, label=node_id)
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=100, marker="o", label="Start") # type: ignore
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=100, marker="x", label="End") # type: ignore

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(f"Trajectory: {node_id}")
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved 3D trajectory plot: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_trajectory_2d_topdown(
    samples: list[Sample],
    node_id: str,
    *,
    show: bool = False,
    save_path: Path | None = None,
) -> Figure | None:
    """Plot 2D top-down (X/Y) trajectory."""
    if not samples:
        logger.warning("No samples for node '%s', skipping 2D plot", node_id)
        return None

    _, x, y, _ = _extract_arrays(samples)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(x, y, linewidth=1.5, label=node_id)
    ax.scatter([x[0]], [y[0]], color="green", s=100, marker="o", label="Start", zorder=5)
    ax.scatter([x[-1]], [y[-1]], color="red", s=100, marker="x", label="End", zorder=5)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Trajectory (top-down): {node_id}")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved 2D trajectory plot: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_position_over_time(
    samples: list[Sample],
    node_id: str,
    *,
    show: bool = False,
    save_path: Path | None = None,
) -> Figure | None:
    """Plot X, Y, Z position components over time."""
    if not samples:
        logger.warning("No samples for node '%s', skipping position/time plot", node_id)
        return None

    t, x, y, z = _extract_arrays(samples)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(t, x, linewidth=1.5, color="tab:blue")
    axes[0].set_ylabel("X [m]")
    axes[0].set_title(f"Position over time: {node_id}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, y, linewidth=1.5, color="tab:orange")
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, z, linewidth=1.5, color="tab:green")
    axes[2].set_ylabel("Z [m]")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved position/time plot: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all_trajectories_3d(
    trajectories: dict[str, list[Sample]],
    *,
    show: bool = False,
    save_path: Path | None = None,
    terrain_obj_path: Path | None = None,
    anchors: Sequence[ResolvedPosition]
) -> Figure | None:
    """Plot all trajectories in a single 3D figure."""
    if not trajectories:
        logger.warning("No trajectories to plot")
        return None

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Render terrain mesh if provided
    if terrain_obj_path is not None and terrain_obj_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(terrain_obj_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices
                ax.plot_trisurf(
                    vertices[:, 0],
                    vertices[:, 1],
                    vertices[:, 2],
                    triangles=mesh.faces,
                    alpha=0.3,
                    cmap='terrain',
                    edgecolor='none',
                )
        except Exception as e:
            logger.warning("Failed to load terrain mesh from %s: %s", terrain_obj_path, e)

    # Plot anchors as static points
    if anchors:
        anchor_x = [a.x for a in anchors]
        anchor_y = [a.y for a in anchors]
        anchor_z = [a.z for a in anchors]
        ax.scatter(anchor_x, anchor_y, anchor_z, color="blue", s=150, marker="^", label="Anchors", zorder=10) # type: ignore

    for node_id, samples in trajectories.items():
        if not samples:
            continue
        _, x, y, z = _extract_arrays(samples)
        ax.plot(x, y, z, linewidth=1.5, label=node_id)
        ax.scatter([x[0]], [y[0]], [z[0]], s=50, marker="o") # type: ignore
        ax.scatter([x[-1]], [y[-1]], [z[-1]], s=50, marker="x") # type: ignore

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("All Trajectories")
    ax.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved combined 3D trajectory plot: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_all_trajectories_2d_topdown(
    trajectories: dict[str, list[Sample]],
    *,
    show: bool = False,
    save_path: Path | None = None,
) -> Figure | None:
    """Plot all trajectories in a single 2D top-down figure."""
    if not trajectories:
        logger.warning("No trajectories to plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 10))

    for node_id, samples in trajectories.items():
        if not samples:
            continue
        _, x, y, _ = _extract_arrays(samples)
        ax.plot(x, y, linewidth=1.5, label=node_id)
        ax.scatter([x[0]], [y[0]], s=50, marker="o")
        ax.scatter([x[-1]], [y[-1]], s=50, marker="x")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("All Trajectories (top-down)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved combined 2D trajectory plot: %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# --- Interactive HTML (Plotly) ---

def _has_plotly() -> bool:
    try:
        import plotly
        return True
    except ImportError:
        return False


def plot_trajectory_3d_html(
    samples: list[Sample],
    node_id: str,
    save_path: Path,
    anchors: Sequence[ResolvedPosition],
    terrain_obj_path: Path | None = None,
) -> None:
    """Save interactive 3D trajectory as HTML using Plotly."""
    if not _has_plotly():
        logger.warning("Plotly not installed, skipping HTML export for %s", node_id)
        return

    import plotly.graph_objects as go

    if not samples:
        logger.warning("No samples for node '%s', skipping HTML plot", node_id)
        return

    _, x, y, z = _extract_arrays(samples)

    fig = go.Figure()

    # Add terrain mesh if provided
    if terrain_obj_path is not None and terrain_obj_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(terrain_obj_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices
                faces = mesh.faces
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.3,
                    color='lightgray',
                    name='Terrain',
                ))
        except Exception as e:
            logger.warning("Failed to load terrain mesh from %s: %s", terrain_obj_path, e)

    # Add anchors as static points
    if anchors:
        anchor_x = [a.x for a in anchors]
        anchor_y = [a.y for a in anchors]
        anchor_z = [a.z for a in anchors]
        fig.add_trace(go.Scatter3d(
            x=anchor_x, y=anchor_y, z=anchor_z,
            mode="markers",
            name="Anchors",
            marker=dict(size=10, color="blue", symbol="diamond"),
        ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="lines",
        name=node_id,
        line=dict(width=3),
    ))

    fig.add_trace(go.Scatter3d(
        x=[x[0]], y=[y[0]], z=[z[0]],
        mode="markers",
        name="Start",
        marker=dict(size=8, color="green", symbol="circle"),
    ))

    fig.add_trace(go.Scatter3d(
        x=[x[-1]], y=[y[-1]], z=[z[-1]],
        mode="markers",
        name="End",
        marker=dict(size=8, color="red", symbol="x"),
    ))

    fig.update_layout(
        title=f"Trajectory: {node_id}",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
        ),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)
    logger.info("Saved interactive 3D trajectory: %s", save_path)


def plot_all_trajectories_3d_html(
    trajectories: dict[str, list[Sample]],
    save_path: Path,
    anchors: Sequence[ResolvedPosition],
    terrain_obj_path: Path | None = None,
) -> None:
    """Save all trajectories as interactive 3D HTML."""
    if not _has_plotly():
        logger.warning("Plotly not installed, skipping combined HTML export")
        return

    import plotly.graph_objects as go

    if not trajectories:
        logger.warning("No trajectories to plot")
        return

    fig = go.Figure()

    # Add terrain mesh if provided
    if terrain_obj_path is not None and terrain_obj_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(terrain_obj_path, force='mesh')
            if isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices
                faces = mesh.faces
                fig.add_trace(go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.3,
                    color='lightgray',
                    name='Terrain',
                ))
        except Exception as e:
            logger.warning("Failed to load terrain mesh from %s: %s", terrain_obj_path, e)

    # Add anchors as static points
    if anchors:
        anchor_x = [a.x for a in anchors]
        anchor_y = [a.y for a in anchors]
        anchor_z = [a.z for a in anchors]
        fig.add_trace(go.Scatter3d(
            x=anchor_x, y=anchor_y, z=anchor_z,
            mode="markers",
            name="Anchors",
            marker=dict(size=10, color="blue", symbol="diamond"),
        ))

    for node_id, samples in trajectories.items():
        if not samples:
            continue
        _, x, y, z = _extract_arrays(samples)

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="lines",
            name=node_id,
            line=dict(width=3),
        ))

    fig.update_layout(
        title="All Trajectories",
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
        ),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(save_path)
    logger.info("Saved combined interactive 3D trajectory: %s", save_path)


# --- High-level save functions ---

def save_trajectory_visualizations(
    node_id: str,
    samples: list[Sample],
    out_dir: Path,
    *,
    png: bool = True,
    html: bool = True,
    terrain_obj_path: Path | None = None,
    anchors: Sequence[ResolvedPosition],
) -> None:
    """Save all visualization types for a single node."""
    if png:
        plot_trajectory_3d(samples, node_id, 
                           save_path=out_dir / f"{node_id}_3d.png",
                           terrain_obj_path=terrain_obj_path, 
                           anchors=anchors)
        plot_trajectory_2d_topdown(samples, node_id, 
                                   save_path=out_dir / f"{node_id}_2d_topdown.png")
        plot_position_over_time(samples, node_id, 
                                save_path=out_dir / f"{node_id}_position_time.png")

    if html:
        plot_trajectory_3d_html(samples, node_id, 
                                out_dir / f"{node_id}_3d.html", 
                                terrain_obj_path=terrain_obj_path, 
                                anchors=anchors)


def save_all_trajectory_visualizations(
    trajectories: dict[str, list[Sample]],
    out_dir: Path,
    *,
    png: bool = True,
    html: bool = True,
    terrain_obj_path: Path | None = None,
    anchors: Sequence[ResolvedPosition],
) -> None:
    """Save visualizations for all trajectories (combined and individual)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combined plots
    if png:
        plot_all_trajectories_3d(trajectories, 
                                 save_path=out_dir / "all_trajectories_3d.png", 
                                 terrain_obj_path=terrain_obj_path, 
                                 anchors=anchors)
        plot_all_trajectories_2d_topdown(trajectories, 
                                         save_path=out_dir / "all_trajectories_2d_topdown.png")

    if html:
        plot_all_trajectories_3d_html(trajectories, 
                                      out_dir / "all_trajectories_3d.html", 
                                      terrain_obj_path=terrain_obj_path,
                                      anchors=anchors)

    # Individual node plots
    for node_id, samples in trajectories.items():
        save_trajectory_visualizations(node_id, samples, out_dir, 
                                       png=png, html=html, 
                                       terrain_obj_path=terrain_obj_path, 
                                       anchors=anchors)

    logger.info("Saved all trajectory visualizations to %s", out_dir)
