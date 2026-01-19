from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from avasimrt.config import AnchorConfig, NodeConfig
from .result import ResolvedPosition

logger = logging.getLogger(__name__)

def _raycast_z_at_position(
    mesh: trimesh.Trimesh,
    x: float,
    y: float,
) -> float | None:
    bounds = mesh.bounds
    z_origin = bounds[1, 2] + 10.0 

    origins = np.array([[x, y, z_origin]])
    directions = np.array([[0.0, 0.0, -1.0]])

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
    )

    if len(locations) == 0:
        return None

    return float(np.max(locations[:, 2]))

def _resolve_position(mesh, x, y, z) -> float:
    "Returns final z"

    if (z is not None):
        return z
    
    z_terrain = _raycast_z_at_position(mesh, x, y)
    if z_terrain is None:
        raise ValueError(f"Could not resolve z at ({x}, {y}): no terrain intersection")
    
    return z_terrain


def resolve_positions(
    mesh_path: Path,
    node: NodeConfig,
    anchors: Sequence[AnchorConfig],
) -> tuple[ResolvedPosition, list[ResolvedPosition]]:
    logger.info(f"Loading mesh for z-resolution: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected single mesh, got {type(mesh)}")

    node_z_final = _resolve_position(mesh, node.x, node.y, node.z)

    resolved_node = ResolvedPosition(
        id="NODE",
        x=node.x,
        y=node.y,
        z=node_z_final,
        size=node.size,
    )

    resolved_anchors = []
    for anchor in anchors:
        anchor_z_final = _resolve_position(mesh, anchor.x, anchor.y, anchor.z)
        resolved_anchors.append(ResolvedPosition(
            id=anchor.id,
            x=anchor.x,
            y=anchor.y,
            z=anchor_z_final,
            size=anchor.size,
        ))

    return resolved_node, resolved_anchors


def generate_heightmap(
    mesh_path: Path,
    resolution: float,
) -> tuple[np.ndarray, dict]:
    logger.info(f"Loading mesh from {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected single mesh, got {type(mesh)}")

    bounds = mesh.bounds
    min_pt, max_pt = bounds[0], bounds[1]

    x_range = max_pt[0] - min_pt[0]
    y_range = max_pt[1] - min_pt[1]

    n_x = int(np.ceil(x_range / resolution)) + 1
    n_y = int(np.ceil(y_range / resolution)) + 1

    logger.info(f"Generating heightmap: {n_x}x{n_y} grid, resolution={resolution}m")
    logger.info(f"Mesh bounds: x=[{min_pt[0]:.2f}, {max_pt[0]:.2f}], y=[{min_pt[1]:.2f}, {max_pt[1]:.2f}]")

    x_coords = np.linspace(min_pt[0], max_pt[0], n_x)
    y_coords = np.linspace(min_pt[1], max_pt[1], n_y)

    xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')

    z_origin = max_pt[2] + 10.0
    origins = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        np.full(xx.size, z_origin),
    ])

    directions = np.tile([0.0, 0.0, -1.0], (origins.shape[0], 1))

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
    )

    heightmap = np.full((n_x, n_y), np.nan, dtype=np.float32)

    if len(locations) > 0:
        for ray_idx, hit_loc in zip(index_ray, locations):
            i = ray_idx // n_y
            j = ray_idx % n_y
            z = hit_loc[2]
            if np.isnan(heightmap[i, j]) or z > heightmap[i, j]:
                heightmap[i, j] = z

    metadata = {
        'x_min': float(min_pt[0]),
        'x_max': float(max_pt[0]),
        'y_min': float(min_pt[1]),
        'y_max': float(max_pt[1]),
        'z_min': float(np.nanmin(heightmap)) if not np.all(np.isnan(heightmap)) else None,
        'z_max': float(np.nanmax(heightmap)) if not np.all(np.isnan(heightmap)) else None,
        'resolution': resolution,
        'shape': list(heightmap.shape),
        'x_coords': x_coords.tolist(),
        'y_coords': y_coords.tolist(),
    }

    valid_cells = np.count_nonzero(~np.isnan(heightmap))
    logger.info(f"Heightmap complete: {valid_cells}/{heightmap.size} cells have data")

    return heightmap, metadata
