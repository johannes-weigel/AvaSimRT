from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import trimesh

from avasimrt.config import PositionConfig
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

def _resolve_position(mesh, x, y, z, z_offset) -> tuple[float, float | None]:
    "Returns final z"

    if (z is not None):
        return z, None
    
    z_terrain = _raycast_z_at_position(mesh, x, y)
    if z_terrain is None:
        raise ValueError(f"Could not resolve z at ({x}, {y}): no terrain intersection")
    
    if (z_offset is None):
        return z_terrain, z_terrain
    else:
        return z_terrain + z_offset, z_terrain

def _resolve_positions(mesh, positions: Sequence[PositionConfig]) -> list[ResolvedPosition]:
    return [
        ResolvedPosition.from_config(p, *_resolve_position(mesh, p.x, p.y, p.z, p.z_offset))
        for p in positions
    ]

def resolve_positions(
    mesh_path: Path,
    nodes: Sequence[PositionConfig],
    anchors: Sequence[PositionConfig],
) -> tuple[list[ResolvedPosition], list[ResolvedPosition]]:
    logger.info(f"Loading mesh for z-resolution: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected single mesh, got {type(mesh)}")

    return _resolve_positions(mesh, nodes), _resolve_positions(mesh, anchors)


def generate_heightmap(
    mesh_path: Path,
    resolution: float | None = None,
) -> tuple[np.ndarray, dict]:
    if resolution is None:
        resolution = 5    
    start_time = time.perf_counter()
    
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

    heightmap_3d = np.zeros((n_x, n_y, 3), dtype=np.float32)
    for i in range(n_x):
        for j in range(n_y):
            heightmap_3d[i, j, 0] = x_coords[i]
            heightmap_3d[i, j, 1] = y_coords[j]
            heightmap_3d[i, j, 2] = heightmap[i, j]
    
    z_min = float(np.nanmin(heightmap)) if not np.all(np.isnan(heightmap)) else None
    z_max = float(np.nanmax(heightmap)) if not np.all(np.isnan(heightmap)) else None
    z_range = float(z_max - z_min) if z_min is not None and z_max is not None else None
    
    total_positions = n_x * n_y
    valid_positions = int(np.count_nonzero(~np.isnan(heightmap)))
    coverage_percent = round((valid_positions / total_positions) * 100, 2)
    
    file_size_bytes = int(heightmap_3d.nbytes)
    file_size_kb = round(file_size_bytes / 1024, 2)
    file_size_mb = round(file_size_kb / 1024, 4)
    
    elapsed_time = time.perf_counter() - start_time
    
    metadata = {
        'resolution_m': resolution,
        'computation_time_s': round(elapsed_time, 4),
        'bounds': {
            'x_min': float(min_pt[0]),
            'x_max': float(max_pt[0]),
            'y_min': float(min_pt[1]),
            'y_max': float(max_pt[1]),
        },
        'grid_size': {
            'n_x': int(n_x),
            'n_y': int(n_y),
            'total_positions': int(total_positions),
        },
        'heightmap_stats': {
            'z_min': z_min,
            'z_max': z_max,
            'z_range': z_range,
        },
        'coverage': {
            'valid_positions': valid_positions,
            'coverage_percent': coverage_percent,
        },
        'memory': {
            'bytes': file_size_bytes,
            'kilobytes': file_size_kb,
            'megabytes': file_size_mb,
        },
    }

    logger.info(f"Heightmap complete: {valid_positions}/{total_positions} cells have data ({coverage_percent}%)")
    logger.info(f"Computation time: {elapsed_time:.2f}s")

    return heightmap_3d, metadata
