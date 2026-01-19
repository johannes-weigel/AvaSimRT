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