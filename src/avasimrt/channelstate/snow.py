from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Sequence

import mitsuba as mi
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from sionna.rt import ITURadioMaterial, Scene, SceneObject
from sionna.rt.utils.meshes import clone_mesh, load_mesh, transform_mesh
import sionna.rt.scene as sionna_scenes

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition

if TYPE_CHECKING:
    from avasimrt.channelstate.config import SnowConfig

logger = logging.getLogger(__name__)


def _build_alpha_shape(points: np.ndarray, alpha: float, margin: float) -> Polygon:
    start = time.perf_counter()
    logger.info("Building alpha shape from %d points (alpha=%.4f, margin=%.1f)", len(points), alpha, margin)

    if len(points) < 3:
        shape = MultiPoint(points).convex_hull.buffer(margin)
        logger.info("Alpha shape built (convex hull fallback) in %.3fs", time.perf_counter() - start)
        return shape

    tri = Delaunay(points)
    triangles = points[tri.simplices]
    logger.debug("Delaunay triangulation: %d triangles", len(tri.simplices))

    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    circumradius = np.where(area > 0, (a * b * c) / (4 * area), np.inf)

    valid = circumradius < (1.0 / alpha) if alpha > 0 else np.ones(len(circumradius), dtype=bool)
    valid_triangles = tri.simplices[valid]
    logger.debug("Valid triangles after alpha filter: %d", len(valid_triangles))

    if len(valid_triangles) == 0:
        shape = MultiPoint(points).convex_hull.buffer(margin)
        logger.info("Alpha shape built (convex hull fallback) in %.3fs", time.perf_counter() - start)
        return shape

    polygons = [Polygon(points[triangle]) for triangle in valid_triangles]
    shape = unary_union(polygons).buffer(margin)
    logger.info("Alpha shape built in %.3fs", time.perf_counter() - start)
    return shape


def _create_mask_from_shape(shape: Polygon, coords_x: np.ndarray, coords_y: np.ndarray) -> np.ndarray:
    start = time.perf_counter()
    n_x, n_y = len(coords_x), len(coords_y)
    total_points = n_x * n_y
    logger.info("Creating mask for %d x %d = %d grid points", n_x, n_y, total_points)

    mask = np.zeros((n_x, n_y), dtype=bool)
    log_interval = max(1, n_x // 10)

    for i, x in enumerate(coords_x):
        if i % log_interval == 0:
            progress = (i / n_x) * 100
            elapsed = time.perf_counter() - start
            if i > 0:
                eta = elapsed / i * (n_x - i)
                logger.info("Mask progress: %.0f%% (%d/%d rows, ETA: %.1fs)", progress, i, n_x, eta)
            else:
                logger.info("Mask progress: %.0f%% (%d/%d rows)", progress, i, n_x)
        for j, y in enumerate(coords_y):
            mask[i, j] = shape.contains(Point(x, y))

    elapsed = time.perf_counter() - start
    true_count = np.sum(mask)
    logger.info("Mask created in %.3fs: %d/%d points inside shape (%.1f%%)",
                elapsed, true_count, total_points, 100 * true_count / total_points)
    return mask


def _extract_points(
    anchors: Sequence[ResolvedPosition],
    nodes: Sequence[Sequence[NodeSnapshot]],
) -> np.ndarray:
    points: list[tuple[float, float]] = [(a.x, a.y) for a in anchors]
    for trajectory in nodes:
        for snapshot in trajectory:
            points.append((snapshot.position[0], snapshot.position[1]))
    return np.array(points)


def extract_relevant_heightmap(
    heightmap: np.ndarray,
    anchors: Sequence[ResolvedPosition],
    nodes: Sequence[Sequence[NodeSnapshot]],
    alpha: float = 0.05,
    margin: float = 10.0,
    quick: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    start = time.perf_counter()
    logger.info("Extracting relevant heightmap: shape=%s, anchors=%d, node_trajectories=%d, quick=%s",
                heightmap.shape, len(anchors), len(nodes), quick)

    points = _extract_points(anchors, nodes)

    coords_x = heightmap[:, 0, 0]
    coords_y = heightmap[0, :, 1]

    if quick:
        x_min, y_min = points.min(axis=0) - margin
        x_max, y_max = points.max(axis=0) + margin

        x_mask = (coords_x >= x_min) & (coords_x <= x_max)
        y_mask = (coords_y >= y_min) & (coords_y <= y_max)

        sliced_heightmap = heightmap[x_mask][:, y_mask]
        filtered_heightmap = sliced_heightmap.reshape(-1, 3)

        mask = np.outer(x_mask, y_mask)

        logger.info("Quick extraction complete in %.3fs: %d points extracted (bbox: x=[%.1f, %.1f], y=[%.1f, %.1f])",
                    time.perf_counter() - start, len(filtered_heightmap), x_min, x_max, y_min, y_max)
    else:
        shape = _build_alpha_shape(points, alpha, margin)
        mask = _create_mask_from_shape(shape, coords_x, coords_y)
        filtered_heightmap = heightmap[mask]

        logger.info("Extraction complete in %.3fs: %d points extracted", time.perf_counter() - start, len(filtered_heightmap))

    return filtered_heightmap, mask


def _create_snow_material(cfg: "SnowConfig") -> ITURadioMaterial:
    return ITURadioMaterial(
        name="avasimrt_snow",
        itu_type=cfg.material.itu_type,
        thickness=cfg.material.thickness,
        scattering_coefficient=cfg.material.scattering_coefficient,
    )


def _create_box_mesh(
    name: str,
    position: tuple[float, float, float],
    size: float,
    material: ITURadioMaterial,
) -> SceneObject:
    base_mesh = load_mesh(sionna_scenes.sphere)
    mesh = clone_mesh(base_mesh, name=name)

    scale = mi.Point3f(size, size, size)
    translation = mi.Point3f(position[0], position[1], position[2])
    transform_mesh(mesh, translation=translation, scale=scale)

    return SceneObject(mi_mesh=mesh, name=name, radio_material=material)


def add_snow_to_scene(
    scene: Scene,
    cfg: "SnowConfig",
    heightmap: np.ndarray,
    anchors: Sequence[ResolvedPosition],
    nodes: Sequence[Sequence[NodeSnapshot]],
) -> int:
    """
    Add snow boxes to the Sionna scene at relevant heightmap positions.

    Args:
        scene: Loaded Sionna scene to add snow to.
        cfg: Snow configuration containing material and placement settings.
        heightmap: 3D array of shape (n_x, n_y, 3) with (x, y, z) coordinates.
        anchors: Sequence of anchor positions.
        nodes: Sequence of node trajectories (each trajectory is a sequence of snapshots).

    Returns:
        Number of snow boxes added to the scene.
    """
    if not cfg.enabled:
        logger.info("Snow disabled, skipping")
        return 0

    start = time.perf_counter()
    logger.info(
        "Adding snow to scene: box_size=%.2f, levels=%d, margin=%.1f",
        cfg.box_size,
        cfg.levels,
        cfg.margin,
    )

    relevant_positions, _ = extract_relevant_heightmap(
        heightmap=heightmap,
        anchors=anchors,
        nodes=nodes,
        margin=cfg.margin,
    )

    if len(relevant_positions) == 0:
        logger.warning("No relevant heightmap positions found for snow placement")
        return 0

    snow_material = _create_snow_material(cfg)
    scene.add(snow_material)

    snow_objects: list[SceneObject] = []
    box_idx = 0

    for pos in relevant_positions:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        for level in range(cfg.levels):
            z_offset = z + (level + 0.5) * cfg.box_size
            name = f"snow_box_{box_idx}"
            obj = _create_box_mesh(
                name=name,
                position=(x, y, z_offset),
                size=cfg.box_size,
                material=snow_material,
            )
            snow_objects.append(obj)
            box_idx += 1

    if snow_objects:
        scene.edit(add=snow_objects)
        logger.info(
            "Added %d snow boxes to scene in %.3fs",
            len(snow_objects),
            time.perf_counter() - start,
        )

    return len(snow_objects)

