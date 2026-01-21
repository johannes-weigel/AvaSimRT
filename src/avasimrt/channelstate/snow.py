from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition

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
) -> tuple[np.ndarray, np.ndarray]:
    start = time.perf_counter()
    logger.info("Extracting relevant heightmap: shape=%s, anchors=%d, node_trajectories=%d",
                heightmap.shape, len(anchors), len(nodes))

    points = _extract_points(anchors, nodes)
    shape = _build_alpha_shape(points, alpha, margin)

    coords_x = heightmap[:, 0, 0]
    coords_y = heightmap[0, :, 1]

    mask = _create_mask_from_shape(shape, coords_x, coords_y)
    filtered_heightmap = heightmap[mask]

    logger.info("Extraction complete in %.3fs: %d points extracted", time.perf_counter() - start, len(filtered_heightmap))
    return filtered_heightmap, mask

