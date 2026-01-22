from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
from sionna.rt import ITURadioMaterial, Scene

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition

if TYPE_CHECKING:
    from avasimrt.channelstate.config import SnowConfig

logger = logging.getLogger(__name__)

SNOW_MESH_ID = "snow_mesh"
SNOW_MATERIAL_ID = "mat-itu_wet_ground"


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
    quick: bool = True,
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


def _generate_sphere_mesh(n_lat: int = 8, n_lon: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Generate a unit sphere mesh centered at origin.

    Returns:
        vertices: (n_vertices, 3) array of vertex positions
        faces: (n_faces, 3) array of vertex indices for each triangle
    """
    vertices = []
    faces = []

    # Generate vertices
    for i in range(n_lat + 1):
        lat = np.pi * i / n_lat - np.pi / 2  # -pi/2 to pi/2
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)

    # Generate faces
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + (j + 1) % n_lon

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.int32)
    return vertices, faces


def _generate_combined_snow_ply(
    positions: np.ndarray,
    box_size: float,
    levels: int,
    output_path: Path,
) -> int:
    """Generate a PLY file with spheres at all snow positions.

    Args:
        positions: (n, 3) array of (x, y, z) positions
        box_size: Size/diameter of each snow sphere
        levels: Number of vertical levels of snow
        output_path: Path to write the PLY file

    Returns:
        Total number of spheres created
    """
    start = time.perf_counter()

    # Generate base sphere geometry
    base_vertices, base_faces = _generate_sphere_mesh(n_lat=6, n_lon=12)
    n_base_verts = len(base_vertices)
    n_base_faces = len(base_faces)

    total_spheres = len(positions) * levels
    total_vertices = total_spheres * n_base_verts
    total_faces = total_spheres * n_base_faces

    logger.info("Generating snow PLY: %d spheres (%d positions x %d levels), %d vertices, %d faces",
                total_spheres, len(positions), levels, total_vertices, total_faces)

    # Pre-allocate arrays
    all_vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    all_faces = np.zeros((total_faces, 3), dtype=np.int32)

    sphere_idx = 0
    log_interval = max(1, len(positions) // 10)
    radius = box_size / 2.0

    for i, pos in enumerate(positions):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        for level in range(levels):
            z_offset = z + (level + 0.5) * box_size

            # Transform vertices for this sphere
            vert_start = sphere_idx * n_base_verts
            vert_end = vert_start + n_base_verts
            all_vertices[vert_start:vert_end] = base_vertices * radius + [x, y, z_offset]

            # Add faces with offset vertex indices
            face_start = sphere_idx * n_base_faces
            face_end = face_start + n_base_faces
            all_faces[face_start:face_end] = base_faces + vert_start

            sphere_idx += 1

        if i % log_interval == 0:
            progress = (i / len(positions)) * 100
            logger.info("Snow PLY generation progress: %.0f%% (%d/%d positions)", progress, i, len(positions))

    # Write PLY file
    logger.info("Writing snow PLY to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {total_vertices}
property float x
property float y
property float z
element face {total_faces}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))

        # Vertices
        all_vertices.tofile(f)

        # Faces (each face: 1 byte count + 3 int indices)
        for face in all_faces:
            f.write(np.uint8(3).tobytes())
            f.write(face.tobytes())

    elapsed = time.perf_counter() - start
    logger.info("Snow PLY generated in %.3fs: %d spheres, %.2f MB",
                elapsed, total_spheres, output_path.stat().st_size / 1024 / 1024)

    return total_spheres


def _inject_snow_into_xml(
    scene_xml: Path,
    snow_ply_path: Path,
    output_xml: Path,
) -> None:
    """Inject snow mesh and material into scene XML.

    Args:
        scene_xml: Original scene XML path
        snow_ply_path: Path to the snow PLY mesh
        output_xml: Path to write the modified XML
    """
    tree = ET.parse(scene_xml)
    root = tree.getroot()

    # Add snow material (using a default bsdf, will be replaced with ITURadioMaterial after loading)
    snow_bsdf = ET.SubElement(root, "bsdf", {"type": "twosided", "id": SNOW_MATERIAL_ID, "name": SNOW_MATERIAL_ID})
    inner_bsdf = ET.SubElement(snow_bsdf, "bsdf", {"type": "principled", "name": "bsdf"})
    ET.SubElement(inner_bsdf, "rgb", {"value": "0.95 0.97 1.0", "name": "base_color"})
    ET.SubElement(inner_bsdf, "float", {"name": "roughness", "value": "0.8"})

    # Add snow mesh shape
    # Make the path relative to the output XML location
    relative_ply_path = snow_ply_path.relative_to(output_xml.parent)

    snow_shape = ET.SubElement(root, "shape", {"type": "ply", "id": SNOW_MESH_ID, "name": SNOW_MESH_ID})
    ET.SubElement(snow_shape, "string", {"name": "filename", "value": str(relative_ply_path)})
    ET.SubElement(snow_shape, "boolean", {"name": "face_normals", "value": "true"})
    ET.SubElement(snow_shape, "ref", {"id": SNOW_MATERIAL_ID, "name": "bsdf"})

    # Write modified XML
    tree.write(output_xml, encoding="unicode", xml_declaration=False)

    # Re-read and add XML declaration + scene version
    with open(output_xml, 'r') as f:
        content = f.read()

    with open(output_xml, 'w') as f:
        f.write(content)

    logger.info("Injected snow mesh into XML: %s", output_xml)


def prepare_snow_scene(
    cfg: "SnowConfig",
    scene_xml: Path,
    heightmap: np.ndarray,
    anchors: Sequence[ResolvedPosition],
    nodes: Sequence[Sequence[NodeSnapshot]],
    out_dir: Path,
) -> tuple[Path, int]:
    """Prepare a scene XML with snow mesh baked in.

    Args:
        cfg: Snow configuration
        scene_xml: Original scene XML path
        heightmap: 3D array of shape (n_x, n_y, 3) with (x, y, z) coordinates
        anchors: Sequence of anchor positions
        nodes: Sequence of node trajectories
        out_dir: Output directory for generated files

    Returns:
        Tuple of (path to modified scene XML, number of snow spheres)
    """
    if not cfg.enabled:
        logger.info("Snow disabled, using original scene")
        return scene_xml, 0

    start = time.perf_counter()
    logger.info("Preparing snow scene: box_size=%.2f, levels=%d, margin=%.1f",
                cfg.box_size, cfg.levels, cfg.margin)

    # Extract relevant positions
    relevant_positions, _ = extract_relevant_heightmap(
        heightmap=heightmap,
        anchors=anchors,
        nodes=nodes,
        margin=cfg.margin,
    )

    if len(relevant_positions) == 0:
        logger.warning("No relevant heightmap positions found for snow placement")
        return scene_xml, 0

    # Generate snow PLY
    snow_ply_path = out_dir / "meshes" / "snow_mesh.ply"
    n_spheres = _generate_combined_snow_ply(
        positions=relevant_positions,
        box_size=cfg.box_size,
        levels=cfg.levels,
        output_path=snow_ply_path,
    )

    # Inject into XML
    snow_scene_xml = out_dir / "scene_with_snow.xml"
    _inject_snow_into_xml(scene_xml, snow_ply_path, snow_scene_xml)

    logger.info("Snow scene prepared in %.3fs", time.perf_counter() - start)
    return snow_scene_xml, n_spheres


class Snow:

    def __init__(self, 
                type: str,
                thickness: float,
                scattering_coef: float,
                color: tuple[float, float, float] = (0.95, 0.97, 1.0)):
        self._material = ITURadioMaterial(
            name="avasimrt_snow",
            itu_type=type,
            thickness=thickness,
            scattering_coefficient=scattering_coef,
            color=color,
        )

    def apply_material(self, scene: Scene) -> None:
        if SNOW_MESH_ID not in scene.objects:
            logger.warning("Snow mesh '%s' not found in scene objects", SNOW_MESH_ID)
            return

        material = self._material

        scene.add(material)
        scene.objects[SNOW_MESH_ID].radio_material = material

        logger.info("Applied snow radio material to mesh '%s'", SNOW_MESH_ID)
