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
from sionna.rt import ITURadioMaterial, Scene, RadioMaterial

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition

if TYPE_CHECKING:
    from avasimrt.channelstate.config import SnowConfig

logger = logging.getLogger(__name__)

SNOW_SPHERE_PREFIX = "snow_"


def ulaby_long_snow_dielectric(Ps: float, mv: float, f_ghz: float) -> tuple[float, float]:
    """
    Calculate complex dielectric constant of snow using Ulaby & Long model.

    Source: Microwave Radar and Radiometric Remote Sensing (Ulaby & Long, 2014)
    Section 4-6.2, Code 4.6

    Args:
        Ps: Dry snow density (g/cm³), typical range 0.1-0.5
        mv: Volumetric water content (%), range 0-30 (use ~0.5 for dry snow)
        f_ghz: Frequency in GHz

    Returns:
        (eps_real, eps_imag): Real and imaginary parts of relative permittivity
    """
    A1 = 0.78 + 0.03 * f_ghz - 0.58e-3 * f_ghz**2
    A2 = 0.97 - 0.39e-2 * f_ghz + 0.39e-3 * f_ghz**2
    B1 = 0.31 - 0.05 * f_ghz + 0.87e-3 * f_ghz**2

    A = A1 * (1.0 + 1.83 * Ps + 0.02 * mv**1.015) + B1
    B = 0.073 * A1
    C = 0.073 * A2
    x = 1.31
    f0 = 9.07

    eps_real = A + (B * mv**x) / (1 + (f_ghz / f0)**2)
    eps_imag = (C * (f_ghz / f0) * mv**x) / (1 + (f_ghz / f0)**2)

    return eps_real, eps_imag


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


def create_sphere_ply(filepath: Path, radius: float, n_lat: int = 16, n_lon: int = 32) -> None:
    """Create a PLY file containing a sphere mesh with the given radius.

    WARNING: Sionna does NOT support transmission through closed 3D objects like spheres.
    Use create_disk_ply() instead for snow obstacles that rays should pass through.

    Args:
        filepath: Path where the PLY file will be written.
        radius: Radius of the sphere.
        n_lat: Number of latitude divisions.
        n_lon: Number of longitude divisions.
    """
    vertices = []
    for i in range(n_lat + 1):
        lat = np.pi * i / n_lat - np.pi / 2
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            x = radius * np.cos(lat) * np.cos(lon)
            y = radius * np.cos(lat) * np.sin(lon)
            z = radius * np.sin(lat)
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)

    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * n_lon + j
            v1 = i * n_lon + (j + 1) % n_lon
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + (j + 1) % n_lon
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    faces = np.array(faces, dtype=np.int32)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode("ascii"))
        vertices.tofile(f)
        for face in faces:
            f.write(np.uint8(3).tobytes())
            f.write(face.tobytes())


def create_disk_ply(filepath: Path, radius: float, n_segments: int = 32) -> None:
    """Create a PLY file containing a flat disk (thin surface) in the XZ plane.

    Sionna models transmission through thin surfaces using the material's thickness
    parameter. This disk is suitable for modeling snow layers that rays pass through.

    The disk is oriented in the XZ plane (vertical, facing Y direction), which is
    suitable for obstacles between TX and RX along the Y axis.

    Args:
        filepath: Path where the PLY file will be written.
        radius: Radius of the disk.
        n_segments: Number of segments around the circumference.
    """
    # Center vertex + circumference vertices
    vertices = [[0.0, 0.0, 0.0]]  # Center at origin

    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        vertices.append([x, 0.0, z])  # Disk in XZ plane (vertical)

    vertices = np.array(vertices, dtype=np.float32)

    # Create triangular faces (fan from center)
    faces = []
    for i in range(n_segments):
        v1 = i + 1
        v2 = (i + 1) % n_segments + 1
        # Front face
        faces.append([0, v1, v2])
        # Back face (reversed winding for double-sided)
        faces.append([0, v2, v1])

    faces = np.array(faces, dtype=np.int32)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode("ascii"))
        vertices.tofile(f)
        for face in faces:
            f.write(np.uint8(3).tobytes())
            f.write(face.tobytes())


def create_scene_with_snow(
    xml_path: Path,
    meshes_dir: Path,
    radius: float,
    positions: np.ndarray,
) -> Path:
    """Create a new scene XML with snow spheres added at multiple positions.

    Args:
        xml_path: Path to the original scene XML file.
        meshes_dir: Directory where the Snow.ply mesh will be created.
        radius: Radius of the snow sphere.
        positions: Array of shape (n, 3) with (x, y, z) coordinates,
            as returned by extract_relevant_heightmap.

    Returns:
        Path to the newly created XML file with "-with_snow" suffix.
    """
    ply_path = meshes_dir / "Snow.ply"
    create_sphere_ply(ply_path, radius)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find existing bsdf to reference
    bsdf = root.find("bsdf")
    bsdf_id = bsdf.get("id") if bsdf is not None else None

    for i, position in enumerate(positions):
        shape_id = f"snow_{i}"
        snow_shape = ET.SubElement(root, "shape", {"type": "ply", "id": shape_id, "name": shape_id})
        ET.SubElement(snow_shape, "string", {"name": "filename", "value": "meshes/Snow.ply"})

        if bsdf_id:
            ET.SubElement(snow_shape, "ref", {"id": bsdf_id, "name": "bsdf"})

        transform = ET.SubElement(snow_shape, "transform", {"name": "to_world"})
        ET.SubElement(transform, "translate", {"x": str(float(position[0])), "y": str(float(position[1])), "z": str(float(position[2]))})

    output_path = xml_path.with_stem(xml_path.stem + "-with_snow")
    tree.write(output_path, encoding="unicode", xml_declaration=False)

    return output_path



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

    # Inject individual snow spheres into XML
    snow_scene_xml = out_dir / "scene_with_snow.xml"
    meshes_dir = out_dir / "meshes"
    snow_scene_xml = create_scene_with_snow(positions=relevant_positions,
                                       radius=cfg.box_size,
                                       xml_path=scene_xml,
                                       meshes_dir=meshes_dir)

    logger.info("Snow scene prepared in %.3fs", time.perf_counter() - start)
    return snow_scene_xml, len(relevant_positions)


class Snow:
    """Snow material for Sionna ray tracing using Ulaby & Long (2014) model.

    Material properties are calculated from physical snow parameters:
    - Ps: Dry snow density (g/cm³)
    - mv: Volumetric water content (%)
    - freq_hz: Frequency (Hz)

    The model computes complex permittivity (ε' + jε'') which determines:
    - ε': Refractive index n = √ε' (affects wave speed, reflection)
    - ε'': Absorption loss (converted to conductivity σ for Sionna)

    A calibration factor adjusts conductivity so Sionna's transmission model
    matches theoretical volume absorption from Ulaby & Long.
    """

    # Calibration: Sionna's model gives ~2x higher attenuation than theoretical
    _CONDUCTIVITY_CALIBRATION = 2.1

    def __init__(
        self,
        thickness_m: float = 1.0,
        Ps: float = 0.4,
        mv: float = 0.5,
        freq_hz: float = 3.5e9,
        color: tuple[float, float, float] = (0.95, 0.97, 1.0),
    ):
        """Initialize snow material from physical parameters.

        Args:
            thickness_m: Snow thickness in meters (default: 1.0)
            Ps: Dry snow density in g/cm³ (default: 0.4, typical for avalanche debris)
            mv: Volumetric water content in % (default: 0.5, nearly dry snow)
            freq_hz: Frequency in Hz (default: 3.5 GHz)
            color: RGB color tuple for visualization
        """
        freq_ghz = freq_hz / 1e9
        eps_r, eps_i = ulaby_long_snow_dielectric(Ps, mv, freq_ghz)

        # Convert ε'' to conductivity: σ = ωε₀ε'' / calibration
        epsilon_0 = 8.854e-12
        sigma = 2 * np.pi * freq_hz * epsilon_0 * eps_i / self._CONDUCTIVITY_CALIBRATION

        logger.info(
            "Snow(Ps=%.2f g/cm³, mv=%.1f%%, f=%.2f GHz, d=%.2f m): ε'=%.3f, ε''=%.4f, σ=%.6f S/m",
            Ps, mv, freq_ghz, thickness_m, eps_r, eps_i, sigma
        )

        self._material = RadioMaterial(
            name="avasimrt_snow",
            relative_permittivity=eps_r,
            conductivity=sigma,
            thickness=thickness_m,
            color=color,
        )
        self.thickness = thickness_m

    def apply_material(self, scene: Scene) -> None:
        snow_objects = [name for name in scene.objects if name.startswith(SNOW_SPHERE_PREFIX)]

        if not snow_objects:
            logger.warning("No snow spheres found in scene (prefix='%s')", SNOW_SPHERE_PREFIX)
            return

        material = self._material
        scene.add(material)

        for obj_name in snow_objects:
            scene.objects[obj_name].radio_material = material

        logger.info("Applied snow radio material to %d snow spheres", len(snow_objects))
