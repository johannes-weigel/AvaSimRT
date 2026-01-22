from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mitsuba as mi
from sionna.rt import (
    Camera,
    ITURadioMaterial,
    PathSolver,
    PlanarArray,
    Receiver,
    Scene,
    Transmitter,
    load_scene,
    subcarrier_frequencies,
    Paths
)
import sionna_vispy

import numpy as np

from avasimrt.preprocessing.result import ResolvedPosition
from avasimrt.result import AnchorReading, AntennaReading, ComplexReading, Sample
from avasimrt.math import distance, mean_db_from_values
from .config import ChannelStateConfig
from .snow import prepare_snow_scene, Snow, SNOW_MESH_ID

logger = logging.getLogger(__name__)

Position3D = tuple[float, float, float]
TransmitterConfig = tuple[str, Position3D, float]
Resolution = tuple[int, int]


# Infinite thickness to prevent any transmition though mountain
terrain_material = ITURadioMaterial(name="avasimrt_terrain",
                                    itu_type="wet_ground",
                                    thickness=float("inf"),
                                    scattering_coefficient=0.3,
                                    color=(0.45, 0.35, 0.25))

class _SionnaContext:
    def __init__(self,
                 scene: Scene,
                 solver: PathSolver,
                 rx: Receiver,
                 txs: list[Transmitter],
                 reflection_depth: int, 
                 seed: int | None):
        self.scene = scene
        self._solver = solver
        self.rx = rx
        self._txs = txs
        
        self._seed = seed if seed else 42
        self._reflection_depth = reflection_depth

    def solve_paths(self) -> Paths:
        return self._solver(scene=self.scene,
                           max_depth=self._reflection_depth,
                           los=True,
                           specular_reflection=True,
                           diffuse_reflection=False,
                           refraction=True,
                           synthetic_array=False,
                           seed=self._seed)
    
    def render_to_file(self, paths: Paths | None, *,
                       origin: Position3D,
                       target: Position3D,
                       file_path: Path,
                       resolution: Resolution):
        cam = Camera(position=mi.Point3f(origin),
                     look_at=mi.Point3f(target))
        self.scene.render_to_file(camera=cam,
                                  paths=paths,
                                  filename=file_path.as_posix(),
                                  resolution=resolution,
                                  # no oriented devices
                                  show_orientations=False)
    
    def render_if_enabled(self, *,
                          cfg: ChannelStateConfig,
                          step_idx: int,
                          node_pos: Position3D,
                          paths: Paths,
                          out_dir: Path,
                          debug: bool) -> Path | None:
        r = cfg.render
        if not r.enabled or r.every_n_steps <= 0:
            return None
        if step_idx % r.every_n_steps != 0:
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"scene_{step_idx}.png"

        a, _ = paths.cir()
        # a is a list of TensorXf; check if num_paths dimension (index 4) is > 0
        has_valid_paths = len(a) > 0 and len(a[0].shape) > 4 and a[0].shape[4] > 0

        if debug:
            with sionna_vispy.patch():
                self.scene.preview(paths=paths if has_valid_paths else None)

            sionna_vispy.get_canvas(self.scene).show()
            sionna_vispy.get_canvas(self.scene).app.run()

        self.render_to_file(origin=(r.camera_x, r.camera_y, r.camera_z),
                            target=node_pos,
                            paths=paths if has_valid_paths else None,
                            file_path=img_path,
                            resolution=(r.width, r.height))

        return img_path



@dataclass(slots=True)
class ChannelStateResult:
    """Result of channelstate computation including timing info."""
    samples: dict[str, list[Sample]]
    durations: dict[str, float]  # node_id -> duration in seconds
    total_duration: float  # full elapsed time including setup


def _setup_scene(*, 
                 scene_src: Path | str | None,
                 snow: Snow | None,
                 freq_center: float | None,
                 bandwidth: float | None) -> Scene:
    if (scene_src):
        scene = load_scene(scene_src.as_posix() 
                           if isinstance(scene_src, Path) 
                           else scene_src)
    else:
        scene = load_scene()

    scene.add(terrain_material)

    for obj_name, obj in scene.objects.items():
        if obj_name == SNOW_MESH_ID:
            continue
        logger.info("Assigning terrain radio material to object: %s", obj_name)
        obj.radio_material = terrain_material

    if (snow):
        snow.apply_material(scene)

    # Defaults from tutorial, experiment-specific tuning
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="tr38901",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="dipole",
                                 polarization="cross")

    if (freq_center):
        scene.frequency = freq_center
    if (bandwidth):
        scene.bandwidth = bandwidth
    
    return scene


def _build_context(*,
                   anchors: Sequence[TransmitterConfig],
                   scene_src: Path | str | None,
                   snow: Snow | None,
                   freq_center: float | None,
                   bandwidth: float | None,
                   reflection_depth: int, 
                   seed: int | None) -> _SionnaContext:
    scene = _setup_scene(scene_src=scene_src,
                         snow=snow,
                         freq_center=freq_center,
                         bandwidth=bandwidth)

    txs: list[Transmitter] = []
    for id, pos, size in anchors:
        tx = Transmitter(name=id,
                         position=mi.Point3f(pos),
                         display_radius=size,
                         # color picked from image of anchor
                         color=(0.180, 0.282, 0.388))
        scene.add(tx)
        txs.append(tx)

    # Placeholder - gets updated for each trajectory / node configuration
    rx = Receiver(name="node",
                  position=mi.Point3f(0, 0, 0),
                  display_radius=1.0,
                  # color picked from image of node
                  color=(1, 0.308, 0.1))
    scene.add(rx)

    solver = PathSolver()
    return _SionnaContext(scene=scene, solver=solver, rx=rx, txs=txs, reflection_depth=reflection_depth, seed=seed)


def _evaluate_cfr(paths, *,
                  freqs,
                  anchors: Sequence[TransmitterConfig],
                  node_pos: Position3D) -> tuple[np.ndarray, np.ndarray]:
    h_raw = paths.cfr(frequencies=freqs, out_type="numpy", normalize_delays=False)

    cfr = h_raw[0][:, :, 0, 0, :]
    cfr = np.transpose(cfr, (1, 0, 2))

    distances = np.array([distance(node_pos, a[1]) for a in anchors], dtype=np.float64)

    return cfr, distances


def _cfr_to_readings(cfr: np.ndarray,
                     distances: np.ndarray,
                     freqs: np.ndarray,
                     anchor_ids: Sequence[str],) -> list[AnchorReading]:
    antenna_labels = ("H", "V")
    readings: list[AnchorReading] = []

    for a, anchor_id in enumerate(anchor_ids):
        antenna_values: list[AntennaReading] = []

        for p, label in enumerate(antenna_labels):
            freq_readings = [
                ComplexReading(freq=float(freqs[f]), real=float(cfr[a, p, f].real), imag=float(cfr[a, p, f].imag))
                for f in range(len(freqs))
            ]
            antenna_values.append(
                AntennaReading(label=label, mean_db=mean_db_from_values(freq_readings), frequencies=freq_readings)
            )

        readings.append(
            AnchorReading(anchor_id=anchor_id, distance=float(distances[a]), values=antenna_values)
        )

    return readings



def estimate_channelstate(
    *,
    cfg: ChannelStateConfig,
    anchors: Sequence[ResolvedPosition],
    trajectories: dict[str, list[Sample]],
    out_dir: Path,
    scene_xml: Path,
    heightmap: np.ndarray | None = None,
) -> ChannelStateResult:
    """
    Computes channel state (CFR-derived readings) for each trajectory.
    Returns new Result objects with readings and optional image paths.

    Returns:
        ChannelStateResult containing samples and per-node durations
    """
    if not trajectories:
        logger.warning("ChannelState: no trajectories to process")
        return ChannelStateResult(samples={}, durations={}, total_duration=0.0)

    start_time = time.perf_counter()

    if any(a.z is None for a in anchors):
        missing = [a.id for a in anchors if a.z is None]
        raise ValueError(f"Anchors must have resolved z before channelstate: {missing}")

    all_results: dict[str, list[Sample]] = {}
    durations: dict[str, float] = {}

    # Prepare snow scene if enabled (generates PLY and modified XML)
    effective_scene_xml = scene_xml

    snow = None
    if cfg.snow.enabled and heightmap is not None:
        nodes = [[sample.node for sample in samples] for samples in trajectories.values()]
        effective_scene_xml, n_snow = prepare_snow_scene(
            cfg=cfg.snow,
            scene_xml=scene_xml,
            heightmap=heightmap,
            anchors=anchors,
            nodes=nodes,
            out_dir=out_dir,
        )
        logger.info("Snow scene prepared with %d spheres", n_snow)

        snow = Snow(cfg.snow.material.itu_type, cfg.snow.material.thickness, cfg.snow.material.scattering_coefficient)

    unpacked_anchors = [(a.id, (a.x, a.y, a.z), a.size) for a in anchors]

    ctx = _build_context(anchors=unpacked_anchors, 
                         scene_src=effective_scene_xml,
                         snow=snow,
                         freq_center=cfg.channel.freq_center,
                         bandwidth=cfg.channel.sc_num * cfg.channel.sc_spacing,
                         reflection_depth=cfg.channel.reflection_depth, seed=cfg.channel.seed)

    freqs = subcarrier_frequencies(cfg.channel.sc_num, cfg.channel.sc_spacing)

    for node_id, motion_results in trajectories.items():
        if not motion_results:
            logger.warning("ChannelState: node '%s' has no motion results, skipping", node_id)
            continue

        node_start = time.perf_counter()
        rx_radius = motion_results[0].node.size
        ctx.rx.display_radius = rx_radius

        logger.info("ChannelState: processing trajectory for node '%s' (rx_radius=%.3f)", node_id, rx_radius)

        total = len(motion_results)
        log_every = max(1, total // 20)
        out: list[Sample] = []

        logger.info("ChannelState: starting for node '%s' with %d time steps", node_id, total)

        for idx, r0 in enumerate(motion_results):
            pos = r0.node.position
            node_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            ctx.rx.position = mi.Point3f(node_pos)

            paths = ctx.solve_paths()
            img = ctx.render_if_enabled(cfg=cfg, step_idx=idx, node_pos=node_pos, paths=paths,
                                     out_dir=out_dir / node_id,
                                     debug=cfg.debug)

            cfr, dists = _evaluate_cfr(paths,
                                        freqs=freqs,
                                        anchors=unpacked_anchors,
                                        node_pos=r0.node.position)
            readings = _cfr_to_readings(cfr, dists, freqs, [a[0] for a in unpacked_anchors])
            out.append(Sample(timestamp=r0.timestamp, node=r0.node, readings=readings, image=img))

            if idx % log_every == 0:
                percent = int(idx * 100 / max(1, total))
                logger.info("ChannelState [%s] progress: %3d%% (%d/%d)", node_id, percent, idx, total)

        node_duration = time.perf_counter() - node_start
        durations[node_id] = node_duration
        logger.info("ChannelState finished for node '%s': %d evaluated time steps in %.2f s", node_id, len(out), node_duration)
        all_results[node_id] = out

    total_duration = time.perf_counter() - start_time
    logger.info("ChannelState completed for %d trajectories in %.2f s", len(all_results), total_duration)
    return ChannelStateResult(samples=all_results, durations=durations, total_duration=total_duration)
