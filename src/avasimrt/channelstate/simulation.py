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
from .snow import prepare_snow_scene, apply_snow_material, SNOW_MESH_ID

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SionnaContext:
    scene: Scene
    solver: PathSolver
    rx: Receiver
    txs: list[Transmitter]


@dataclass(slots=True)
class ChannelStateResult:
    """Result of channelstate computation including timing info."""
    samples: dict[str, list[Sample]]
    durations: dict[str, float]  # node_id -> duration in seconds
    total_duration: float  # full elapsed time including setup


def _setup_scene(cfg: ChannelStateConfig, *, scene_xml: Path) -> Scene:
    scene = load_scene(scene_xml.as_posix())

    terrain_material = ITURadioMaterial(
        name="avasimrt_terrain",
        itu_type="wet_ground",
        thickness=float("inf"),
        scattering_coefficient=0.3,
        color=(0.45, 0.35, 0.25)
    )
    scene.add(terrain_material)

    for obj_name, obj in scene.objects.items():
        # Skip snow mesh - it gets its own material
        if obj_name == SNOW_MESH_ID:
            continue
        logger.info("Assigning terrain radio material to object: %s", obj_name)
        obj.radio_material = terrain_material

    # Apply snow material if snow mesh is present
    apply_snow_material(scene, cfg.snow)

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    scene.frequency = cfg.channel.freq_center
    scene.bandwidth = cfg.channel.sc_num * cfg.channel.sc_spacing
    return scene


def _build_context(
    cfg: ChannelStateConfig,
    anchors: Sequence[ResolvedPosition],
    *,
    scene_xml: Path,
) -> _SionnaContext:
    scene = _setup_scene(cfg, scene_xml=scene_xml)

    txs: list[Transmitter] = []
    for a in anchors:
        if a.z is None:
            raise ValueError(
                f"Anchor {a.id} has z=None. Resolve anchor heights in motion step before channelstate."
            )
        tx = Transmitter(
            name=a.id,
            position=mi.Point3f(a.x, a.y, a.z),
            display_radius=a.size,
        )
        scene.add(tx)
        txs.append(tx)

    rx = Receiver(
        name="node",
        position=mi.Point3f(0, 0, 0),
        display_radius=1.0,  # default, updated per trajectory
    )
    scene.add(rx)

    solver = PathSolver()
    return _SionnaContext(scene=scene, solver=solver, rx=rx, txs=txs)


def _solve_paths(ctx: _SionnaContext, cfg: ChannelStateConfig) -> Paths:
    return ctx.solver(
        scene=ctx.scene,
        max_depth=cfg.channel.reflection_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=False,
        seed=cfg.channel.seed,
    )


def _evaluate_cfr(
    *,
    paths,
    cfg: ChannelStateConfig,
    anchors: Sequence[ResolvedPosition],
    node_snapshot,
) -> list[AnchorReading]:
    freqs = subcarrier_frequencies(cfg.channel.sc_num, cfg.channel.sc_spacing)
    h_raw = paths.cfr(frequencies=freqs, out_type="numpy", normalize_delays=False)

    readings: list[AnchorReading] = []

    for i_anchor, anchor_cfg in enumerate(anchors):

        def values_from_index(i_pol: int) -> list[ComplexReading]:
            h = h_raw[0][i_pol][i_anchor][0][0]
            return [
                ComplexReading(freq=float(freqs[c_id]), real=float(c.real), imag=float(c.imag))
                for c_id, c in enumerate(h)
            ]

        values_h = values_from_index(0)
        values_v = values_from_index(1)

        readings.append(
            AnchorReading(
                anchor_id=anchor_cfg.id,
                distance=distance(node_snapshot, anchor_cfg),
                values=[
                    AntennaReading(label="H", mean_db=mean_db_from_values(values_h), frequencies=values_h),
                    AntennaReading(label="V", mean_db=mean_db_from_values(values_v), frequencies=values_v),
                ],
            )
        )

    return readings


def _render_if_enabled(
    *,
    ctx: _SionnaContext,
    cfg: ChannelStateConfig,
    step_idx: int,
    node_pos: mi.Point3f,
    paths: Paths,
    out_dir: Path,
    debug: bool
) -> Path | None:
    r = cfg.render
    if not r.enabled or r.every_n_steps <= 0:
        return None
    if step_idx % r.every_n_steps != 0:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"scene_{step_idx}.png"

    cam = Camera(
        position=mi.Point3f(r.camera_x, r.camera_y, r.camera_z),
        look_at=node_pos,
    )

    a, _ = paths.cir()
    # a is a list of TensorXf; check if num_paths dimension (index 4) is > 0
    has_valid_paths = len(a) > 0 and len(a[0].shape) > 4 and a[0].shape[4] > 0

    if debug:
        with sionna_vispy.patch():
            ctx.scene.preview(paths=paths if has_valid_paths else None)

        sionna_vispy.get_canvas(ctx.scene).show()
        sionna_vispy.get_canvas(ctx.scene).app.run()

    ctx.scene.render_to_file(
        camera=cam,
        paths=paths if has_valid_paths else None,
        filename=img_path.as_posix(),
        resolution=(r.width, r.height),
    )

    return img_path


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

    ctx = _build_context(cfg, anchors, scene_xml=effective_scene_xml)

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
            node_pos = mi.Point3f(float(pos[0]), float(pos[1]), float(pos[2]))
            ctx.rx.position = node_pos

            paths = _solve_paths(ctx, cfg)
            img = _render_if_enabled(ctx=ctx, cfg=cfg, step_idx=idx, node_pos=node_pos, paths=paths,
                                     out_dir=out_dir / node_id,
                                     debug=cfg.debug)

            readings = _evaluate_cfr(paths=paths, cfg=cfg, anchors=anchors, node_snapshot=r0.node)
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
