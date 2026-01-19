from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mitsuba as mi
from sionna.rt import (
    Camera,
    PathSolver,
    PlanarArray,
    Receiver,
    Scene,
    Transmitter,
    load_scene,
    subcarrier_frequencies,
)

from avasimrt.config import AnchorConfig
from avasimrt.result import AnchorReading, AntennaReading, ComplexReading, Sample
from avasimrt.math import distance, mean_db_from_values
from .config import ChannelStateConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SionnaContext:
    scene: Scene
    solver: PathSolver
    rx: Receiver
    txs: list[Transmitter]


def _setup_scene(cfg: ChannelStateConfig, *, scene_xml: Path) -> Scene:
    scene = load_scene(scene_xml.as_posix())

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


def _build_context(cfg: ChannelStateConfig, anchors: Sequence[AnchorConfig], *, rx_radius: float, scene_xml: Path) -> _SionnaContext:
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
        display_radius=rx_radius,
    )
    scene.add(rx)

    solver = PathSolver()
    return _SionnaContext(scene=scene, solver=solver, rx=rx, txs=txs)


def _solve_paths(ctx: _SionnaContext, cfg: ChannelStateConfig):
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
    anchors: Sequence[AnchorConfig],
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
    paths,
    out_dir: Path
) -> Path | None:
    r = cfg.render
    if not r.enabled or r.every_n_steps <= 0:
        return None
    if step_idx % r.every_n_steps != 0:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / f"scene_{step_idx}.png"

    cam = Camera(
        position=mi.Point3f(
            node_pos[0] + r.distance / 2,
            node_pos[1] - r.distance * 2,
            node_pos[2] + r.distance,
        ),
        look_at=node_pos,
    )

    ctx.scene.render_to_file(
        camera=cam,
        paths=paths,
        filename=img_path.as_posix(),
        resolution=(r.width, r.height),
    )
    return img_path


def estimate_channelstate(
    *,
    cfg: ChannelStateConfig,
    anchors: Sequence[AnchorConfig],
    motion_results: Sequence[Sample],
    out_dir: Path,
    scene_xml: Path
) -> list[Sample]:
    """
    Computes channel state (CFR-derived readings) for each motion result time step.
    Returns new Result objects with readings and optional image paths.
    """
    if not motion_results:
        logger.warning("ChannelState: no motion results to process")
        return []

    rx_radius = 0.2
    try:
        rx_radius = 0.2
    except Exception:
        pass

    if any(a.z is None for a in anchors):
        missing = [a.id for a in anchors if a.z is None]
        raise ValueError(f"Anchors must have resolved z before channelstate: {missing}")

    ctx = _build_context(cfg, anchors, rx_radius=rx_radius, scene_xml=scene_xml)

    total = len(motion_results)
    log_every = max(1, total // 20)
    out: list[Sample] = []

    logger.info("ChannelState: starting for %d time steps", total)

    for idx, r0 in enumerate(motion_results):
        node_pos = mi.Point3f(r0.node.position)
        ctx.rx.position = node_pos

        paths = _solve_paths(ctx, cfg)
        img = _render_if_enabled(ctx=ctx, cfg=cfg, step_idx=idx, node_pos=node_pos, paths=paths, out_dir=out_dir)

        readings = _evaluate_cfr(paths=paths, cfg=cfg, anchors=anchors, node_snapshot=r0.node)
        out.append(Sample(timestamp=r0.timestamp, node=r0.node, readings=readings, image=img))

        if idx % log_every == 0:
            percent = int(idx * 100 / max(1, total))
            logger.info("ChannelState progress: %3d%% (%d/%d)", percent, idx, total)

    logger.info("ChannelState finished: %d evaluated time steps", len(out))
    return out
