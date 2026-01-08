from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Sequence

import pybullet_data

from avasimrt.config import AnchorConfig, NodeConfig
from avasimrt.result import NodeSnapshot, Sample
from avasimrt.motion.config import MotionConfig

logger = logging.getLogger(__name__)


def _load_pybullet():
    import os
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        import pybullet as p
    return p


def _connect(p, cfg: MotionConfig) -> int:
    mode = p.GUI if cfg.debug.mode == "GUI" else p.DIRECT
    cid = p.connect(mode)
    if cid < 0:
        raise RuntimeError("failed to connect to pybullet")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, cfg.physics.gravity_z)
    p.setTimeStep(cfg.time.time_step)
    return cid


def _disconnect(p) -> None:
    # pybullet tolerates disconnect() even if already disconnected in many setups,
    # but we keep it explicit.
    try:
        p.disconnect()
    except Exception:
        logger.exception("pybullet disconnect failed")


def _load_terrain_urdf(p, urdf: str = "plane.urdf", *, base_position=(0, 0, 0)) -> int:
    terrain_id = p.loadURDF(urdf, basePosition=list(base_position))
    if terrain_id < 0:
        raise RuntimeError(f"failed to load URDF terrain: {urdf}")
    return terrain_id


def _load_terrain_mesh(p, mesh_path: str | Path, *, scale: float = 1.0) -> int:
    mesh_path = str(Path(mesh_path))

    col_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[scale, scale, scale],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    if col_id < 0:
        raise RuntimeError(f"createCollisionShape failed for file: {mesh_path}")

    vis_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[scale, scale, scale],
    )
    if vis_id < 0:
        logger.warning("createVisualShape returned negative id for %s", mesh_path)

    terrain_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0, 0, 0],
    )
    if terrain_id < 0:
        raise RuntimeError("createMultiBody failed for terrain mesh")

    return terrain_id


def _apply_terrain_dynamics(p, terrain_id: int, cfg: MotionConfig) -> None:
    d = cfg.terrain
    p.changeDynamics(
        terrain_id,
        -1,
        restitution=d.restitution,
        lateralFriction=d.lateral_friction,
        rollingFriction=d.rolling_friction,
        spinningFriction=d.spinning_friction,
        contactDamping=d.contact_damping,
        contactStiffness=d.contact_stiffness,
    )


def _apply_node_dynamics(p, node_id: int, cfg: MotionConfig) -> None:
    d = cfg.node
    p.changeDynamics(
        node_id,
        -1,
        restitution=d.restitution,
        lateralFriction=d.lateral_friction,
        rollingFriction=d.rolling_friction,
        spinningFriction=d.spinning_friction,
        linearDamping=d.linear_damping,
        angularDamping=d.angular_damping,
        contactDamping=d.contact_damping,
        contactStiffness=d.contact_stiffness,
    )


def height_on_terrain(
    p,
    *,
    x: float,
    y: float,
    terrain_id: int | None = None,
    z_start: float = 1000.0,
    z_end: float = -10.0,
) -> float:
    hits = p.rayTest([x, y, z_start], [x, y, z_end])
    hit_uid, hit_fraction, hit_pos = hits[0][0], hits[0][2], hits[0][3]
    if hit_fraction == 1.0 or hit_uid < 0:
        raise RuntimeError(f"No hit at x={x}, y={y}.")
    return float(hit_pos[2])


def spawn_node(
    p,
    *,
    cfg: MotionConfig,
    node_cfg: NodeConfig,
    terrain_id: int,
) -> int:
    if node_cfg.z is None:
        h = height_on_terrain(p, x=node_cfg.x, y=node_cfg.y, terrain_id=terrain_id)
        z = h + node_cfg.size * 1.02
    else:
        z = node_cfg.z

    col = p.createCollisionShape(p.GEOM_SPHERE, radius=node_cfg.size)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=node_cfg.size, rgbaColor=[0.89, 0.39, 0.0, 1.0])
    node_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[node_cfg.x, node_cfg.y, z],
    )
    _apply_node_dynamics(p, node_id, cfg)
    return node_id


def resolve_anchor_heights(
    p,
    anchors: Sequence[AnchorConfig],
    *,
    terrain_id: int,
) -> list[tuple[str, float, float, float]]:
    """
    Returns anchor positions with z resolved (does NOT mutate configs).
    """
    out: list[tuple[str, float, float, float]] = []
    for a in anchors:
        z = a.z
        if z is None:
            h = height_on_terrain(p, x=a.x, y=a.y, terrain_id=terrain_id)
            z = h + a.size
        out.append((a.id, a.x, a.y, float(z)))
    return out


def _set_camera_target(p, cfg: MotionConfig, pos) -> None:
    if cfg.debug.mode != "GUI":
        return
    d = cfg.debug
    p.resetDebugVisualizerCamera(
        cameraDistance=d.camera_distance,
        cameraYaw=d.camera_yaw,
        cameraPitch=d.camera_pitch,
        cameraTargetPosition=list(pos),
    )


def simulate_motion(
    *,
    cfg: MotionConfig,
    node: NodeConfig,
    anchors: Sequence[AnchorConfig] = (),
    terrain_mesh: str | Path | None = None,
    terrain_mesh_scale: float = 1.0,
    use_plane_if_no_mesh: bool = True,
) -> tuple[list[Sample], list[tuple[str, float, float, float]]]:
    """
    Runs the pybullet motion step and returns:
      - motion_results: list[Result] containing timestamp+NodeSnapshot
      - resolved_anchors: list of (id, x, y, z) for convenience for next step
    """
    p = _load_pybullet()
    _connect(p, cfg)

    try:
        if terrain_mesh is not None:
            terrain_id = _load_terrain_mesh(p, terrain_mesh, scale=terrain_mesh_scale)
        else:
            if not use_plane_if_no_mesh:
                raise ValueError("terrain_mesh is None and use_plane_if_no_mesh=False")
            terrain_id = _load_terrain_urdf(p, "plane.urdf")

        _apply_terrain_dynamics(p, terrain_id, cfg)

        resolved = resolve_anchor_heights(p, anchors, terrain_id=terrain_id) if anchors else []

        node_id = spawn_node(p, cfg=cfg, node_cfg=node, terrain_id=terrain_id)

        dt = cfg.time.time_step
        sim_time = cfg.time.sim_time
        sample_every = cfg.time.sampling_rate

        total_steps = int(sim_time / dt)
        log_every = max(1, total_steps // 20)

        t = 0.0
        next_sample = 0.0
        step_idx = 0

        out: list[Sample] = []

        logger.info("Motion: sim_time=%.2fs, dt=%.4fs, steps=%d", sim_time, dt, total_steps)

        while t < sim_time:
            p.stepSimulation()
            step_idx += 1

            pos, orn = p.getBasePositionAndOrientation(node_id)
            _set_camera_target(p, cfg, pos)

            if t >= next_sample:
                next_sample += sample_every
                lin_vel, _ang_vel = p.getBaseVelocity(node_id)
                out.append(
                    Sample(
                        timestamp=t,
                        node=NodeSnapshot(
                            position=pos,
                            orientation=orn,
                            linear_velocity=lin_vel,
                        ),
                    )
                )

            if step_idx % log_every == 0:
                percent = int(step_idx * 100 / max(1, total_steps))
                logger.info("Motion progress: %3d%% (%d/%d)", percent, step_idx, total_steps)

            t += dt
            if cfg.debug.mode == "GUI":
                time.sleep(dt)

        logger.info("Motion finished: %d snapshots collected", len(out))
        return out, resolved

    finally:
        _disconnect(p)
