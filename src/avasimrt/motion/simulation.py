from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Sequence

import pybullet_data

from avasimrt.preprocessing.result import ResolvedPosition
from avasimrt.motion.result import NodeSnapshot
from avasimrt.result import Sample
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
    p.setPhysicsEngineParameter(numSolverIterations=50)
    p.setPhysicsEngineParameter(numSubSteps=4)
    return cid


def _disconnect(p) -> None:
    try:
        p.disconnect()
    except Exception:
        logger.exception("pybullet disconnect failed")

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


def spawn_node(
    p,
    *,
    cfg: MotionConfig,
    node_cfg: ResolvedPosition,
) -> int:

    col = p.createCollisionShape(p.GEOM_SPHERE, radius=node_cfg.size)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=node_cfg.size, rgbaColor=[0.89, 0.39, 0.0, 1.0])
    node_id = p.createMultiBody(
        baseMass=1,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[node_cfg.x, node_cfg.y, node_cfg.z],
    )
    _apply_node_dynamics(p, node_id, cfg)
    p.changeDynamics(node_id, -1, ccdSweptSphereRadius=node_cfg.size * 0.9)
    p.changeDynamics(node_id, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    return node_id


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


def _simulate_single_node(
    *,
    cfg: MotionConfig,
    node_cfg: ResolvedPosition,
    terrain_mesh_scale: float,
    scene_obj: Path,
) -> list[Sample]:
    """Simulate a single node independently."""
    p = _load_pybullet()
    _connect(p, cfg)

    try:
        terrain_id = _load_terrain_mesh(p, scene_obj, scale=terrain_mesh_scale)
        _apply_terrain_dynamics(p, terrain_id, cfg)

        node_id = spawn_node(p, cfg=cfg, node_cfg=node_cfg)

        dt = cfg.time.time_step
        sim_time = cfg.time.sim_time
        sample_every = cfg.time.sampling_rate

        total_steps = int(sim_time / dt)
        log_every = max(1, total_steps // 20)

        t = 0.0
        next_sample = 0.0
        step_idx = 0

        out: list[Sample] = []

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
                            size=node_cfg.size,
                        ),
                    )
                )

            if step_idx % log_every == 0:
                percent = int(step_idx * 100 / max(1, total_steps))
                logger.info("Motion [%s] progress: %3d%% (%d/%d)", node_cfg.id, percent, step_idx, total_steps)

            t += dt
            if cfg.debug.mode == "GUI":
                time.sleep(dt)

        return out

    finally:
        _disconnect(p)


def simulate_motion(
    *,
    cfg: MotionConfig,
    nodes: Sequence[ResolvedPosition],
    terrain_mesh_scale: float = 1.0,
    scene_obj: Path
) -> dict[str, list[Sample]]:
    """
    Runs independent pybullet simulations for each node (no inter-node collisions).
    """
    if scene_obj is None:
        raise ValueError("scene_obj is None")
    

    logger.info("Motion: sim_time=%.2fs, dt=%.4fs, nodes=%d", cfg.time.sim_time, cfg.time.time_step, len(nodes))

    results: dict[str, list[Sample]] = {}
    for node_cfg in nodes:
        logger.info("Motion: simulating node '%s'", node_cfg.id)
        samples = _simulate_single_node(
            cfg=cfg,
            node_cfg=node_cfg,
            terrain_mesh_scale=terrain_mesh_scale,
            scene_obj=scene_obj,
        )
        results[node_cfg.id] = samples
        logger.info("Motion: node '%s' finished with %d samples", node_cfg.id, len(samples))

    return results
