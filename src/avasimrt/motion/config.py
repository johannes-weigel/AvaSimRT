from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class MotionTime:
    """Time settings for the motion simulation."""
    sim_time: float = 60.0
    sampling_rate: float = 1.0
    time_step: float = 1.0 / 240.0

    def __post_init__(self) -> None:
        if self.sim_time <= 0:
            raise ValueError(f"sim_time must be > 0, got {self.sim_time!r}")
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be > 0, got {self.sampling_rate!r}")
        if self.time_step <= 0:
            raise ValueError(f"time_step must be > 0, got {self.time_step!r}")
        if self.sampling_rate > self.sim_time:
            raise ValueError("sampling_rate must be <= sim_time")


@dataclass(frozen=True, slots=True)
class MotionPhysics:
    """Physics parameters for motion simulation."""
    gravity_z: float = -9.81


@dataclass(frozen=True, slots=True)
class MotionDebug:
    """Debug/preview settings (GUI, camera)."""
    mode: Literal["DIRECT", "GUI"] = "DIRECT"
    camera_distance: float = 6.0
    camera_yaw: float = 0.0
    camera_pitch: float = -30.0

    def __post_init__(self) -> None:
        if self.camera_distance <= 0:
            raise ValueError(f"camera_distance must be > 0, got {self.camera_distance!r}")

@dataclass(frozen=True, slots=True)
class TerrainDynamics:
    restitution: float = 0.0
    lateral_friction: float = 1.6
    rolling_friction: float = 0.03
    spinning_friction: float = 0.02
    contact_damping: float = 0.1
    contact_stiffness: float = 3000


@dataclass(frozen=True, slots=True)
class NodeDynamics:
    restitution: float = 0.0
    lateral_friction: float = 0.2
    rolling_friction: float = 0.04
    spinning_friction: float = 0.03
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    contact_damping: float = 0.05
    contact_stiffness: float = 5000

@dataclass(frozen=True, slots=True)
class MotionConfig:
    """Configuration for the motion step (pybullet)."""
    time: MotionTime = MotionTime()
    physics: MotionPhysics = MotionPhysics()
    debug: MotionDebug = MotionDebug()
    terrain: TerrainDynamics = TerrainDynamics()
    node: NodeDynamics = NodeDynamics()