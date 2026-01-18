from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping
from pathlib import Path

from avasimrt.helpers import coerce_float


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionTime":
        d = dict(data or {})
        return cls(
            sim_time=coerce_float(d.get("sim_time", cls.sim_time)),
            sampling_rate=coerce_float(d.get("sampling_rate", cls.sampling_rate)),
            time_step=coerce_float(d.get("time_step", cls.time_step)),
        )


@dataclass(frozen=True, slots=True)
class MotionPhysics:
    """Physics parameters for motion simulation."""
    gravity_z: float = -9.81

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionPhysics":
        d = dict(data or {})
        return cls(
            gravity_z=coerce_float(d.get("gravity_z", cls.gravity_z)),
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionDebug":
        d = dict(data or {})
        if "camera_distance" in d:
            d["camera_distance"] = coerce_float(d["camera_distance"])
        if "camera_yaw" in d:
            d["camera_yaw"] = coerce_float(d["camera_yaw"])
        if "camera_pitch" in d:
            d["camera_pitch"] = coerce_float(d["camera_pitch"])
        return cls(**d)


@dataclass(frozen=True, slots=True)
class TerrainDynamics:
    restitution: float = 0.0
    lateral_friction: float = 1.6
    rolling_friction: float = 0.03
    spinning_friction: float = 0.02
    contact_damping: float = 0.1
    contact_stiffness: float = 3000.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TerrainDynamics":
        d = dict(data or {})
        for k, v in list(d.items()):
            d[k] = coerce_float(v)
        return cls(**d)


@dataclass(frozen=True, slots=True)
class NodeDynamics:
    restitution: float = 0.0
    lateral_friction: float = 0.2
    rolling_friction: float = 0.04
    spinning_friction: float = 0.03
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    contact_damping: float = 0.05
    contact_stiffness: float = 5000.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NodeDynamics":
        d = dict(data or {})
        for k, v in list(d.items()):
            d[k] = coerce_float(v)
        return cls(**d)


@dataclass(frozen=True, slots=True)
class MotionSceneConfig:
    obj_path: Path | None = None

    def __post_init__(self) -> None:
        if self.obj_path is not None and not self.obj_path.exists():
            raise FileNotFoundError(f"Scene OBJ does not exist: {self.obj_path}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionSceneConfig":
        d = dict(data or {})
        obj_path = d.get("obj_path")
        return cls(
            obj_path=Path(obj_path) if obj_path else None,
        )


@dataclass(frozen=True, slots=True)
class MotionConfig:
    """Configuration for the motion step (pybullet)."""
    scene: MotionSceneConfig = MotionSceneConfig()
    time: MotionTime = MotionTime()
    physics: MotionPhysics = MotionPhysics()
    debug: MotionDebug = MotionDebug()
    terrain: TerrainDynamics = TerrainDynamics()
    node: NodeDynamics = NodeDynamics()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotionConfig":
        m = dict(data or {})

        scene = MotionSceneConfig.from_dict(m.get("scene"))
        time = MotionTime.from_dict(m.get("time"))
        physics = MotionPhysics.from_dict(m.get("physics"))
        debug = MotionDebug.from_dict(m.get("debug"))
        terrain = TerrainDynamics.from_dict(m.get("terrain"))
        node = NodeDynamics.from_dict(m.get("node"))

        return cls(
            scene=scene,
            time=time,
            physics=physics,
            debug=debug,
            terrain=terrain,
            node=node,
        )
