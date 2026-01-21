from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from avasimrt.helpers import coerce_float, coerce_all_floats


@dataclass(frozen=True, slots=True)
class MotionTime:
    """Time settings for the motion simulation."""
    sim_time: float = 120.0
    sampling_rate: float = 0.5
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
            sim_time=coerce_float(d.get("sim_time", 60.0)),
            sampling_rate=coerce_float(d.get("sampling_rate", 1.0)),
            time_step=coerce_float(d.get("time_step", 1.0 / 240.0)),
        )


@dataclass(frozen=True, slots=True)
class MotionPhysics:
    """Physics parameters for motion simulation."""
    gravity_z: float = -9.81

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "MotionPhysics":
        d = dict(data or {})
        return cls(
            gravity_z=coerce_float(d.get("gravity_z", -9.81)),
        )


@dataclass(frozen=True, slots=True)
class MotionDebug:
    """Debug/preview settings (GUI, camera)."""
    mode: Literal["DIRECT", "GUI"] = "DIRECT"
    camera_distance: float = 100.0
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
    lateral_friction: float = 0.1
    rolling_friction: float = 0.001
    spinning_friction: float = 0.001
    contact_damping: float = 500.0
    contact_stiffness: float = 30000.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "TerrainDynamics":
        return cls(**coerce_all_floats(dict(data or {})))


@dataclass(frozen=True, slots=True)
class NodeDynamics:
    restitution: float = 0.0
    lateral_friction: float = 0.05
    rolling_friction: float = 0.001
    spinning_friction: float = 0.001
    linear_damping: float = 0.001
    angular_damping: float = 0.001
    contact_damping: float = 500.0
    contact_stiffness: float = 30000.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "NodeDynamics":
        return cls(**coerce_all_floats(dict(data or {})))

@dataclass(frozen=True, slots=True)
class MotionConfig:
    """Configuration for the motion step (pybullet)."""
    time: MotionTime = MotionTime()
    physics: MotionPhysics = MotionPhysics()
    debug: MotionDebug = MotionDebug()
    terrain: TerrainDynamics = TerrainDynamics()
    node: NodeDynamics = NodeDynamics()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotionConfig":
        m = dict(data or {})

        time = MotionTime.from_dict(m.get("time") or {})
        physics = MotionPhysics.from_dict(m.get("physics"))
        debug = MotionDebug.from_dict(m.get("debug"))
        terrain = TerrainDynamics.from_dict(m.get("terrain"))
        node = NodeDynamics.from_dict(m.get("node"))

        return cls(
            time=time,
            physics=physics,
            debug=debug,
            terrain=terrain,
            node=node,
        )
