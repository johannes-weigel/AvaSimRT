from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from avasimrt.helpers import coerce_int, coerce_float, coerce_bool


@dataclass(frozen=True, slots=True)
class SnowMaterialConfig:
    """Radio material properties for snow objects."""

    itu_type: str = "wet_ground"
    thickness: float = 0.1
    scattering_coefficient: float = 0.5

    def __post_init__(self) -> None:
        if self.thickness <= 0:
            raise ValueError("thickness must be > 0")
        if not 0 <= self.scattering_coefficient <= 1:
            raise ValueError("scattering_coefficient must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SnowMaterialConfig":
        d = dict(data or {})
        defaults = cls()
        return cls(
            itu_type=str(d.get("itu_type", defaults.itu_type)),
            thickness=coerce_float(d.get("thickness", defaults.thickness)),
            scattering_coefficient=coerce_float(
                d.get("scattering_coefficient", defaults.scattering_coefficient)
            ),
        )


@dataclass(frozen=True, slots=True)
class SnowConfig:
    """Configuration for snow layer in channel simulation."""

    enabled: bool = False
    box_size: float = 1.0
    levels: int = 1
    margin: float = 10.0
    material: SnowMaterialConfig = SnowMaterialConfig()

    def __post_init__(self) -> None:
        if self.box_size <= 0:
            raise ValueError("box_size must be > 0")
        if self.levels < 1:
            raise ValueError("levels must be >= 1")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SnowConfig":
        d = dict(data or {})
        defaults = cls()
        return cls(
            enabled=coerce_bool(d.get("enabled", defaults.enabled)),
            box_size=coerce_float(d.get("box_size", defaults.box_size)),
            levels=coerce_int(d.get("levels", defaults.levels)),
            margin=coerce_float(d.get("margin", defaults.margin)),
            material=SnowMaterialConfig.from_dict(d.get("material")),
        )


@dataclass(frozen=True, slots=True)
class ChannelConfig:
    freq_center: float = 3.8e9
    sc_num: int = 101
    sc_spacing: float = 5e6
    reflection_depth: int = 3
    seed: int = 41

    def __post_init__(self) -> None:
        if self.freq_center <= 0:
            raise ValueError("freq_center must be > 0")
        if self.sc_num <= 0:
            raise ValueError("sc_num must be > 0")
        if self.sc_spacing <= 0:
            raise ValueError("sc_spacing must be > 0")
        if self.reflection_depth < 0:
            raise ValueError("reflection_depth must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ChannelConfig":
        d = dict(data or {})
        return cls(
            freq_center=coerce_float(d.get("freq_center", 3.8e9)),
            sc_num=coerce_int(d.get("sc_num", 101)),
            sc_spacing=coerce_float(d.get("sc_spacing", 5e6)),
            reflection_depth=coerce_int(d.get("reflection_depth", 3)),
            seed=coerce_int(d.get("seed", 41)),
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    enabled: bool = False
    every_n_steps: int = 0
    width: int = 640
    height: int = 360
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_z: float = 100.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")
        if self.every_n_steps < 0:
            raise ValueError("every_n_steps must be >= 0")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RenderConfig":
        d = dict(data or {})
        defaults = cls()

        return cls(
            enabled=coerce_bool(d.get("enabled", defaults.enabled)),
            every_n_steps=coerce_int(d.get("every_n_steps", defaults.every_n_steps)),
            width=coerce_int(d.get("width", defaults.width)),
            height=coerce_int(d.get("height", defaults.height)),
            camera_x=coerce_float(d.get("camera_x", defaults.camera_x)),
            camera_y=coerce_float(d.get("camera_y", defaults.camera_y)),
            camera_z=coerce_float(d.get("camera_z", defaults.camera_z)),
        )


@dataclass(frozen=True, slots=True)
class ChannelStateConfig:
    channel: ChannelConfig = ChannelConfig()
    render: RenderConfig = RenderConfig()
    snow: SnowConfig = SnowConfig()
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelStateConfig":
        d = dict(data)

        channel = ChannelConfig.from_dict(d.get("channel"))
        render = RenderConfig.from_dict(d.get("render"))
        snow = SnowConfig.from_dict(d.get("snow"))
        debug = coerce_bool(d.get("debug", False))

        return cls(
            channel=channel,
            render=render,
            snow=snow,
            debug=debug,
        )
