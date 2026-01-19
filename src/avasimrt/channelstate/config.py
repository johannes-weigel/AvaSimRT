from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

from avasimrt.helpers import coerce_int, coerce_float, coerce_bool


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
            freq_center=coerce_float(d.get("freq_center", cls.freq_center)),
            sc_num=coerce_int(d.get("sc_num", cls.sc_num)),
            sc_spacing=coerce_float(d.get("sc_spacing", cls.sc_spacing)),
            reflection_depth=coerce_int(d.get("reflection_depth", cls.reflection_depth)),
            seed=coerce_int(d.get("seed", cls.seed)),
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    enabled: bool = False
    every_n_steps: int = 0
    width: int = 650
    height: int = 500
    distance: float = 6.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be > 0")
        if self.distance <= 0:
            raise ValueError("distance must be > 0")
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
            distance=coerce_float(d.get("distance", defaults.distance)),
        )


@dataclass(frozen=True, slots=True)
class ChannelStateConfig:
    channel: ChannelConfig = ChannelConfig()
    render: RenderConfig = RenderConfig()
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChannelStateConfig":
        d = dict(data)


        channel = ChannelConfig.from_dict(d.get("channel"))
        render = RenderConfig.from_dict(d.get("render"))
        debug = coerce_bool(d.get("debug", cls.debug))

        return cls(
            channel=channel,
            render=render,
            debug=debug,
        )
