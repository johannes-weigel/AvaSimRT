from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SceneConfig:
    xml_path: Path
    out_dir: Path

    def __post_init__(self) -> None:
        if not self.xml_path.exists():
            raise FileNotFoundError(f"Scene XML does not exist: {self.xml_path}")


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


@dataclass(frozen=True, slots=True)
class ChannelStateConfig:
    scene: SceneConfig
    channel: ChannelConfig = ChannelConfig()
    render: RenderConfig = RenderConfig()
    debug: bool = False
