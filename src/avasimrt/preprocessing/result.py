from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, NamedTuple

import numpy as np

from avasimrt.config import PositionConfig


@dataclass
class ResolvedPosition:
    """A position with resolved z coordinate."""
    id: str
    x: float
    y: float
    z: float
    z_terrain: float | None
    size: float

    @classmethod
    def from_config(cls, config: PositionConfig, z: float, z_terrain: float | None) -> "ResolvedPosition":
        return cls(
            id=config.id,
            x=config.x,
            y=config.y,
            z=z,
            z_terrain=z_terrain,
            size=config.size,
        )

class PreprocessorResult(NamedTuple):
    nodes: Sequence[ResolvedPosition]
    anchors: Sequence[ResolvedPosition]
    heightmap: np.ndarray
    out_dir: Path
    scene_obj: Path
    scene_xml: Path
    run_id: str
