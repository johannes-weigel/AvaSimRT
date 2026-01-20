from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, NamedTuple

import numpy as np


@dataclass
class ResolvedPosition:
    """A position with resolved z coordinate."""
    id: str
    x: float
    y: float
    z: float
    size: float

class PreprocessorResult(NamedTuple):
    nodes: Sequence[ResolvedPosition]
    anchors: Sequence[ResolvedPosition]
    heightmap: np.ndarray
    out_dir: Path
    scene_obj: Path
    scene_xml: Path
    run_id: str
