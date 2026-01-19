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
    node: ResolvedPosition
    anchors: Sequence[ResolvedPosition]
    heightmap: np.ndarray