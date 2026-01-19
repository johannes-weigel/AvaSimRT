from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import json

from avasimrt.config import AnchorConfig, NodeConfig
from .heights import generate_heightmap, resolve_positions
from .result import PreprocessorResult

logger = logging.getLogger(__name__)


def prepare(*,
            out_dir: Path,
            scene_obj: Path,
            node: NodeConfig,
            anchors: Sequence[AnchorConfig],
            heightmap: np.ndarray | None = None,) -> PreprocessorResult:
    
    # HEIGHT MAP
    if (heightmap is None):
        heightmap, heightmap_metadata = generate_heightmap(scene_obj, 1)
        with open(out_dir/"heightmap_meta.json", 'w') as f:
            json.dump(heightmap_metadata, f, indent=2)
    np.save(out_dir/"heightmap.npy", heightmap)

    # NODE & ANCHORS
    resolved_node, resolved_anchors = resolve_positions(scene_obj, node, anchors)
    
    return PreprocessorResult(
        node=resolved_node, 
        anchors=resolved_anchors,
        heightmap=heightmap)
