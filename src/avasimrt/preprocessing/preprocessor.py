from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence


from avasimrt.config import AnchorConfig, NodeConfig
from .heights import resolve_positions
from .result import PreprocessorResult

logger = logging.getLogger(__name__)


def prepare(*,
            scene_obj: Path,
            node: NodeConfig,
            anchors: Sequence[AnchorConfig]) -> PreprocessorResult:
    
    
    
    resolved_node, resolved_anchors = resolve_positions(scene_obj, node, anchors)
    
    
    return PreprocessorResult(node=resolved_node, anchors=resolved_anchors)
