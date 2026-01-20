from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import json

from avasimrt.config import AnchorConfig, NodeConfig
from .blender import run_blender_export
from .heights import generate_heightmap, resolve_positions
from .result import PreprocessorResult

logger = logging.getLogger(__name__)


def prepare(*,
            out_dir: Path,
            scene_blender: Path | None = None,
            scene_obj: Path | None = None,
            scene_xml: Path | None = None,
            node: NodeConfig,
            anchors: Sequence[AnchorConfig],
            heightmap: np.ndarray | None = None,) -> PreprocessorResult:
    # Validate input
    has_blender = scene_blender is not None
    has_obj_xml = scene_obj is not None and scene_xml is not None
    
    if not has_blender and not has_obj_xml:
        raise ValueError("Must provide either scene_blender or both scene_obj and scene_xml")
    if has_blender and has_obj_xml:
        raise ValueError("Cannot provide both scene_blender and scene_obj/scene_xml")
    

    # SCENE FILES
    final_obj = out_dir / "scene.obj"
    final_xml = out_dir / "scene.xml"
    final_ply = out_dir / "scene.ply"
    
    if has_blender:
        assert scene_blender is not None
        logger.info(f"Exporting scene from Blender: {scene_blender}")
        run_blender_export(
            blend_path=scene_blender,
            obj_output=final_obj,
            xml_output=final_xml,
            ply_output=final_ply)
        logger.info(f"Scene exported to: {out_dir}")

    else:
        assert scene_obj is not None and scene_xml is not None
        logger.info(f"Copying scene files to: {out_dir}")
        shutil.copy2(scene_obj, final_obj)
        shutil.copy2(scene_xml, final_xml)
    

    # HEIGHT MAP
    if heightmap is None:
        heightmap, heightmap_metadata = generate_heightmap(final_obj, 0.1)
        with open(out_dir/"heightmap_meta.json", 'w') as f:
            json.dump(heightmap_metadata, f, indent=2)
    
    np.save(out_dir/"heightmap.npy", heightmap)


    # NODE & ANCHORS
    resolved_node, resolved_anchors = resolve_positions(final_obj, node, anchors)
    
    
    return PreprocessorResult(
        node=resolved_node, 
        anchors=resolved_anchors,
        heightmap=heightmap,
        scene_obj=final_obj,
        scene_xml=final_xml,)
