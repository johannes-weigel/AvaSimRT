from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import json

from avasimrt.config import PositionConfig
from .output import prepare_output
from .blender import run_blender_export
from .heights import generate_heightmap, resolve_positions
from .result import PreprocessorResult

logger = logging.getLogger(__name__)


def prepare(*,
            run_id: str | None = None,
            out_base: Path | None = None,
            delete_existing: bool = False,
            scene_blender: Path | None = None,
            scene_obj: Path | None = None,
            scene_xml: Path | None = None,
            blender_cmd: str | None = None,
            nodes: Sequence[PositionConfig],
            anchors: Sequence[PositionConfig],
            heightmap_npy: Path | None = None,
            heightmap_resolution: float | None = None,) -> PreprocessorResult:
    # Validate input
    has_blender = scene_blender is not None
    has_obj_xml = scene_obj is not None and scene_xml is not None
    
    if not has_blender and not has_obj_xml:
        raise ValueError("Must provide either scene_blender or both scene_obj and scene_xml")
    if has_blender and has_obj_xml:
        raise ValueError("Cannot provide both scene_blender and scene_obj/scene_xml")

    # OUT
    if (out_base is None):
        out_base = Path("output")
    out_dir, final_run_id = prepare_output(out_base, run_id, delete_existing)

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
            ply_output=final_ply,
            blender_cmd=blender_cmd)
        logger.info(f"Scene exported to: {out_dir}")

    else:
        assert scene_obj is not None and scene_xml is not None
        logger.info(f"Copying scene files to: {out_dir}")
        
        try:
            shutil.copy2(scene_obj, final_obj)
            shutil.copy2(scene_xml, final_xml)
        except OSError as e:
            raise OSError(
                f"Failed to copy scene files to '{out_dir}': {e}"
            ) from e
    

    # HEIGHT MAP
    heightmap = None
    if heightmap_npy is not None:
        logger.info(f"Loading heightmap from: {heightmap_npy}")
        try:
            heightmap = np.load(heightmap_npy)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load heightmap from '{heightmap_npy}': {e}"
            ) from e
    
    if heightmap is None:
        logger.info("Generating heightmap from scene geometry")
        try:
            heightmap, heightmap_metadata = generate_heightmap(final_obj, heightmap_resolution)
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate heightmap from '{final_obj}': {e}"
            ) from e
        
        try:
            with open(out_dir / "heightmap_meta.json", 'w', encoding='utf-8') as f:
                json.dump(heightmap_metadata, f, indent=2)
        except OSError as e:
            raise OSError(
                f"Failed to save heightmap metadata to '{out_dir}': {e}"
            ) from e
    else:
        logger.info("Using provided heightmap (not generating new one)")
    
    try:
        np.save(out_dir / "heightmap.npy", heightmap)
    except Exception as e:
        raise OSError(
            f"Failed to save heightmap to '{out_dir / 'heightmap.npy'}': {e}"
        ) from e


    # NODE & ANCHORS
    logger.info("Resolving node and anchor positions")
    try:
        resolved_nodes, resolved_anchors = resolve_positions(final_obj, nodes, anchors)
    except Exception as e:
        raise RuntimeError(
            f"Failed to resolve positions from scene '{final_obj}': {e}"
        ) from e

    positions_data = {
        "nodes": [
            {"id": n.id, "x": n.x, "y": n.y, "z": n.z, "size": n.size}
            for n in resolved_nodes
        ],
        "anchors": [
            {"id": a.id, "x": a.x, "y": a.y, "z": a.z, "size": a.size}
            for a in resolved_anchors
        ],
    }
    try:
        with open(out_dir / "positions_resolved.json", 'w', encoding='utf-8') as f:
            json.dump(positions_data, f, indent=2)
            f.write('\n')
    except OSError as e:
        raise OSError(
            f"Failed to save resolved positions to '{out_dir}': {e}"
        ) from e

    return PreprocessorResult(
        run_id=final_run_id,
        nodes=resolved_nodes,
        anchors=resolved_anchors,
        heightmap=heightmap,
        out_dir=out_dir,
        scene_obj=final_obj,
        scene_xml=final_xml,
    )
