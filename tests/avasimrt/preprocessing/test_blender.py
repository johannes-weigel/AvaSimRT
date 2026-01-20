from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import pytest

from avasimrt.preprocessing.blender import (
    _get_blender_export_script, 
    _resolve_blender_cmd,
    run_blender_export
)


logger = logging.getLogger(__name__)


def test_generate_script_without_errors() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        obj_out = tmp_dir/"test.obj"
        xml_out = tmp_dir/"test.xml"
        ply_out = tmp_dir/"test.ply"

        script = _get_blender_export_script(obj_out, xml_out, ply_out)
        print(script)


def test_resolve_cmd_from_env() -> None:
    actual = _resolve_blender_cmd(None)
    if (not actual.find("4.2.15") and not actual.endswith("/blender")):
        pytest.skip("Contains information from uncommitted .env file, allowed to fail if changed explizit.")


def test_resolve_cmd_explicit_override() -> None:
    actual = _resolve_blender_cmd('custom-blender')
    assert actual == 'custom-blender'

def test_run_export(examples: Path) -> None:
    examples_dir = examples / "nordkette" / "assets" / "scene.blend"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tmp_dir = Path(tempfile.gettempdir()) / f"avasimrt_test_blender_{timestamp}"
    tmp_dir.mkdir(exist_ok=True)

    logger.info(f"Creating blender exports in {tmp_dir}")

    obj_out = tmp_dir/"test.obj"
    xml_out = tmp_dir/"test.xml"
    ply_out = tmp_dir/"test.ply"

    run_blender_export(blend_path=examples_dir, obj_output=obj_out, xml_output=xml_out, ply_output=ply_out)

    assert obj_out.exists(), f"OBJ file not found in {tmp_dir}"
    assert xml_out.exists(), f"XML file not found in {tmp_dir}"
    assert ply_out.exists(), f"PLY file not found in {tmp_dir}"
