from __future__ import annotations

from pathlib import Path

from avasimrt.app import run
from avasimrt.config import SimConfig


def test_smoke_run_returns_result(tmp_path: Path) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    
    cfg = SimConfig(scene_xml=scene_xml, scene_obj=scene_obj)
    res = run(cfg)

    assert res.successful is True
    assert isinstance(res.run_id, str) and res.run_id
    assert res.message is not None
