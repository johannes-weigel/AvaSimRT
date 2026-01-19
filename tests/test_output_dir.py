from __future__ import annotations

from pathlib import Path

from avasimrt.app import run
from avasimrt.config import SimConfig


def test_run_returns_output_dir_as_path(tmp_path, monkeypatch) -> None:
    """
    The run should create <output>/<run_id> and return it as a Path.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    cfg = SimConfig(run_id="run-1", output=Path("output"), delete_existing=False, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is True
    assert isinstance(result.output_dir, Path)

    assert result.output_dir.exists()
    assert result.output_dir.is_dir()

    # default output base
    assert result.output_dir.parent.name == "output"
    assert result.output_dir.name == "run-1"


def test_run_fails_if_output_dir_not_empty_and_no_delete(tmp_path, monkeypatch) -> None:
    """
    If the output directory already exists and is not empty, the run must abort.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    run_dir = tmp_path / "output" / "run-2"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("data", encoding="utf-8")

    cfg = SimConfig(run_id="run-2", output=Path("output"), delete_existing=False, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is False
    assert result.output_dir is None
    assert result.message is not None
    assert "not empty" in result.message.lower()


def test_run_deletes_existing_output_dir_if_flag_set(tmp_path, monkeypatch) -> None:
    """
    If delete_existing is True, an existing non-empty directory is removed and recreated.
    The new run may only contain results.csv.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

    run_dir = tmp_path / "output" / "run-3"
    run_dir.mkdir(parents=True)
    (run_dir / "old.txt").write_text("old", encoding="utf-8")

    cfg = SimConfig(run_id="run-3", output=Path("output"), delete_existing=True, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is True
    assert isinstance(result.output_dir, Path)

    out = result.output_dir
    assert out.exists()
    assert out.is_dir()

    entries = list(out.iterdir())

    # exactly one file: results.csv
    assert len(entries) == 1
    assert entries[0].is_file()
    assert entries[0].name == "results.csv"
