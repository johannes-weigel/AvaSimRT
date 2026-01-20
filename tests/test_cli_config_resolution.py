from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import pytest

from avasimrt.cli import CliArgs, resolve_config
from avasimrt.config import SimConfig


def _base_args(*, output: str = "out", run_id: str | None = "r1") -> CliArgs:
    return CliArgs(
        config=None,
        run_id=run_id,
        output=output,
        delete_existing=True,
        debug=False,
        nodes=["Node-1,0,0,1,0.2"],
        anchors=["A-01,0,0,none,0.2"],
        scene_xml=None,
        scene_obj=None,
        scene_blender=None,
        blender_cmd=None,
        sim_time=1.0,
        sampling_rate=1.0,
        time_step=0.01,
        freq_center=3.8e9,
        sc_num=5,
        sc_spacing=5e6,
        reflection_depth=1,
        seed=1,
        render=False,
        render_every_n=0,
        interactive_plots=False,
        save_all_plots=False,
        no_csv=False,
        heightmap_npy=None,
        heightmap_resolution=None
    )


def test_resolve_config_from_cli_flags() -> None:
    args = _base_args(run_id="r1", output="out")
    cfg = resolve_config(args)

    assert isinstance(cfg, SimConfig)
    assert cfg.run_id == "r1"
    assert cfg.output == Path("out")
    assert cfg.delete_existing is True

    assert len(cfg.nodes) == 1
    assert len(cfg.anchors) == 1
    assert cfg.motion is not None


def test_resolve_config_generates_run_id_when_not_given() -> None:
    args = _base_args(run_id=None, output="out")
    cfg = resolve_config(args)

    assert cfg.output == Path("out")
    assert cfg.delete_existing is True
    # run_id is None in config - it gets generated later in preprocessing
    assert cfg.run_id is None


def test_resolve_config_uses_config_file_and_ignores_other_flags(tmp_path: Path) -> None:
    # Create dummy scene files
    xml_file = tmp_path / "scene.xml"
    xml_file.write_text("<scene/>", encoding="utf-8")
    obj_file = tmp_path / "scene.obj"
    obj_file.write_text("dummy", encoding="utf-8")
    
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(
        "\n".join(
            [
                "run_id: from-file",
                "output: file-out",
                "delete_existing: false",
                f"xml: {xml_file}",
                f"obj: {obj_file}",
                "nodes:",
                "  - id: N-1",
                "    x: 0.0",
                "    y: 0.0",
                "    z: 1.0",
                "    size: 0.2",
                "anchors:",
                "  - id: A-01",
                "    x: 0.0",
                "    y: 0.0",
                "    z: 0.2",
                "    size: 0.2",
                "motion:",
                "  time:",
                "    sim_time: 60.0",
                "    sampling_rate: 1.0",
                "    time_step: 0.004166",
                "  physics:",
                "    gravity_z: -9.81",
                "  debug:",
                "    mode: DIRECT",
                "reporting:",
                "  enabled: true",
                "  csv: true",
                "visualization:",
                "  interactive_plots: false",
                "  save_all_plots: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = _base_args(run_id="ignored", output="ignored")
    args = replace(args, config=str(cfg_file), delete_existing=True)

    cfg = resolve_config(args)

    assert cfg.run_id == "from-file"
    assert cfg.output == Path("file-out")
    assert cfg.delete_existing is False



def test_resolve_config_creates_channelstate_when_scene_xml_given(tmp_path: Path) -> None:
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("dummy", encoding="utf-8")
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("dummy", encoding="utf-8")

    args = _base_args()
    args = replace(args, scene_xml=str(scene_xml), scene_obj=str(scene_obj), render=True, render_every_n=2)

    cfg = resolve_config(args)

    assert cfg.channelstate is not None
    assert cfg.channelstate.render.enabled is True
    assert cfg.channelstate.render.every_n_steps == 2
    assert cfg.scene_xml == scene_xml
    assert cfg.scene_obj == scene_obj
