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
        node="0,0,1,0.2",
        anchors=["A-01,0,0,none,0.2"],
        scene_xml=None,
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
    )


def test_resolve_config_from_cli_flags() -> None:
    args = _base_args(run_id="r1", output="out")
    cfg = resolve_config(args)

    assert isinstance(cfg, SimConfig)
    assert cfg.run_id == "r1"
    assert cfg.output == "out"
    assert cfg.delete_existing is True

    assert cfg.node is not None
    assert len(cfg.anchors) == 1
    assert cfg.motion is not None


def test_resolve_config_generates_run_id_when_not_given() -> None:
    args = _base_args(run_id=None, output="out")
    cfg = resolve_config(args)

    assert cfg.output == "out"
    assert cfg.delete_existing is True
    assert isinstance(cfg.run_id, str) and cfg.run_id


def test_resolve_config_uses_config_file_and_ignores_other_flags(tmp_path: Path) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(
        "\n".join(
            [
                "run_id: from-file",
                "output: file-out",
                "delete_existing: false",
                "node:",
                "  x: 0.0",
                "  y: 0.0",
                "  z: 1.0",
                "  size: 0.2",
                "anchors:",
                "  - id: A-01",
                "    x: 0.0",
                "    y: 0.0",
                "    z: 0.2",
                "    size: 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    args = _base_args(run_id="ignored", output="ignored")
    args = replace(args, config=str(cfg_file), delete_existing=True)

    cfg = resolve_config(args)

    assert cfg.run_id == "from-file"
    assert cfg.output == "file-out"
    assert cfg.delete_existing is False


def test_resolve_config_requires_node_when_no_config() -> None:
    args = _base_args()
    args = replace(args, node=None)

    with pytest.raises(ValueError):
        resolve_config(args)


def test_resolve_config_requires_at_least_one_anchor_when_no_config() -> None:
    args = _base_args()
    args = replace(args, anchors=[])

    with pytest.raises(ValueError):
        resolve_config(args)


def test_resolve_config_creates_channelstate_when_scene_xml_given(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text("dummy", encoding="utf-8")

    args = _base_args()
    args = replace(args, scene_xml=str(scene), render=True, render_every_n=2)

    cfg = resolve_config(args)

    assert cfg.channelstate is not None
    assert cfg.channelstate.render.enabled is True
    assert cfg.channelstate.render.every_n_steps == 2
