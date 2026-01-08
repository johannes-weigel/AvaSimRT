from __future__ import annotations

from pathlib import Path

from avasimrt.cli import CliArgs, resolve_config
from avasimrt.config import SimConfig


def test_resolve_config_from_cli_flags() -> None:
    args = CliArgs(config=None, run_id="r1", output="out", delete_existing=True)
    cfg = resolve_config(args)

    assert isinstance(cfg, SimConfig)
    assert cfg.run_id == "r1"
    assert cfg.output == "out"
    assert cfg.delete_existing is True


def test_resolve_config_generates_run_id_when_not_given() -> None:
    args = CliArgs(config=None, run_id=None, output="out", delete_existing=False)
    cfg = resolve_config(args)

    assert cfg.output == "out"
    assert cfg.delete_existing is False
    assert isinstance(cfg.run_id, str) and cfg.run_id


def test_resolve_config_uses_config_file_and_ignores_other_flags(tmp_path: Path, monkeypatch) -> None:
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text(
        "run_id: from-file\noutput: file-out\ndelete_existing: false\n",
        encoding="utf-8",
    )

    args = CliArgs(
        config=str(cfg_file),
        run_id="ignored",
        output="ignored",
        delete_existing=True,
    )
    cfg = resolve_config(args)

    assert cfg.run_id == "from-file"
    assert cfg.output == "file-out"
    assert cfg.delete_existing is False
