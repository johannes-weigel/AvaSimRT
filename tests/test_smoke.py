from __future__ import annotations

from avasimrt.app import run
from avasimrt.config import SimConfig


def test_smoke_run_returns_result() -> None:
    cfg = SimConfig()
    res = run(cfg)

    assert res.successful is True
    assert isinstance(res.run_id, str) and res.run_id
    assert res.message is not None
