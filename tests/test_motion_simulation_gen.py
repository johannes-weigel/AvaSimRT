# tests/test_motion_simulation.py
from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Sequence

import pytest

from avasimrt.motion.config import (
    MotionConfig,
    MotionDebug,
    MotionPhysics,
    MotionTime,
)
from avasimrt.motion import simulation as ms
from avasimrt.preprocessing.result import ResolvedPosition


def _cfg(*, sim_time: float = 2.01, sampling_rate: float = 1.0, time_step: float = 0.01) -> MotionConfig:
    return MotionConfig(
        time=MotionTime(sim_time=sim_time, sampling_rate=sampling_rate, time_step=time_step),
        physics=MotionPhysics(gravity_z=-9.81),
        debug=MotionDebug(mode="DIRECT"),
    )


def test_simulate_motion_plane_collects_three_samples(tmp_path: Path) -> None:
    cfg = _cfg(sim_time=2.01, sampling_rate=1.0, time_step=0.01)

    node = ResolvedPosition(id="NODE", x=0.0, y=0.0, z=1.0, size=0.1)
    
    scene_obj = tmp_path / "plane.obj"
    scene_obj.write_text("v -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nf 1 2 3\nf 1 3 4\n", encoding="utf-8")

    results = ms.simulate_motion(
        cfg=cfg,
        node=node,
        scene_obj=scene_obj,
    )

    assert len(results) == 3  # t≈0,1,2

    ts = [r.timestamp for r in results]
    assert ts[0] == pytest.approx(0.0, abs=1e-6)
    assert ts[1] == pytest.approx(1.0, abs=0.02)
    assert ts[2] == pytest.approx(2.0, abs=0.02)

    for r in results:
        pos = r.node.position
        orn = r.node.orientation
        vel = r.node.linear_velocity

        assert isinstance(pos, Sequence) and len(pos) == 3
        assert isinstance(orn, Sequence) and len(orn) == 4
        assert isinstance(vel, Sequence) and len(vel) == 3

        assert all(isfinite(float(x)) for x in pos)
        assert all(isfinite(float(x)) for x in orn)
        assert all(isfinite(float(x)) for x in vel)


def test_simulate_motion_single_sample(tmp_path: Path) -> None:
    """Test that a very short simulation produces at least one sample."""
    cfg = _cfg(sim_time=0.05, sampling_rate=0.05, time_step=0.01)

    node = ResolvedPosition(id="NODE", x=0.0, y=0.0, z=1.0, size=0.1)
    
    scene_obj = tmp_path / "plane.obj"
    scene_obj.write_text("v -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nf 1 2 3\nf 1 3 4\n", encoding="utf-8")

    results = ms.simulate_motion(
        cfg=cfg,
        node=node,
        scene_obj=scene_obj,
    )

    assert len(results) >= 1
    assert all(r.timestamp >= 0.0 for r in results)
