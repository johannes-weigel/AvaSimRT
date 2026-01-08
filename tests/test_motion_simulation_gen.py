# tests/test_motion_simulation.py
from __future__ import annotations

from math import isfinite
from typing import Sequence

import pytest

from avasimrt.config import AnchorConfig, NodeConfig
from avasimrt.motion.config import (
    MotionConfig,
    MotionDebug,
    MotionPhysics,
    MotionTime,
)
from avasimrt.motion import simulation as ms


def _cfg(*, sim_time: float = 2.01, sampling_rate: float = 1.0, time_step: float = 0.01) -> MotionConfig:
    return MotionConfig(
        time=MotionTime(sim_time=sim_time, sampling_rate=sampling_rate, time_step=time_step),
        physics=MotionPhysics(gravity_z=-9.81),
        debug=MotionDebug(mode="DIRECT"),
    )


def test_simulate_motion_plane_collects_three_samples() -> None:
    cfg = _cfg(sim_time=2.01, sampling_rate=1.0, time_step=0.01)

    node = NodeConfig(x=0.0, y=0.0, z=1.0, size=0.1)

    results, resolved = ms.simulate_motion(
        cfg=cfg,
        node=node,
        anchors=(),
        terrain_mesh=None,
        use_plane_if_no_mesh=True,
    )

    assert resolved == []
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


def test_simulate_motion_resolves_anchor_heights_without_mutation() -> None:
    cfg = _cfg(sim_time=0.05, sampling_rate=0.05, time_step=0.01)

    anchors = [
        AnchorConfig(id="A-01", x=0.0, y=0.0, z=None, size=0.2),
        AnchorConfig(id="A-02", x=1.0, y=0.0, z=5.0, size=0.2),
    ]
    node = NodeConfig(x=0.0, y=0.0, z=1.0, size=0.1)

    _results, resolved = ms.simulate_motion(
        cfg=cfg,
        node=node,
        anchors=anchors,
        terrain_mesh=None,
        use_plane_if_no_mesh=True,
    )

    assert len(resolved) == 2
    assert resolved[0][0] == "A-01"
    assert resolved[1][0] == "A-02"

    # A-01: resolved ~ plane height + size
    assert isfinite(resolved[0][3])
    assert resolved[0][3] == pytest.approx(0.0 + anchors[0].size, abs=1e-2)

    # A-02: preserved
    assert resolved[1][3] == pytest.approx(5.0, abs=1e-9)

    # No mutation of original configs
    assert anchors[0].z is None
    assert anchors[1].z == 5.0


def test_height_on_terrain_hits_plane_near_zero() -> None:
    # Test the helper directly with a real pybullet session, but keep it minimal.
    cfg = _cfg(sim_time=0.01, sampling_rate=0.01, time_step=0.01)

    p = ms._load_pybullet()
    ms._connect(p, cfg)
    try:
        terrain_id = ms._load_terrain_urdf(p, "plane.urdf")
        h = ms.height_on_terrain(p, x=0.0, y=0.0, terrain_id=terrain_id, z_start=10.0, z_end=-10.0)
        assert abs(h - 0.0) < 1e-3
    finally:
        ms._disconnect(p)
