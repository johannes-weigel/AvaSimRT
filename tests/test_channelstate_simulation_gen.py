from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from avasimrt.config import AnchorConfig
from avasimrt.channelstate.config import (
    ChannelStateConfig,
    ChannelConfig,
    RenderConfig,
    SceneConfig,
)
import avasimrt.channelstate.simulation as cs

from avasimrt.result import Sample, NodeSnapshot


# -----------------------------
# Fakes used by monkeypatching
# -----------------------------

@dataclass
class FakeRx:
    position: Any = None


@dataclass
class FakeTx:
    name: str

    def look_at(self, rx: FakeRx) -> None:
        # just record that we were called by setting a flag
        setattr(self, "_looked_at", True)


@dataclass
class FakeCtx:
    rx: FakeRx
    txs: list[FakeTx]


def _cfg(tmp_path: Path, *, render_enabled: bool = False, every_n: int = 0) -> ChannelStateConfig:
    # xml must exist due to __post_init__ validation
    xml = tmp_path / "scene.xml"
    xml.write_text("<scene/>", encoding="utf-8")

    return ChannelStateConfig(
        scene=SceneConfig(xml_path=xml, out_dir=tmp_path / "frames"),
        channel=ChannelConfig(freq_center=3.8e9, sc_num=5, sc_spacing=5e6, reflection_depth=1, seed=1),
        render=RenderConfig(enabled=render_enabled, every_n_steps=every_n),
        debug=False,
    )


def _motion_results() -> list[Sample]:
    return [
        Sample(timestamp=0.0, node=NodeSnapshot(position=(0.0, 0.0, 1.0), orientation=(1, 0, 0, 0), linear_velocity=(0, 0, 0))),
        Sample(timestamp=1.0, node=NodeSnapshot(position=(1.0, 0.0, 1.0), orientation=(1, 0, 0, 0), linear_velocity=(0, 0, 0))),
        Sample(timestamp=2.0, node=NodeSnapshot(position=(2.0, 0.0, 1.0), orientation=(1, 0, 0, 0), linear_velocity=(0, 0, 0))),
    ]


def _anchors_with_z() -> list[AnchorConfig]:
    return [
        AnchorConfig(id="A-01", x=0.0, y=0.0, z=0.2, size=0.2),
        AnchorConfig(id="A-02", x=5.0, y=0.0, z=0.2, size=0.2),
    ]


# -----------------------------
# Tests
# -----------------------------

def test_estimate_channelstate_empty_motion_results_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    anchors = _anchors_with_z()

    # Ensure no expensive setup accidentally happens
    called = {"build": 0}

    def fake_build_context(*args, **kwargs):
        called["build"] += 1
        return FakeCtx(rx=FakeRx(), txs=[FakeTx("A-01")])

    monkeypatch.setattr(cs, "_build_context", fake_build_context)

    out = cs.estimate_channelstate(cfg=cfg, anchors=anchors, motion_results=[])

    assert out == []
    assert called["build"] == 0


def test_estimate_channelstate_preserves_timestamps_and_sets_rx_position(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path)
    anchors = _anchors_with_z()
    motion = _motion_results()

    ctx = FakeCtx(rx=FakeRx(), txs=[FakeTx(a.id) for a in anchors])

    monkeypatch.setattr(cs, "_build_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(cs, "_solve_paths", lambda *args, **kwargs: object())

    fake_readings = ["READINGS"]
    calls = {"eval": 0}

    def fake_eval(*, paths, cfg, anchors, node_snapshot):
        calls["eval"] += 1
        assert node_snapshot.position in [m.node.position for m in motion]
        return fake_readings

    monkeypatch.setattr(cs, "_evaluate_cfr", fake_eval)
    monkeypatch.setattr(cs, "_render_if_enabled", lambda **kwargs: None)

    out = cs.estimate_channelstate(cfg=cfg, anchors=anchors, motion_results=motion)

    assert len(out) == 3
    assert [r.timestamp for r in out] == [0.0, 1.0, 2.0]
    assert all(r.readings == fake_readings for r in out)
    assert calls["eval"] == 3

    assert ctx.rx.position is not None
    assert tuple(ctx.rx.position) == tuple(motion[-1].node.position)



def test_estimate_channelstate_calls_render_on_schedule(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(tmp_path, render_enabled=True, every_n=2)
    anchors = _anchors_with_z()
    motion = _motion_results()

    ctx = FakeCtx(rx=FakeRx(), txs=[FakeTx(a.id) for a in anchors])

    monkeypatch.setattr(cs, "_build_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(cs, "_solve_paths", lambda *args, **kwargs: object())
    monkeypatch.setattr(cs, "_evaluate_cfr", lambda **kwargs: ["X"])

    rendered_steps: list[int] = []

    def fake_render_if_enabled(*, ctx, cfg, step_idx, node_pos, paths):
        # emulate the real scheduling logic
        r = cfg.render
        if not r.enabled or r.every_n_steps <= 0:
            return None
        if step_idx % r.every_n_steps != 0:
            return None

        rendered_steps.append(step_idx)
        return cfg.scene.out_dir / f"scene_{step_idx}.png"

    monkeypatch.setattr(cs, "_render_if_enabled", fake_render_if_enabled)

    out = cs.estimate_channelstate(cfg=cfg, anchors=anchors, motion_results=motion)

    assert len(out) == 3
    assert rendered_steps == [0, 2]
    assert out[0].image is not None
    assert out[1].image is None
    assert out[2].image is not None



def test_estimate_channelstate_requires_anchor_z(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    motion = _motion_results()

    anchors = [AnchorConfig(id="A-01", x=0.0, y=0.0, z=None, size=0.2)]

    with pytest.raises(ValueError):
        cs.estimate_channelstate(cfg=cfg, anchors=anchors, motion_results=motion)

