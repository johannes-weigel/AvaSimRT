from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from avasimrt.result import (
    AnchorReading,
    AntennaReading,
    ComplexReading,
    NodeSnapshot,
    Sample,
    SimResult,
)
from avasimrt.visualization import interactive as vis_interactive


def _make_samples() -> list[Sample]:
    node = NodeSnapshot(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=(0.0, 0.0, 0.0),
    )
    freqs = [ComplexReading(freq=1.0, real=1.0, imag=0.0)]
    ant = AntennaReading(label="A", mean_db=-10.0, frequencies=freqs)
    anchor = AnchorReading(anchor_id="anchor-1", distance=1.0, values=[ant])

    return [
        Sample(timestamp=0.0, node=node, readings=[anchor], image=None),
        Sample(timestamp=1.0, node=node, readings=[anchor], image=None),
    ]


def test_handle_command_overview_calls_overview_plot(monkeypatch) -> None:
    samples = _make_samples()

    called = {"overview": 0}

    def fake_overview(*args, **kwargs) -> None:
        called["overview"] += 1

    monkeypatch.setattr(vis_interactive, "plot_mean_db_and_distance_over_time", fake_overview)

    vis_interactive.handle_visualization_command(samples, "@")
    assert called["overview"] == 1


def test_handle_command_anchor_at_step_calls_detail_plot(monkeypatch) -> None:
    samples = _make_samples()

    called = {"detail": 0}

    def fake_detail(*args, **kwargs) -> None:
        called["detail"] += 1

    monkeypatch.setattr(vis_interactive, "plot_amp_phase_for_reading", fake_detail)

    vis_interactive.handle_visualization_command(samples, "anchor-1@0")
    assert called["detail"] == 1


def test_handle_command_anchor_over_time_calls_3d_view(monkeypatch) -> None:
    samples = _make_samples()

    called = {"overtime": 0}

    def fake_overtime(*args, **kwargs) -> None:
        called["overtime"] += 1

    monkeypatch.setattr(vis_interactive, "show_anchor_over_time", fake_overtime)

    vis_interactive.handle_visualization_command(samples, "anchor-1@")
    assert called["overtime"] == 1


def test_handle_command_invalid_raises(samples=_make_samples()) -> None:
    with pytest.raises(ValueError):
        vis_interactive.handle_visualization_command(samples, "anchor-1")  # missing '@'

    with pytest.raises(ValueError):
        vis_interactive.handle_visualization_command(samples, "anchor-1@nope")  # bad step

    with pytest.raises(IndexError):
        vis_interactive.handle_visualization_command(samples, "anchor-1@999")  # out of range
