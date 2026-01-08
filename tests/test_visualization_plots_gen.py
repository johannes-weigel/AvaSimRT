from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from avasimrt.result import (
    AnchorReading,
    AntennaReading,
    ComplexReading,
    NodeSnapshot,
    Sample,
    SimResult,
)
from avasimrt.visualization.plots import (
    plot_amp_phase_for_reading,
    plot_mean_db_and_distance_over_time,
    save_all_visualizations,
)


def _make_sample(t: float, anchor_id: str, distance: float, mean_db: float) -> Sample:
    # Minimal node snapshot (values don't matter for these plots)
    node = NodeSnapshot(
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=(0.0, 0.0, 0.0),
    )

    freqs = [
        ComplexReading(freq=1.0, real=1.0, imag=0.0),
        ComplexReading(freq=2.0, real=0.0, imag=1.0),
    ]
    antenna = AntennaReading(label="A", mean_db=mean_db, frequencies=freqs)
    anchor = AnchorReading(anchor_id=anchor_id, distance=distance, values=[antenna])

    return Sample(timestamp=t, node=node, readings=[anchor], image=None)


def test_plot_mean_db_distance_saves_png(tmp_path: Path) -> None:
    samples = [
        _make_sample(0.0, "anchor-1", distance=10.0, mean_db=-40.0),
        _make_sample(1.0, "anchor-1", distance=12.0, mean_db=-42.0),
    ]

    plot_mean_db_and_distance_over_time(samples, show=False, save_dir=tmp_path)

    out = tmp_path / "mean_db_distance_over_time.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_amp_phase_for_reading_saves_png(tmp_path: Path) -> None:
    sample = _make_sample(0.0, "anchor-2", distance=5.0, mean_db=-30.0)
    assert sample.readings is not None
    reading = sample.readings[0]

    plot_amp_phase_for_reading(
        reading,
        graphs="both",
        prefix="t0000_",
        show=False,
        save_dir=tmp_path,
        amp_in_db=True,
    )

    out = tmp_path / "t0000_anchor-2_amp_phase_freq_ampdb.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_all_visualizations_creates_multiple_files(tmp_path: Path) -> None:
    samples = [
        _make_sample(0.0, "anchor-1", distance=10.0, mean_db=-40.0),
        _make_sample(1.0, "anchor-1", distance=11.0, mean_db=-41.0),
    ]

    save_all_visualizations(samples, out_dir=tmp_path)

    # overview
    overview = tmp_path / "mean_db_distance_over_time.png"
    assert overview.exists()
    assert overview.stat().st_size > 0

    # detail for each step (at least one)
    detail_files = sorted(tmp_path.glob("t*_anchor-1_amp_phase_freq_ampdb.png"))
    assert len(detail_files) >= 1
    assert all(p.stat().st_size > 0 for p in detail_files)
