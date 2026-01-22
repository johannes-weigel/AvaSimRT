from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from avasimrt.channelstate.simulation import (
    _build_context,
    _evaluate_cfr
)

from avasimrt.math import mean_amp_in_db_from_cfr, distance

@pytest.mark.presentation
def test_closer_transmitter_has_higher_amplitude(
    presentation_output: Path, 
    presentation_config
):
    """
    Setup:
        - Empty scene (no terrain, no obstacles)
        - TX1 at (-1, -8, 0) - distance ~8.06m from origin
        - TX2 at (1, -10, 0) - distance ~10.05m from origin
        - RX at (0, 0, 0)

    Expected:
        TX1 amplitude > TX2 amplitude (closer = stronger signal)
    """

    out_dir = presentation_output / "sionna_distance_amplitude"
    out_dir.mkdir(exist_ok=True)

    tx1_pos = (-1.0, -8.0, 0.0)
    tx2_pos = (1.0, -10.0, 0.0)
    txs = [("tx_close", tx1_pos, 1.0),
           ("tx_far",   tx2_pos, 1.0)]

    ctx = _build_context(anchors=txs, 
                         scene_src=None,
                         freq_center=None,
                         bandwidth=None,
                         snow=None,
                         reflection_depth=3,
                         seed=None)
    
    # should be default, but validate to guard against changes
    rx_pos = (0.0, 0.0, 0.0)
    actual_pos = (ctx.rx.position.x, ctx.rx.position.y, ctx.rx.position.z)
    assert actual_pos == rx_pos


    dist1 = distance(rx_pos, tx1_pos)
    dist2 = distance(rx_pos, tx2_pos)

    paths = ctx.solve_paths()

    ctx.render_to_file(paths, 
                       origin=(0, 0, 30), 
                       target=(0, -5, 0), 
                       file_path=out_dir / "scene_top_view.png",
                       resolution=presentation_config.resolution)
    
    cfr, dists =_evaluate_cfr(paths, freqs=0, anchors=txs, node_pos=rx_pos)

    mean_amp_db_close = mean_amp_in_db_from_cfr(cfr[0])  # tx_close
    mean_amp_db_far = mean_amp_in_db_from_cfr(cfr[1])    # tx_far

    # Plot comparison
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = [f"TX Close\n({dist1:.1f}m)", f"TX Far\n({dist2:.1f}m)"]
    values = [mean_amp_db_close, mean_amp_db_far]
    bars = ax.bar(labels, values)
    ax.set_ylabel("Mean Amplitude (dB)")
    ax.set_title("Signal Amplitude: Close vs Far Transmitter")
    ax.bar_label(bars, fmt="%.1f dB", label_type="center")
    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_comparison.png", dpi=150)
    plt.close()

    # closer TX should have higher amplitude (less negative dB)
    assert mean_amp_db_close > mean_amp_db_far, (
        f"Expected closer TX to have higher amplitude. "
        f"Got: close={mean_amp_db_close:.2f} dB, far={mean_amp_db_far:.2f} dB"
    )

    # calculated distance is close to expected
    # KEEP: ensures correct behavior if distance calulation is improved later
    assert np.isclose(dists[0], dist1, rtol=1e-3), f"Expected {dists[0]} to be close to {dist1}"
    assert np.isclose(dists[1], dist2, rtol=1e-3), f"Expected {dists[1]} to be close to {dist2}"


@pytest.mark.presentation
def test_amplitude_vs_distance_plot(
    presentation_output: Path,
    presentation_config
):
    """
    Setup:
        - Empty scene (no terrain, no obstacles)
        - 100 transmitters at distances from 5m to 500m
        - RX at origin (0, 0, 0)

    Output:
        - Plot of mean amplitude (dB) vs distance
    """
    import matplotlib.pyplot as plt

    out_dir = presentation_output / "sionna_amplitude_vs_distance"
    out_dir.mkdir(exist_ok=True)

    rx_pos = (0.0, 0.0, 0.0)

    # Create 100 transmitters at varying distances (5m to 500m)
    n_transmitters = 100
    distances_m = np.linspace(5, 500, n_transmitters)

    # Place transmitters along negative y-axis
    txs = [(f"tx_{i}", (0.0, -float(d), 0.0), 1.0) for i, d in enumerate(distances_m)]

    ctx = _build_context(
        anchors=txs,
        scene_src=None,
        freq_center=None,
        bandwidth=None,
        snow=None,
        reflection_depth=3,
        seed=None
    )

    paths = ctx.solve_paths()
    cfr, dists = _evaluate_cfr(paths, freqs=0, anchors=txs, node_pos=rx_pos)

    # Calculate mean amplitude in dB for each transmitter
    mean_amp_db = np.array([mean_amp_in_db_from_cfr(cfr[i]) for i in range(n_transmitters)])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(dists, mean_amp_db, alpha=0.7, s=20)
    plt.xlabel("Distance (m)")
    plt.ylabel("Mean Amplitude (dB)")
    plt.title("Signal Amplitude vs Distance (Free Space)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "amplitude_vs_distance.png", dpi=150)
    plt.close()

    # Verify monotonic decrease (closer = stronger)
    corr = np.corrcoef(dists, mean_amp_db)[0, 1]
    assert corr < -0.85, f"Expected strong negative correlation, got {corr:.4f}"
