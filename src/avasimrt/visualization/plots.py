from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from avasimrt.result import AnchorReading, Sample, SimResult
from avasimrt.math import amps_to_db, magnitude, phase_deg

logger = logging.getLogger(__name__)


def _iter_anchor_readings(sample: Sample) -> list[AnchorReading]:
    return sample.readings or []


def plot_amp_phase_for_reading(
    reading: AnchorReading,
    *,
    graphs: Literal["amp", "phase", "both"] = "both",
    prefix: str = "",
    show: bool = True,
    save_dir: Path | None = None,
    amp_in_db: bool = True,
) -> None:
    """Plot amplitude and/or phase over frequency for a single anchor reading."""
    if not reading.values:
        logger.info("No antenna values for anchor %s, nothing to plot.", reading.anchor_id)
        return

    if graphs == "both":
        fig, (ax_amp, ax_phase) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    elif graphs == "amp":
        fig, ax_amp = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        ax_phase = None
    else:
        fig, ax_phase = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        ax_amp = None

    for antenna in reading.values:
        freqs = np.array([v.freq for v in antenna.frequencies], dtype=float)
        if freqs.size == 0:
            continue

        amps = np.array([magnitude(v) for v in antenna.frequencies], dtype=float)
        phases = np.array([phase_deg(v) for v in antenna.frequencies], dtype=float)

        amps_plot = amps_to_db(amps) if amp_in_db else amps

        phases_unwrapped = None
        if graphs in ("phase", "both"):
            phases_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phases)))

        if ax_amp is not None:
            ax_amp.plot(freqs, amps_plot, marker="o", linestyle="-", label=f"{antenna.label} antenna")

        if ax_phase is not None and phases_unwrapped is not None:
            ax_phase.plot(freqs, phases_unwrapped, marker="o", linestyle="-", label=f"{antenna.label} antenna")

    if ax_amp is not None:
        ax_amp.set_xlabel("Frequency [Hz]")
        ax_amp.set_ylabel("Amplitude [dB]" if amp_in_db else "Amplitude (linear)")
        ax_amp.set_title(f"Anchor {reading.anchor_id}: Amplitude / Frequency" + (" (dB)" if amp_in_db else ""))
        ax_amp.legend()

    if ax_phase is not None:
        ax_phase.set_xlabel("Frequency [Hz]")
        ax_phase.set_ylabel("Phase [deg]")
        ax_phase.set_title(f"Anchor {reading.anchor_id}: Phase / Frequency")
        ax_phase.legend()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix_amp = "_ampdb" if amp_in_db and ax_amp is not None else ""
        if graphs == "both":
            filename = f"{prefix}{reading.anchor_id}_amp_phase_freq{suffix_amp}.png"
        elif graphs == "amp":
            filename = f"{prefix}{reading.anchor_id}_amp_freq{suffix_amp}.png"
        else:
            filename = f"{prefix}{reading.anchor_id}_phase_freq.png"
        fig.savefig(save_dir / filename, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def show_anchor_over_time(samples: list[Sample], anchor_id: str) -> None:
    """Show two 3D plots per antenna for one anchor over all samples.

    Time axis is the sample index (0-based), not necessarily physical time.
    """
    per_antenna: dict[str, dict[str, list[float]]] = {}

    for step_idx, sample in enumerate(samples):
        reading = next((ar for ar in _iter_anchor_readings(sample) if ar.anchor_id == anchor_id), None)
        if reading is None or not reading.values:
            continue

        for antenna in reading.values:
            label = antenna.label or "antenna"
            ant_data = per_antenna.setdefault(label, {"t": [], "freq": [], "amp_db": [], "phase": []})

            for cval in antenna.frequencies:
                ant_data["t"].append(float(step_idx))
                ant_data["freq"].append(float(cval.freq))

                amp = magnitude(cval)
                ant_data["amp_db"].append(float(amps_to_db(np.array([amp]))[0]))

                ant_data["phase"].append(float(phase_deg(cval)))

    if not per_antenna:
        raise ValueError(f"No data found for anchor {anchor_id!r} over time.")

    for label, data in per_antenna.items():
        t = np.asarray(data["t"], dtype=float)
        freq = np.asarray(data["freq"], dtype=float)
        amp_db = np.asarray(data["amp_db"], dtype=float)
        phase_vals = np.asarray(data["phase"], dtype=float)

        if t.size == 0:
            continue

        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        ax_amp = fig.add_subplot(1, 2, 1, projection="3d")
        ax_phase = fig.add_subplot(1, 2, 2, projection="3d")

        ax_amp.scatter(t, freq, amp_db.tolist(), s=5)
        ax_amp.set_xlabel("Step index")
        ax_amp.set_ylabel("Frequency [Hz]")
        ax_amp.set_zlabel("Amplitude [dB]")
        ax_amp.set_title(f"Anchor {anchor_id}, {label}: Amplitude vs step/frequency")

        ax_phase.scatter(t, freq, phase_vals.tolist(), s=5)
        ax_phase.set_xlabel("Step index")
        ax_phase.set_ylabel("Frequency [Hz]")
        ax_phase.set_zlabel("Phase [deg]")
        ax_phase.set_title(f"Anchor {anchor_id}, {label}: Phase vs step/frequency")

        plt.show()


def plot_mean_db_and_distance_over_time(
    samples: list[Sample],
    *,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    """Plot mean_db (left axis) and distance (right axis) over time for all anchors."""
    per_anchor: dict[str, dict[str, list[float]]] = {}

    for sample in samples:
        t = float(sample.timestamp)
        for ar in _iter_anchor_readings(sample):
            mean_values = [ant.mean_db for ant in ar.values]
            if not mean_values:
                continue

            mean_db = float(np.mean(mean_values))
            dist = float(ar.distance)

            d = per_anchor.setdefault(ar.anchor_id, {"t": [], "mean_db": [], "dist": []})
            d["t"].append(t)
            d["mean_db"].append(mean_db)
            d["dist"].append(dist)

    if not per_anchor:
        logger.info("No per-anchor data available; skipping mean_db/distance plot.")
        return

    fig, ax_db = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    ax_dist = ax_db.twinx()

    color_list = rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    color_cycle = iter(color_list)

    for anchor_id, data in per_anchor.items():
        t = np.asarray(data["t"], dtype=float)
        mean_db_vals = np.asarray(data["mean_db"], dtype=float)
        dist_vals = np.asarray(data["dist"], dtype=float)

        if t.size == 0:
            continue

        try:
            base_color = next(color_cycle)
        except StopIteration:
            color_cycle = iter(color_list)
            base_color = next(color_cycle)

        ax_db.plot(t, mean_db_vals, marker="o", linestyle="-", label=f"{anchor_id} mean_db", color=base_color)
        ax_dist.plot(t, dist_vals, marker="x", linestyle="--", label=f"{anchor_id} distance", color=base_color, alpha=0.6)

    ax_db.set_xlabel("Time [s]")
    ax_db.set_ylabel("Mean amplitude [dB]")
    ax_dist.set_ylabel("Distance [m]")
    ax_db.set_title("Mean dB and distance over time for all anchors")

    handles_db, labels_db = ax_db.get_legend_handles_labels()
    handles_dist, labels_dist = ax_dist.get_legend_handles_labels()
    ax_db.legend(handles_db + handles_dist, labels_db + labels_dist, loc="upper right")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "mean_db_distance_over_time.png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_all_visualizations(sim_result: SimResult, out_dir: Path) -> None:
    """Save overview plot and per-sample per-anchor detail plots."""
    samples = sim_result.samples
    if not samples:
        logger.info("No samples given, skipping save_all_visualizations.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving overview mean_db/distance plot to %s", out_dir)
    plot_mean_db_and_distance_over_time(samples, show=False, save_dir=out_dir)

    logger.info("Saving per-anchor, per-step detail plots to %s", out_dir)
    for step_idx, sample in enumerate(samples):
        for ar in _iter_anchor_readings(sample):
            prefix = f"t{step_idx:04d}_"
            plot_amp_phase_for_reading(
                reading=ar,
                graphs="both",
                prefix=prefix,
                show=False,
                save_dir=out_dir,
                amp_in_db=True,
            )
