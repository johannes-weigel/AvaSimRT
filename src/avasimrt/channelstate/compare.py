"""
Compare two channelstate results (.npz files) and visualize differences.

Usage:
    python -m avasimrt.channelstate.compare file1.npz file2.npz
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


@dataclass
class ChannelStateData:
    """Loaded channelstate data from .npz file."""
    path: Path
    timestamps: np.ndarray
    frequencies: np.ndarray
    anchor_ids: np.ndarray
    antenna_labels: np.ndarray
    distances: np.ndarray
    real: np.ndarray
    imag: np.ndarray

    @property
    def node_name(self) -> str:
        return self.path.stem

    @property
    def shape(self) -> tuple[int, ...]:
        return self.real.shape

    @classmethod
    def load(cls, path: Path) -> "ChannelStateData":
        data = np.load(path, allow_pickle=True)
        return cls(
            path=path,
            timestamps=data["timestamps"],
            frequencies=data["frequencies"],
            anchor_ids=data["anchor_ids"],
            antenna_labels=data["antenna_labels"],
            distances=data["distances"],
            real=data["real"],
            imag=data["imag"],
        )


def compute_mean_db(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """
    Compute mean_db for each (sample, anchor, antenna) from real/imag arrays.

    Input shape: (N, A, P, F)
    Output shape: (N, A, P)
    """
    magnitudes = np.sqrt(real**2 + imag**2)
    magnitudes_safe = np.maximum(magnitudes, EPS)
    db = 20.0 * np.log10(magnitudes_safe)
    return db.mean(axis=-1)


@dataclass
class ComparisonResult:
    """Result of comparing two channelstate files."""
    match: bool
    mismatches: list[str]

    def __str__(self) -> str:
        if self.match:
            return "All metadata matches."
        return "Mismatches found:\n  - " + "\n  - ".join(self.mismatches)


def compare_metadata(a: ChannelStateData, b: ChannelStateData) -> ComparisonResult:
    """
    Compare metadata between two channelstate files.

    Checks: node_name, dimensions, timestamps, frequencies, anchor_ids,
            antenna_labels, distances.
    """
    mismatches: list[str] = []

    if a.node_name != b.node_name:
        mismatches.append(f"node_name: '{a.node_name}' vs '{b.node_name}'")

    if a.shape != b.shape:
        mismatches.append(f"shape: {a.shape} vs {b.shape}")
    else:
        if not np.allclose(a.timestamps, b.timestamps, rtol=1e-9, atol=1e-12):
            mismatches.append(f"timestamps differ (max delta: {np.abs(a.timestamps - b.timestamps).max():.2e})")

        if not np.allclose(a.frequencies, b.frequencies, rtol=1e-9, atol=1e-12):
            mismatches.append(f"frequencies differ (max delta: {np.abs(a.frequencies - b.frequencies).max():.2e})")

        if not np.array_equal(a.anchor_ids, b.anchor_ids):
            mismatches.append(f"anchor_ids: {list(a.anchor_ids)} vs {list(b.anchor_ids)}")

        if not np.array_equal(a.antenna_labels, b.antenna_labels):
            mismatches.append(f"antenna_labels: {list(a.antenna_labels)} vs {list(b.antenna_labels)}")

        if not np.allclose(a.distances, b.distances, rtol=1e-9, atol=1e-12):
            mismatches.append(f"distances differ (max delta: {np.abs(a.distances - b.distances).max():.2e})")

    return ComparisonResult(match=len(mismatches) == 0, mismatches=mismatches)


def plot_comparison(a: ChannelStateData, b: ChannelStateData) -> None:
    """
    Plot mean_db over time per anchor, showing all antennas for both files.

    Creates one subplot per anchor.
    """
    mean_db_a = compute_mean_db(a.real, a.imag)  # (N, A, P)
    mean_db_b = compute_mean_db(b.real, b.imag)

    n_anchors = len(a.anchor_ids)
    n_antennas = len(a.antenna_labels)

    fig, axes = plt.subplots(n_anchors, 1, figsize=(12, 4 * n_anchors), squeeze=False)
    fig.suptitle(f"Channelstate Comparison: {a.path.name} vs {b.path.name}", fontsize=14)

    colors_a = plt.cm.tab10(np.linspace(0, 1, n_antennas))
    colors_b = plt.cm.Set2(np.linspace(0, 1, n_antennas))

    for anchor_idx, anchor_id in enumerate(a.anchor_ids):
        ax = axes[anchor_idx, 0]
        ax.set_title(f"Anchor: {anchor_id}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean dB")

        for ant_idx, ant_label in enumerate(a.antenna_labels):
            ax.plot(
                a.timestamps,
                mean_db_a[:, anchor_idx, ant_idx],
                label=f"{a.path.stem} - {ant_label}",
                color=colors_a[ant_idx],
                linestyle="-",
                linewidth=1.5,
            )
            ax.plot(
                b.timestamps,
                mean_db_b[:, anchor_idx, ant_idx],
                label=f"{b.path.stem} - {ant_label}",
                color=colors_b[ant_idx],
                linestyle="--",
                linewidth=1.5,
            )

        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two channelstate .npz files and visualize differences."
    )
    parser.add_argument("file1", type=Path, help="First channelstate .npz file")
    parser.add_argument("file2", type=Path, help="Second channelstate .npz file")

    args = parser.parse_args(argv)

    if not args.file1.exists():
        print(f"ERROR: File not found: {args.file1}", file=sys.stderr)
        return 1
    if not args.file2.exists():
        print(f"ERROR: File not found: {args.file2}", file=sys.stderr)
        return 1

    print(f"Loading {args.file1}...")
    data_a = ChannelStateData.load(args.file1)
    print(f"  Shape: {data_a.shape}")
    print(f"  Anchors: {list(data_a.anchor_ids)}")
    print(f"  Antennas: {list(data_a.antenna_labels)}")

    print(f"\nLoading {args.file2}...")
    data_b = ChannelStateData.load(args.file2)
    print(f"  Shape: {data_b.shape}")
    print(f"  Anchors: {list(data_b.anchor_ids)}")
    print(f"  Antennas: {list(data_b.antenna_labels)}")

    print("\n--- Metadata Comparison ---")
    result = compare_metadata(data_a, data_b)
    print(result)

    if not result.match:
        print("\nWARNING: Metadata mismatch - plots may not be meaningful.", file=sys.stderr)

    print("\nGenerating comparison plots...")
    plot_comparison(data_a, data_b)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
