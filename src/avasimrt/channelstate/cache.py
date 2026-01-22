from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from avasimrt.motion.result import NodeSnapshot
from avasimrt.result import Sample, AnchorReading, AntennaReading, ComplexReading
from avasimrt.math import mean_db_from_values

logger = logging.getLogger(__name__)


def save_channelstate(node_id: str, samples: list[Sample], out_dir: Path) -> Path:
    """
    Save channelstate results for a single node to a compressed numpy file.

    File structure:
        - timestamps: (N,) float64
        - frequencies: (F,) float64
        - anchor_ids: (A,) str
        - antenna_labels: (P,) str
        - distances: (N, A) float64
        - real: (N, A, P, F) float64
        - imag: (N, A, P, F) float64

    Where:
        N = number of samples (time steps)
        A = number of anchors
        P = number of antennas (polarizations)
        F = number of frequencies
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}.npz"

    if not samples:
        raise ValueError(f"Cannot save empty channelstate for node '{node_id}'")

    first_readings = samples[0].readings
    if not first_readings:
        raise ValueError(f"First sample for node '{node_id}' has no readings")

    # Extract structure from first sample
    anchor_ids = [r.anchor_id for r in first_readings]
    antenna_labels = [v.label for v in first_readings[0].values]
    frequencies = np.array(
        [c.freq for c in first_readings[0].values[0].frequencies], dtype=np.float64
    )

    n_samples = len(samples)
    n_anchors = len(anchor_ids)
    n_antennas = len(antenna_labels)
    n_freqs = len(frequencies)

    # Pre-allocate arrays
    timestamps = np.empty(n_samples, dtype=np.float64)
    distances = np.empty((n_samples, n_anchors), dtype=np.float64)
    real = np.empty((n_samples, n_anchors, n_antennas, n_freqs), dtype=np.float64)
    imag = np.empty((n_samples, n_anchors, n_antennas, n_freqs), dtype=np.float64)

    for i, s in enumerate(samples):
        timestamps[i] = s.timestamp

        if s.readings is None:
            raise ValueError(f"Sample {i} for node '{node_id}' has no readings")

        for a, anchor_reading in enumerate(s.readings):
            distances[i, a] = anchor_reading.distance

            for p, antenna_reading in enumerate(anchor_reading.values):
                for f, freq_reading in enumerate(antenna_reading.frequencies):
                    real[i, a, p, f] = freq_reading.real
                    imag[i, a, p, f] = freq_reading.imag

    np.savez_compressed(
        out_path,
        timestamps=timestamps,
        frequencies=frequencies,
        anchor_ids=np.array(anchor_ids, dtype=object),
        antenna_labels=np.array(antenna_labels, dtype=object),
        distances=distances,
        real=real,
        imag=imag,
    )

    logger.info(
        "Saved channelstate for '%s' (%d samples, %d anchors, %d antennas, %d freqs) to %s",
        node_id,
        n_samples,
        n_anchors,
        n_antennas,
        n_freqs,
        out_path,
    )
    return out_path


def save_all_channelstates(
    results: dict[str, list[Sample]], out_dir: Path
) -> dict[str, Path]:
    """Save channelstate results for all nodes."""
    paths: dict[str, Path] = {}
    for node_id, samples in results.items():
        paths[node_id] = save_channelstate(node_id, samples, out_dir)
    return paths


def load_channelstate(path: Path, node_snapshots: list[NodeSnapshot] | None = None) -> list[Sample]:
    """
    Load channelstate results from a compressed numpy file.

    Args:
        path: Path to the .npz file
        node_snapshots: Optional list of NodeSnapshot objects to attach to samples.
                       If None, samples will have a placeholder NodeSnapshot.

    Returns:
        List of Sample objects with reconstructed readings
    """
    data = np.load(path, allow_pickle=True)

    timestamps = data["timestamps"]
    frequencies = data["frequencies"]
    anchor_ids = data["anchor_ids"]
    antenna_labels = data["antenna_labels"]
    distances = data["distances"]
    real = data["real"]
    imag = data["imag"]

    n_samples = len(timestamps)

    samples: list[Sample] = []
    for i in range(n_samples):
        readings: list[AnchorReading] = []

        for a, anchor_id in enumerate(anchor_ids):
            antenna_values: list[AntennaReading] = []

            for p, antenna_label in enumerate(antenna_labels):
                freq_readings = [
                    ComplexReading(
                        freq=float(frequencies[f]),
                        real=float(real[i, a, p, f]),
                        imag=float(imag[i, a, p, f]),
                    )
                    for f in range(len(frequencies))
                ]

                antenna_values.append(
                    AntennaReading(
                        label=str(antenna_label),
                        mean_db=mean_db_from_values(freq_readings),
                        frequencies=freq_readings,
                    )
                )

            readings.append(
                AnchorReading(
                    anchor_id=str(anchor_id),
                    distance=float(distances[i, a]),
                    values=antenna_values,
                )
            )

        # Use provided node snapshot or create placeholder
        if node_snapshots is not None and i < len(node_snapshots):
            node = node_snapshots[i]
        else:
            node = NodeSnapshot(
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
                linear_velocity=(0.0, 0.0, 0.0),
                size=0.0,
            )

        samples.append(
            Sample(
                timestamp=float(timestamps[i]),
                node=node,
                readings=readings,
            )
        )

    logger.info("Loaded channelstate from %s (%d samples)", path, len(samples))
    return samples


def load_all_channelstates(
    cache_dir: Path, trajectories: dict[str, list[Sample]] | None = None
) -> dict[str, list[Sample]]:
    """
    Load channelstate results for all nodes from a cache directory.

    Args:
        cache_dir: Directory containing .npz files
        trajectories: Optional dict of trajectories to extract NodeSnapshots from.
                     Keys are node IDs, values are lists of Samples with node data.

    Returns:
        Dictionary mapping node_id to list of samples with channel state readings
    """
    if not cache_dir.is_dir():
        raise ValueError(f"Channelstate cache directory does not exist: {cache_dir}")

    results: dict[str, list[Sample]] = {}
    for npz_path in sorted(cache_dir.glob("*.npz")):
        node_id = npz_path.stem

        # Extract node snapshots from trajectories if available
        node_snapshots: list[NodeSnapshot] | None = None
        if trajectories is not None and node_id in trajectories:
            node_snapshots = [s.node for s in trajectories[node_id]]

        results[node_id] = load_channelstate(npz_path, node_snapshots)

    logger.info("Loaded %d channelstates from %s", len(results), cache_dir)
    return results
