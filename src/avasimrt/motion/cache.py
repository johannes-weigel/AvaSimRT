from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from avasimrt.motion.result import NodeSnapshot
from avasimrt.result import Sample

logger = logging.getLogger(__name__)


def save_trajectory(node_id: str, samples: list[Sample], out_dir: Path) -> Path:
    """
    File structure:
        - timestamps: (N,) float64
        - positions: (N, 3) float64
        - orientations: (N, 4) float64
        - linear_velocities: (N, 3) float64
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{node_id}.npz"

    n = len(samples)
    timestamps = np.empty(n, dtype=np.float64)
    positions = np.empty((n, 3), dtype=np.float64)
    orientations = np.empty((n, 4), dtype=np.float64)
    linear_velocities = np.empty((n, 3), dtype=np.float64)

    for i, s in enumerate(samples):
        timestamps[i] = s.timestamp
        positions[i] = s.node.position
        orientations[i] = s.node.orientation
        linear_velocities[i] = s.node.linear_velocity

    np.savez_compressed(
        out_path,
        timestamps=timestamps,
        positions=positions,
        orientations=orientations,
        linear_velocities=linear_velocities,
    )

    logger.info("Saved trajectory for '%s' (%d samples) to %s", node_id, n, out_path)
    return out_path


def save_all_trajectories(
    results: dict[str, list[Sample]], out_dir: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for node_id, samples in results.items():
        paths[node_id] = save_trajectory(node_id, samples, out_dir)
    return paths


def load_trajectory(path: Path) -> list[Sample]:
    data = np.load(path)

    timestamps = data["timestamps"]
    positions = data["positions"]
    orientations = data["orientations"]
    linear_velocities = data["linear_velocities"]

    samples: list[Sample] = []
    for i in range(len(timestamps)):
        samples.append(
            Sample(
                timestamp=float(timestamps[i]),
                node=NodeSnapshot(
                    position=tuple(positions[i]),
                    orientation=tuple(orientations[i]),
                    linear_velocity=tuple(linear_velocities[i]),
                ),
            )
        )

    logger.info("Loaded trajectory from %s (%d samples)", path, len(samples))
    return samples


def load_all_trajectories(cache_dir: Path) -> dict[str, list[Sample]]:
    if not cache_dir.is_dir():
        raise ValueError(f"Trajectory cache directory does not exist: {cache_dir}")

    results: dict[str, list[Sample]] = {}
    for npz_path in sorted(cache_dir.glob("*.npz")):
        node_id = npz_path.stem
        results[node_id] = load_trajectory(npz_path)

    logger.info("Loaded %d trajectories from %s", len(results), cache_dir)
    return results
