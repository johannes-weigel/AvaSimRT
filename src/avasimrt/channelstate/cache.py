from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition
from avasimrt.result import Sample, AnchorReading, AntennaReading, ComplexReading
from avasimrt.math import mean_db_from_values
from .config import ChannelStateConfig

logger = logging.getLogger(__name__)


def _get_versions() -> dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    versions = {"python": sys.version.split()[0]}

    try:
        import sionna
        versions["sionna"] = getattr(sionna, "__version__", "unknown")
    except ImportError:
        pass

    versions["numpy"] = np.__version__

    return versions


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_trajectory(samples: list[Sample]) -> str:
    """Compute SHA256 hash over trajectory data (positions, timestamps)."""
    h = hashlib.sha256()
    for s in samples:
        h.update(str(s.timestamp).encode())
        h.update(str(s.node.position).encode())
        h.update(str(s.node.orientation).encode())
    return h.hexdigest()


def _hash_anchors(anchors: Sequence[ResolvedPosition]) -> str:
    """Compute SHA256 hash over anchor positions."""
    h = hashlib.sha256()
    for a in sorted(anchors, key=lambda x: x.id):
        h.update(a.id.encode())
        h.update(str(a.x).encode())
        h.update(str(a.y).encode())
        h.update(str(a.z).encode())
    return h.hexdigest()


@dataclass
class NodeCacheInfo:
    """Metadata about a single node's cached channelstate."""
    file: str
    file_hash_sha256: str
    trajectory_hash_sha256: str
    duration_s: float
    n_samples: int
    n_anchors: int
    n_antennas: int
    n_frequencies: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "file_hash_sha256": self.file_hash_sha256,
            "trajectory_hash_sha256": self.trajectory_hash_sha256,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
            "n_anchors": self.n_anchors,
            "n_antennas": self.n_antennas,
            "n_frequencies": self.n_frequencies,
        }


@dataclass
class CacheMetadata:
    """Complete metadata for a channelstate cache."""
    created_at: str
    total_duration_s: float  # full elapsed time including setup
    node_durations_sum_s: float  # sum of per-node durations (excludes setup)
    config: dict[str, Any]
    versions: dict[str, str]
    anchors_hash_sha256: str
    scene_xml_hash_sha256: str
    nodes: dict[str, NodeCacheInfo]

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "total_duration_s": self.total_duration_s,
            "node_durations_sum_s": self.node_durations_sum_s,
            "config": self.config,
            "versions": self.versions,
            "anchors_hash_sha256": self.anchors_hash_sha256,
            "scene_xml_hash_sha256": self.scene_xml_hash_sha256,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
        }


def save_channelstate(
    node_id: str,
    samples: list[Sample],
    out_dir: Path,
    *,
    trajectory_samples: list[Sample] | None = None,
    duration_s: float = 0.0,
) -> tuple[Path, NodeCacheInfo]:
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

    Returns:
        Tuple of (path to saved file, metadata about the saved file)
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

    # Compute hashes
    file_hash = _hash_file(out_path)
    traj_hash = _hash_trajectory(trajectory_samples if trajectory_samples else samples)

    info = NodeCacheInfo(
        file=out_path.name,
        file_hash_sha256=file_hash,
        trajectory_hash_sha256=traj_hash,
        duration_s=duration_s,
        n_samples=n_samples,
        n_anchors=n_anchors,
        n_antennas=n_antennas,
        n_frequencies=n_freqs,
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
    return out_path, info


def save_all_channelstates(
    results: dict[str, list[Sample]],
    out_dir: Path,
    *,
    cfg: ChannelStateConfig | None = None,
    trajectories: dict[str, list[Sample]] | None = None,
    anchors: Sequence[ResolvedPosition] | None = None,
    scene_xml: Path | None = None,
    durations: dict[str, float] | None = None,
    total_duration: float | None = None,
) -> dict[str, Path]:
    """
    Save channelstate results for all nodes and write metadata JSON.

    Args:
        results: Dict mapping node_id to list of samples with channelstate readings
        out_dir: Output directory for .npz files and metadata.json
        cfg: ChannelStateConfig used for computation (for metadata)
        trajectories: Dict of trajectory samples (for hashing)
        anchors: List of anchor positions (for hashing)
        scene_xml: Path to scene XML file (for hashing)
        durations: Dict mapping node_id to computation duration in seconds
        total_duration: Full elapsed time including setup (from ChannelStateResult)

    Returns:
        Dictionary mapping node_id to saved file paths
    """
    paths: dict[str, Path] = {}
    node_infos: dict[str, NodeCacheInfo] = {}
    durations = durations or {}

    node_durations_sum = sum(durations.values())

    for node_id, samples in results.items():
        traj_samples = trajectories.get(node_id) if trajectories else None
        duration = durations.get(node_id, 0.0)

        path, info = save_channelstate(
            node_id,
            samples,
            out_dir,
            trajectory_samples=traj_samples,
            duration_s=duration,
        )
        paths[node_id] = path
        node_infos[node_id] = info

    # Build and save metadata
    metadata = CacheMetadata(
        created_at=datetime.now(timezone.utc).isoformat(),
        total_duration_s=total_duration if total_duration is not None else node_durations_sum,
        node_durations_sum_s=node_durations_sum,
        config=cfg.to_dict() if cfg else {},
        versions=_get_versions(),
        anchors_hash_sha256=_hash_anchors(anchors) if anchors else "",
        scene_xml_hash_sha256=_hash_file(scene_xml) if scene_xml else "",
        nodes=node_infos,
    )

    metadata_path = out_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    logger.info("Saved channelstate metadata to %s", metadata_path)

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
