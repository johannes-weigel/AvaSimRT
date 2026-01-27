from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from avasimrt.channelstate.config import SnowConfig
from avasimrt.channelstate.simulation import (
    _build_context,
    _evaluate_cfr
)
from avasimrt.channelstate.snow import (
    create_scene_with_snow,
    prepare_snow_scene,
)
from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition

from avasimrt.math import mean_amp_in_db_from_cfr, distance


@pytest.mark.presentation
def test_snow(
    presentation_output: Path,
    examples: Path,
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

    out_dir = presentation_output / "sionna_snow_1"
    out_dir.mkdir(exist_ok=True)

    tx1_pos = (-1.0, -8.0, 0.0)
    tx2_pos = (1.0, -10.0, 0.0)
    txs = [("tx_close", tx1_pos, 1.0),
           ("tx_far",   tx2_pos, 1.0)]

    new = create_scene_with_snow(xml_path=examples / "seegrube/assets/scene.xml",
                                 meshes_dir= examples/"seegrube/assets/meshes",
                                 radius=1,
                                 positions=np.array([(0, 0, 0, 0), (0, 0, 0, 0), (400, 401, 402, 403)]))

    ctx = _build_context(anchors=txs,
                         scene_src=new,
                         freq_center=None,
                         bandwidth=None,
                         snow=None,
                         reflection_depth=3,
                         seed=None)

    print(ctx.scene.objects)


@dataclass
class TimingResult:
    resolution: float
    n_objects: int
    prepare_time: float
    load_time: float


@pytest.mark.presentation
def test_snow_performance(
    presentation_output: Path,
    examples: Path,
    presentation_config
):
    """Benchmark prepare_snow_scene and scene loading across different heightmap resolutions."""

    heighmaps = [
        #("heightmap-100.npy", 100),
        #("heightmap-50.npy", 50),
        ("heightmap-25.npy", 25),
        ("heightmap-10.npy", 10),
        ("heightmap-5.npy", 5),
        ("heightmap-4.npy", 4),
        ("heightmap-3.npy", 3),
        ("heightmap-2.npy", 2),
        ("heightmap-1.npy", 1),
        ("heightmap.npy", 0.5),
    ]

    out_dir = presentation_output / "sionna_snow_performance"
    out_dir.mkdir(exist_ok=True)

    scene_xml = examples / "seegrube/assets/scene.xml"

    cfg = SnowConfig(enabled=True, box_size=1.0, levels=1, margin=5.0)

    anchors = [
        ResolvedPosition(id="anchor1", x=-190, y=310, z=0, z_terrain=None, size=1),
        ResolvedPosition(id="anchor2", x=-150, y=135, z=0, z_terrain=None, size=1),
        ResolvedPosition(id="anchor3", x=-65, y=165, z=0, z_terrain=None, size=1),
        ResolvedPosition(id="anchor4", x=-85, y=352, z=0, z_terrain=None, size=1),
        ResolvedPosition(id="anchor5", x=-190, y=415, z=0, z_terrain=None, size=1)
    ]

    nodes: list[list[NodeSnapshot]] = []

    txs = []

    results: list[TimingResult] = []

    for heightmap_file, resolution in heighmaps:
        heightmap = np.load(examples / "seegrube/assets" / heightmap_file)

        run_out_dir = out_dir / f"res_{resolution}"
        run_out_dir.mkdir(exist_ok=True)

        start = time.perf_counter()
        snow_scene_xml, n_objects = prepare_snow_scene(
            cfg=cfg,
            scene_xml=scene_xml,
            heightmap=heightmap,
            anchors=anchors,
            nodes=nodes,
            out_dir=run_out_dir,
        )
        prepare_time = time.perf_counter() - start

        start = time.perf_counter()
        ctx = _build_context(
            anchors=txs,
            scene_src=snow_scene_xml,
            freq_center=None,
            bandwidth=None,
            snow=None,
            reflection_depth=3,
            seed=None,
        )
        load_time = time.perf_counter() - start

        results.append(TimingResult(
            resolution=resolution,
            n_objects=n_objects,
            prepare_time=prepare_time,
            load_time=load_time,
        ))

        print(f"Resolution {resolution}: {n_objects} objects, prepare={prepare_time:.3f}s, load={load_time:.3f}s")

    # Generate incremental plots (one for each resolution including all previous)
    for i in range(len(results)):
        subset = results[:i + 1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        res_labels = [f"{r.resolution}×{r.resolution}m" for r in subset]
        x_pos = np.arange(len(subset))

        ax1.bar(x_pos - 0.2, [r.prepare_time for r in subset], 0.4, label='Prepare Time')
        ax1.bar(x_pos + 0.2, [r.load_time for r in subset], 0.4, label='Load Time')
        ax1.set_xlabel('Block Size')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Time vs Block Size')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(res_labels)
        ax1.legend()

        n_objects_list = [r.n_objects for r in subset]
        ax2.scatter(n_objects_list, [r.prepare_time for r in subset], label='Prepare Time', marker='o')
        ax2.scatter(n_objects_list, [r.load_time for r in subset], label='Load Time', marker='s')
        ax2.plot(n_objects_list, [r.prepare_time for r in subset], linestyle='--', alpha=0.5)
        ax2.plot(n_objects_list, [r.load_time for r in subset], linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Objects')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Time vs Number of Objects')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(out_dir / f"performance_{i + 1}.png", dpi=150)
        plt.close()

        # Prepare time only plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(x_pos, [r.prepare_time for r in subset], 0.4)
        ax1.set_xlabel('Block Size')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Prepare Time vs Block Size')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(res_labels)

        ax2.scatter(n_objects_list, [r.prepare_time for r in subset], marker='o')
        ax2.plot(n_objects_list, [r.prepare_time for r in subset], linestyle='--', alpha=0.5)
        ax2.set_xlabel('Number of Objects')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Prepare Time vs Number of Objects')

        plt.tight_layout()
        plt.savefig(out_dir / f"prepare_time_{i + 1}.png", dpi=150)
        plt.close()

        # Load time only plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.bar(x_pos, [r.load_time for r in subset], 0.4, color='tab:orange')
        ax1.set_xlabel('Block Size')
        ax1.set_ylabel('Time (s)')
        ax1.set_title('Load Time vs Block Size')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(res_labels)

        ax2.scatter(n_objects_list, [r.load_time for r in subset], marker='s', color='tab:orange')
        ax2.plot(n_objects_list, [r.load_time for r in subset], linestyle='--', alpha=0.5, color='tab:orange')
        ax2.set_xlabel('Number of Objects')
        ax2.set_ylabel('Time (s)')
        ax2.set_title('Load Time vs Number of Objects')

        plt.tight_layout()
        plt.savefig(out_dir / f"load_time_{i + 1}.png", dpi=150)
        plt.close()
