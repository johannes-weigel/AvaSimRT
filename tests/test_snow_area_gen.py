from __future__ import annotations

import numpy as np
from pathlib import Path

from avasimrt.motion.result import NodeSnapshot
from avasimrt.preprocessing.result import ResolvedPosition
from avasimrt.channelstate.snow import extract_relevant_heightmap

def test_with_nordkette_data(examples: Path) -> None:
    """Test the alpha shape extraction with nordkette example data and visualize in 3D."""
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    assets_path = examples / "seegrube" / "assets"

    heightmap = np.load(assets_path / "heightmap.npy")

    with open(assets_path / "positions_resolved.json") as f:
        positions_data = json.load(f)

    anchors = [
        ResolvedPosition(
            id=a["id"],
            x=a["x"],
            y=a["y"],
            z=a["z"],
            z_terrain=a["z_terrain"],
            size=a["size"],
        )
        for a in positions_data["anchors"]
    ]

    # Use nodes as single-snapshot trajectories for testing
    nodes: list[list[NodeSnapshot]] = [
        [
            NodeSnapshot(
                position=(n["x"], n["y"], n["z"]),
                orientation=(0.0, 0.0, 0.0, 1.0),
                linear_velocity=(0.0, 0.0, 0.0),
                size=n["size"],
            )
        ]
        for n in positions_data["nodes"]
    ]

    filtered_heightmap, mask = extract_relevant_heightmap(
        heightmap, anchors, nodes, alpha=0.01, margin=50.0
    )

    print(f"Original heightmap shape: {heightmap.shape}")
    print(f"Filtered points: {len(filtered_heightmap)}")
    print(f"Mask shape: {mask.shape}, True count: {np.sum(mask)}")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        filtered_heightmap[:, 0],
        filtered_heightmap[:, 1],
        filtered_heightmap[:, 2],
        c=filtered_heightmap[:, 2],
        cmap="terrain",
        s=1,
        alpha=0.6,
    )

    anchor_x = [a.x for a in anchors]
    anchor_y = [a.y for a in anchors]
    anchor_z = [a.z for a in anchors]
    ax.scatter(anchor_x, anchor_y, anchor_z, c="red", s=100, marker="^", label="Anchors")

    for trajectory in nodes:
        for snapshot in trajectory:
            ax.scatter(
                snapshot.position[0],
                snapshot.position[1],
                snapshot.position[2],
                c="blue",
                s=100,
                marker="o",
            )
    ax.scatter([], [], [], c="blue", s=100, marker="o", label="Nodes")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Alpha Shape Filtered Heightmap")
    ax.legend()

    plt.tight_layout()

    output_path = assets_path / "alpha_shape_heightmap.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

