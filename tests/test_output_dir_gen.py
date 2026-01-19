from __future__ import annotations

from pathlib import Path

from avasimrt.app import run
from avasimrt.config import SimConfig


def test_run_returns_output_dir_as_path(tmp_path, monkeypatch) -> None:
    """
    The run should create <output>/<run_id> and return it as a Path.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")

    cfg = SimConfig(run_id="run-1", output=Path("output"), delete_existing=False, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is True
    assert isinstance(result.output_dir, Path)

    assert result.output_dir.exists()
    assert result.output_dir.is_dir()

    # default output base
    assert result.output_dir.parent.name == "output"
    assert result.output_dir.name == "run-1"


def test_run_fails_if_output_dir_not_empty_and_no_delete(tmp_path, monkeypatch) -> None:
    """
    If the output directory already exists and is not empty, the run must abort.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")

    run_dir = tmp_path / "output" / "run-2"
    run_dir.mkdir(parents=True)
    (run_dir / "existing.txt").write_text("data", encoding="utf-8")

    cfg = SimConfig(run_id="run-2", output=Path("output"), delete_existing=False, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is False
    assert result.output_dir is None
    assert result.message is not None
    assert "not empty" in result.message.lower()


def test_run_deletes_existing_output_dir_if_flag_set(tmp_path, monkeypatch) -> None:
    """
    If delete_existing is True, an existing non-empty directory is removed and recreated.
    The new run may only contain results.csv.
    """
    monkeypatch.chdir(tmp_path)
    
    scene_obj = tmp_path / "scene.obj"
    scene_obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    scene_xml = tmp_path / "scene.xml"
    scene_xml.write_text("<scene/>", encoding="utf-8")

    run_dir = tmp_path / "output" / "run-3"
    run_dir.mkdir(parents=True)
    (run_dir / "old.txt").write_text("old", encoding="utf-8")

    cfg = SimConfig(run_id="run-3", output=Path("output"), delete_existing=True, scene_xml=scene_xml, scene_obj=scene_obj)
    result = run(cfg)

    assert result.successful is True
    assert isinstance(result.output_dir, Path)

    out = result.output_dir
    assert out.exists()
    assert out.is_dir()

    entries = list(out.iterdir())

    # exepct 3 files
    assert len(entries) == 3


def test_heightmap_visualization(tmp_path, monkeypatch) -> None:
    """
    Generate a simple terrain mesh and verify the heightmap is correctly computed
    by creating a visual output.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from avasimrt.preprocessing.heights import generate_heightmap
    
    monkeypatch.chdir(tmp_path)
    
    # Create a simple sloped plane mesh: z increases with x
    # This creates a terrain that slopes from z=0 at x=-5 to z=5 at x=5
    scene_obj = tmp_path / "sloped_terrain.obj"
    vertices = [
        "v -5 -5 0",   # Bottom-left-back
        "v 5 -5 5",    # Bottom-right-back (higher)
        "v 5 5 5",     # Top-right-front (higher)
        "v -5 5 0",    # Top-left-front
    ]
    faces = [
        "f 1 2 3",     # First triangle
        "f 1 3 4",     # Second triangle
    ]
    scene_obj.write_text("\n".join(vertices + faces) + "\n", encoding="utf-8")
    
    # Generate heightmap
    heightmap, metadata = generate_heightmap(scene_obj, resolution=0.5)
    
    # Verify basic properties
    assert heightmap.shape[0] > 0
    assert heightmap.shape[1] > 0
    assert not np.all(np.isnan(heightmap))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Heightmap as heatmap
    x_coords = np.array(metadata['x_coords'])
    y_coords = np.array(metadata['y_coords'])
    im1 = ax1.imshow(
        heightmap.T,
        origin='lower',
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        cmap='terrain',
        aspect='auto'
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Heightmap (Top View)')
    plt.colorbar(im1, ax=ax1, label='Height (m)')
    
    # Plot 2: 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    ax2 = fig.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    ax2.plot_surface(X, Y, heightmap, cmap='terrain', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Height (m)')
    ax2.set_title('Heightmap (3D View)')
    
    # Save visualization
    output_file = tmp_path / "heightmap_visualization.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Verify output file exists
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    
    # Verify heightmap values match expected slope
    # The terrain slopes from z=0 to z=5 as x goes from -5 to 5
    # So z should approximately equal (x + 5) / 2
    valid_mask = ~np.isnan(heightmap)
    if np.any(valid_mask):
        # Check a few points along the x-axis
        mid_y_idx = heightmap.shape[1] // 2
        for i in range(heightmap.shape[0]):
            if not np.isnan(heightmap[i, mid_y_idx]):
                x_val = x_coords[i]
                z_val = heightmap[i, mid_y_idx]
                expected_z = (x_val + 5) / 2  # Linear slope from 0 to 5
                # Allow some tolerance for raycasting discretization
                assert abs(z_val - expected_z) < 0.5, f"At x={x_val:.2f}, expected z≈{expected_z:.2f}, got z={z_val:.2f}"
    
    print(f"\n✓ Heightmap visualization saved to: {output_file}")
    print(f"  Shape: {heightmap.shape}")
    print(f"  Z range: [{metadata['z_min']:.2f}, {metadata['z_max']:.2f}]")
    print(f"  Valid cells: {np.count_nonzero(valid_mask)}/{heightmap.size}")
    